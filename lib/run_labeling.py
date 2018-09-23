#!/usr/bin/env python

import argparse
from astropy.io import fits
from dask.distributed import Client
from collections import defaultdict
from email.message import EmailMessage
from email.headerregistry import Address
from email.utils import make_msgid
import glob
import h5py
import numpy as np
import os
import pandas as pd
import time

import smtplib
import shutil
import sys
import yaml
sys.path.append('/user/nmiles/animated_fits/lib')

# local imports
from mkAnimation import AnimationObj
from CosmicRayLabel import CosmicRayLabel
from ComputeStats import ComputeStats
from FindData import FindData
from ProcessData import ProcessData
from GenerateMetadata import GenerateMetadata

enocding_file = '/grp/hst/acs7/nmiles/cr_repo/lib/encoding_errors.txt'

parser = argparse.ArgumentParser()

parser.add_argument('-instr',
                    default='acs_wfc',
                    help='HST instrument to process (acs_wfc, '
                         'wfc3_uvis, stis_ccd, acs_hrc)')
parser.add_argument('-initialize',
                    action='store_true',
                    default=False,
                    help='Use this flag if this is the first time the '
                         'pipeline has been run. It will ensure all HDF5 files '
                         'are properly initialized with the correct structure.')


def email_styling():
    # Set CSS properties for th elements in dataframe
    th_props = [
        ('font-size', '14px'),
        ('text-align', 'center'),
        ('font-weight', 'bold'),
        ('color', 'black'),
        ('background-color', 'LightGray'),
        ('border', '1px solid black')
    ]

    # Set CSS properties for td elements in dataframe
    td_props = [
        ('font-size', '14px'),
        ('text-align', 'center'),
        ('border', '1px solid black')
    ]

    # Set table styles
    styles = [
        dict(selector="th", props=th_props),
        dict(selector="td", props=td_props)
    ]
    return styles

def highlight_max(s):
    """ highlight the max value in the series with dark red

    Parameters
    ----------
    s

    Returns
    -------

    """
    is_max = s == s.max()
    return ['background-color: #DC143C' if v else '' for v in is_max]

def highlight_min(s):
    """ Highlight the min value in the series with dark blue

    Parameters
    ----------
    s

    Returns
    -------

    """
    is_min = s == s.min()
    return ['background-color: #1E90FF' if v else '' for v in is_min]

def low_outliers(s):
    """ Highlight outliers below the mean with light blue

    Parameters
    ----------
    s : pd.Series with

    Returns
    -------

    """
    med = s.median()
    std = s.std()
    print(std)
    flags = s < med - 1.25*std
    return ['background-color: #87CEEB' if a else '' for a in flags]

def high_outliers(s):
    """ Highlight outliers above the mean with light red

    Parameters
    ----------
    s

    Returns
    -------

    """
    med = s.median()
    std = s.std()
    print(std)
    flags = s > med + 1.25*std
    return ['background-color: #CD5C5CC' if a else '' for a in flags]

def SendEmail(toSubj, data_for_email, gif_file, gif=False):
    """Send out an html markup email with an embedded gif and table

    Parameters
    ----------
    toSubj: email subject line
    data_for_email: data to render into an html table
    gif_file:

    Returns
    -------

    """
    css = email_styling()
    df = pd.DataFrame(data_for_email, index=data_for_email['date-obs'])
    df.drop(columns='date-obs', inplace=True)
    df.sort_index(inplace=True)
    s = (df.style
         .apply(high_outliers, subset=['shape [pix]',
                                       'size [pix]',
                                       'electron_deposition',
                                       'CR count'])
         .apply(low_outliers, subset=['shape [pix]',
                                      'size [pix]',
                                      'electron_deposition',
                                      'CR count'])
         .apply(highlight_max, subset=['shape [pix]',
                                       'size [pix]',
                                       'electron_deposition',
                                       'CR count'])
          .apply(highlight_min, subset=['shape [pix]',
                                        'size [pix]',
                                        'electron_deposition',
                                        'CR count'])

          .set_properties(**{'text-align':'center'})
         .format({'shape [pix]':'{:.3f}','size [pix]':'{:.3f}'})
         .set_table_styles(css)
         )
    html_tb = s.render(index=False)
    msg = EmailMessage()
    msg['Subject'] = toSubj
    msg['From'] = Address('', 'nmiles', 'stsci.edu')
    msg['To'] = Address('', 'nmiles', 'stsci.edu')
    gif_cid = make_msgid()
    if gif:
        body_str = """
        <html>
            <head></head>
            <body>
                <p><b> All cosmic ray statistics reported are averages for 
                the entire image</b></p>
                {}
                <img src="cid:{}">
            </body>
        </html>
        """.format(html_tb, gif_cid[1:-1])
        msg.add_alternative(body_str, subtype='html')
    else:
        body_str = """
                <html>
                    <head></head>
                    <body>
                        <p><b> All cosmic ray statistics reported are averages for 
                        the entire image</b></p>
                        {}
                    </body>
                </html>
                """.format(html_tb)
        msg.add_alternative(body_str, subtype='html')
    if gif:
        with open(gif_file,'rb') as img:
            msg.get_payload()[0].add_related(img.read(), 'image', 'gif',
                                         cid=gif_cid)
    msg.add_alternative(body_str, subtype='html')

    with smtplib.SMTP('smtp.stsci.edu') as s:
        s.send_message(msg)


def get_metadata(imgname):
    """Extract header information to save with each dataset


    Parameters
    ----------
    imgname : path to image

    Returns
    -------

    """
    meta = GenerateMetadata(imgname)
    # Get image specific data (date-obs, expstart, expend, etc..)
    meta.get_image_data()
    # Get pointing information (WCS)
    meta.get_wcs_info()
    # Get HST specific data (latitude, longitude, altitude)
    meta.get_observatory_info()
    return meta.metadata


def write_out(dset_name, fout, data, grp, subgrp):
    """ Write out the data

    Parameters
    ----------
    dset_name : name of the dataset --> ipppssoot
    fout : HDF5 file we are writing to
    data : data to be written
    grp : instrument information (e.g. ACS_WFC, STIS_CCD, etc..)
    subgrp : statistics we are saving (e.g. sizes, shapes, deposition etc..)

    Returns
    -------

    """
    print('Output filename: {}'.format(fout))
    print('HDF5 structure: /{}/{}'.format(grp, subgrp))
    with h5py.File(fout,'a', libver='latest') as f:
        print('/{}/{}'.format(grp, subgrp))
        grp = f['/{}/{}'.format(grp,subgrp)]
        metadata = get_metadata(dset_name)
        dset = grp.create_dataset(name='{}'.format(os.path.basename(dset_name)),
                                      shape=data.shape,
                                      data=data,
                                      dtype=np.float64)
        for (key, val) in metadata.items():
            print(key, val)
            dset.attrs[key] = val


def initialize_hdf5(instr, instr_cfg, subgrp_names):
    """ Create the required hdf5 files that we will write to

    For each file, we create 4 copies _1.hdf5, _2.hdf5 ... This allows us
    to get around issues with writing large numbers of datasets to a
    single HDF5 files, which can cause significant slow down.


    Parameters
    ----------
    instr
    instr_cfg
    subgrp_names

    Returns
    -------

    """
    flist = instr_cfg['hdf5_files']
    new_flist = []
    for f in flist:
        i = 0
        while i < 4:
            fnew = f.replace('.hdf5','_{}.hdf5'.format(i+1))
            i+=1
            new_flist.append(fnew)
    i = 0
    for j, f in enumerate(new_flist):
        if not os.path.isdir(os.path.dirname(f)):
            os.mkdir(os.path.dirname(f))
        print('File structure: /{}/{}'.format(instr, subgrp_names[i]))
        with h5py.File(f,'w') as fobj:
            grp = fobj.create_group(instr)
            subgrp = grp.create_group(subgrp_names[i])

        if (j+1) % 4 == 0:
            i += 1

def write_out_errors(fname, imgs ):
    """ Write out problem files

    Parameters
    ----------
    fname : filename to write too
    imgs : images to write out

    Returns
    -------

    """
    with open(fname,'a+') as fobj:
        for img_name in imgs:
            print('Failed to process {}'.format(img_name))
            fobj.write('{}\n'.format(img_name))


def find_files_to_download(instr):
    """ Query mast to find observations to download

    Parameters
    ----------
    instr

    Returns
    -------
    finder : Class containing useful attributes for processing steps
    """
    finder = FindData(instr)
    finder.get_date_ranges()
    return finder


def process_dataset(instr, flist):
    """ Run cr crejection on the dataset

    Parameters
    ----------
    instr
    flist

    Returns
    -------
    failed : list of images that could not be CR rejected for various reasons
    """
    processor = ProcessData(instr, flist)
    # Sort the files by exposure time and chunk to smaller datasets
    processor.sort()

    processor.cr_reject()
    if 'failed' in processor.output.keys():
        f_out = './../crrejtab/{}/{}_failed_' \
                'to_process.txt'.format(instr.split('_')[0], instr)
        write_out_errors(f_out,
                         processor.output['failed'])
    failed = set(processor.output['failed'])
    return failed


def generate_gif(flist, start_date, instr):
    """ Use animated fits to generate a gif of the data; save it to disk

    Parameters
    ----------
    flist
    start_date
    instr

    Returns
    -------

    """
    prefix = instr.split('_')[0]
    path, suffix, x_center, y_center = None, None, None, None
    dx, dy = None, None
    fps=1.25
    ext = 1
    keyword='date-obs'
    scale=False
    save = './../crrejtab/{}/{}_{}.gif'.format(prefix.upper(),
                                               instr,
                                               start_date.datetime.date())
    ani = AnimationObj(path, suffix,
                       x_center, y_center,
                       dx, dy, ext, keyword,
                       scale, save, fps)
    ani.animate(flist=flist)
    return save

def analyze_file(f):
    """ Analyze the cosmic rays marked in the input file

    Parameters
    ----------
    f : input file

    Returns
    -------
    cr_affected : pixels affected
    cr_rate : cr incidence rate
    sizes : cr sizes
    shapes : cr shapes
    deposition : cr signal
    """
    start_time = time.time()
    label_obj = CosmicRayLabel(f)
    label_obj.get_data()
    label_obj.get_label()
    stats_obj = ComputeStats(f, label_obj.label)
    cr_affected, cr_rate, sizes, shapes, deposition = \
        stats_obj.compute_stats()
    end_time = time.time()
    processing_time = (end_time - start_time)/ 60
    return cr_affected, cr_rate, sizes, shapes, deposition, processing_time


def analyze_data(flist, instr, start, subgrp_names, i):
    """ Analyze the list of files that were successfully processed

    Parameters
    ----------
    flist : list of files to process
    instr : instr we are processing (e.g. ACS/WFC, ACS/HRC
    start : start date of month
    subgrp_names : subgrp_names for stats (sizes, shapes, etc.)
    i : integer correspoding which hdf5 file to write data too (1 - 4)

    Returns
    -------

    """
    run_start = time.time()
    prefix = instr.split('_')[0]
    data_for_email = defaultdict(list)
    cr_data = defaultdict(list)
    # Start the client to generate multiple works for analysis portion
    client = Client()
    results = client.map(analyze_file, flist)
    results = client.gather(results)
    # We are done with parallelization portion, so close to the client
    client.close()

    if 'hrc' in instr.lower():
        path = prefix.upper()
        fs = instr.lower()
    else:
        path = prefix.upper()
        fs = prefix.lower()

    fout = [
        './../data/{}/{}_cr_affected_pixels_{}.hdf5'.format(path,
                                                            fs,
                                                            i + 1),
        './../data/{}/{}_cr_rate_{}.hdf5'.format(path, fs, i + 1),
        './../data/{}/{}_cr_sizes_{}.hdf5'.format(path, fs, i + 1),
        './../data/{}/{}_cr_shapes_{}.hdf5'.format(path, fs, i + 1),
        './../data/{}/{}_cr_deposition_{}.hdf5'.format(path, fs, i + 1)
    ]

    # affected, rate, sizes, shapes, deposition
    for i, result in enumerate(results):
        # format the data for writing out
        # Keys are the file and subgrp combos (e.g acs_cr_sizes_1.hdf5, sizes)
        cr_data[(fout[0], subgrp_names[0])].append(result[0])
        cr_data[(fout[1], subgrp_names[1])].append(result[1])
        cr_data[(fout[2], subgrp_names[2])].append(result[2])
        cr_data[(fout[3], subgrp_names[3])].append(result[3])
        cr_data[(fout[4], subgrp_names[4])].append(result[4])

        # grab data for email notification
        data_for_email['processing_time [min]'].append(result[5])
        data_for_email['CR count'].append(len(result[2][1]))
        try:
            data_for_email['size [pix]'].append(np.nanmean(result[2][1]))
        except ValueError:
            data_for_email['size [pix]'].append(np.nan)
        try:
            data_for_email['shape [pix]'].append(np.nanmean(result[3][1]))
        except ValueError:
            data_for_email['shape [pix]'].append(np.nan)
        try:
            data_for_email['electron_deposition'].append(np.nanmedian(result[4][1]))
        except ValueError:
            data_for_email['electron_deposition'].append(np.nan)

        hdr = fits.getheader(flist[i])
        if 'date-obs' in hdr:
            data_for_email['date-obs'].append(hdr['date-obs'] + ' ' +
                                              hdr['time-obs'])
        elif 'tdateobs' in hdr:
            print(hdr['tdateobs'])
            data_for_email['date-obs'].append(hdr['tdateobs'] + ' '
                                              + hdr['ttimeobs'])
        if 'exptime' in hdr:
            data_for_email['exptime'].append(hdr['exptime'])
        elif 'TEXPTIME' in hdr:
            data_for_email['exptime'].append(hdr['texptime'])
        else:
            data_for_email['exptime'].append('EXPTIME missing')

    for j, key in enumerate(cr_data.keys()):
        for i, data in enumerate(cr_data[key]):
            write_out(dset_name=flist[i],
                      fout=key[0],
                      data=data,
                      grp=instr,
                      subgrp=key[1])


    run_stop = time.time()
    total_time = (run_stop - run_start)/60
    gif_file = generate_gif(flist=flist, start_date=start, instr=instr)
    return gif_file, data_for_email, total_time


def clean_files(instr):
    """ Clean up after processing

    This will delete the downloaded files, as well as the files produced by
    the CR rejection process (i.e. the combined image and all trailer files)

    Parameters
    ----------
    instr

    Returns
    -------

    """
    crjs = glob.glob('./tmp*')
    val = instr.split('_')[0]
    if not crjs:
        pass
    else:
        for a in crjs:
            os.remove(a)
        shutil.rmtree('./../crrejtab/{}/mastDownload'.format(val),
                      ignore_errors=True)


def write_processed_ranges(start, stop, instr):
    """ Write out the date range that was analyzed

    Once the pipeline is finished analyzing the chunk of data falling between
    the start and stop dates, write this range out to a file for the given
    instrument. If the pipeline breaks or hangs from some reason, this allows
    us to start it over again with attempting to reprocess already analyzed
    data.

    Parameters
    ----------
    start : start date
    stop : stop date
    instr : instrument currently being processed

    Returns
    -------

    """
    with open('./../CONFIG/processed_dates_{}.txt'.format(instr),'a+') as fobj:
        fobj.write('{} {}\n'.format(start.iso, stop.iso))


def read_processed_ranges(instr):
    """ Read in the processed date ranges

    This will be used to check if the date range in question has already been
    analyzed, if it has then it will be skipped and the next range will be
    tried.

    Parameters
    ----------
    instr

    Returns
    -------

    """
    try:
        with open('./../CONFIG/processed_dates_{}.txt'.format(instr),'r') as fobj:
            lines = fobj.readlines()
            dates = [line.strip('\n') for line in lines]
    except FileNotFoundError as e:
        return []
    return dates


def main(instr, initialize):
    """ Run the pipeline as whole

    1) Generate a range of dates to analyze
    2) If it hasn't been analyzed, query MAST for darks in that range
    3) Download all darks found
    4) Run CR rejection on all files
        - Combined images with similar apertures and exposure times
        - Images that fail processing will be excluded from further analysis
    5) Analyze the cosmic rays flagged in the processed images
    6) Save the results and send email notification

    Parameters
    ----------
    instr : Instrument to analyze (STIS_CCD, ACS_WFC, ACS_HRC, WFC3_UVIS, etc..)
    initialize :  Whether or not to create a new batch of HDF5 files

    Returns
    -------

    """
    with open('./../CONFIG/pipeline_config.yaml', 'r') as fobj:
        cfg = yaml.load(fobj)

    if initialize:
        initialize_hdf5(instr, cfg[instr], cfg['subgrp_names'])
    finder = find_files_to_download(instr)
    search_pattern = cfg[instr]['search_pattern'][0]
    analyzed_dates = read_processed_ranges(instr)
    date_chunks = np.array_split(finder.dates, 4)
    for i, chunk in enumerate(date_chunks):
        for (start, stop) in chunk:
            if '{} {}'.format(start.iso, stop.iso) in analyzed_dates:
                print('Already analyzed {} to {}'.format(start.iso, stop.iso))
                continue
            print('Analyzing data from {} to {}'.format(start.iso, stop.iso))
            finder.query(range=(start, stop))
            finder.download(start.datetime.date().isoformat())
            flist = glob.glob(search_pattern)
            failed = process_dataset(instr, flist)
            f_to_analyze = list(set(flist).difference(failed))
            if not f_to_analyze:
                print('No files to analyze, something happened with processing.')
            else:
                gif_file, data_for_email, total_time = \
                    analyze_data(f_to_analyze,
                                 instr,
                                 start,
                                 cfg['subgrp_names'],
                                 i)
                subj = 'Finished analyzing darks from {} to {}.' \
                       ' Total processing time {:.3f} minutes'.\
                    format(start.datetime.date(),
                           stop.datetime.date(),
                           total_time)
                SendEmail(subj, data_for_email, gif_file, gif=False)
                write_processed_ranges(start, stop, instr)
            clean_files(instr)


if __name__ == '__main__':
    args = parser.parse_args()
    instr = args.instr.upper()
    main(args.instr.upper(), args.initialize)
    # main('ACS_WFC', True)
