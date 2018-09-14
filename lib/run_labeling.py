#!/usr/bin/env python

import argparse
from astropy.io import fits
from collections import Iterable, defaultdict
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
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.max()
    return ['background-color: #DC143C' if v else '' for v in is_max]

def highlight_min(s):
    is_min = s == s.min()
    return ['background-color: #1E90FF' if v else '' for v in is_min]

def low_outliers(s):
    med = s.median()
    std = s.std()
    print(std)
    flags = s < med - 2.5*std
    return ['background-color: #87CEEB' if a else '' for a in flags]

def high_outliers(s):
    med = s.median()
    std = s.std()
    print(std)
    flags = s > med + 2.5*std
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
                                       'electron_deposition'])
         .apply(low_outliers, subset=['shape [pix]',
                                            'size [pix]',
                                            'electron_deposition'])
         .apply(highlight_max, subset=['shape [pix]',
                                                    'size [pix]',
                                                    'electron_deposition'])
          .apply(highlight_min, subset=['shape [pix]',
                                        'size [pix]',
                                        'electron_deposition'])

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


def get_metadata(fname):
    meta = GenerateMetadata(fname)
    # Get image specific data (date-obs, expstart, expend, etc..)
    meta.get_image_data()
    # Get pointing information (WCS)
    meta.get_wcs_info()
    # Get HST specific data (latitude, longitude, altitude)
    meta.get_observatory_info()
    return meta.metadata


def write_out(fname, fout, data, grp, subgrp, update=False):
    print('Output filename: {}'.format(fout))
    print('HDF5 structure: /{}/{}'.format(grp, subgrp))
    with h5py.File(fout,'a', libver='latest') as f:
        print('/{}/{}'.format(grp, subgrp))
        grp = f['/{}/{}'.format(grp,subgrp)]
        subgrps = list(grp.keys())
        metadata = get_metadata(fname)
        # No need to save these statistics
        # if subgrp in ['sizes', 'shapes']:
        #     try:
        #         metadata['mean'] = np.nanmean(data[1])
        #     except Exception as e:
        #         metadata['mean'] = np.nan
        #     try:
        #         metadata['std'] = np.nanstd(data[1])
        #     except Exception as e:
        #         metadata['std'] = np.nan
        #     try:
        #         metadata['max'] = np.nanmax(data[1])
        #     except Exception as e:
        #          metadata['max'] = np.nan
        #     try:
        #         metadata['min'] = np.nanmin(data[1])
        #     except Exception as e:
        #         metadata['min'] = np.nan
        #
        # elif subgrp in ['cr_affected_pixels']:
        #     # these statistics can't be computed on a per image basis
        #     metadata['number_affected'] = len(data)
        if os.path.basename(fname) in subgrps and update:
            dset = grp['{}'.format(os.path.basename(fname))]
            if isinstance(data, Iterable):
                try:
                    dset[()] = data
                except Exception as e:
                    pass
            else:
                try:    
                    dset[...] = data
                except Exception as e:
                    pass
        if os.path.basename(fname) in subgrps:
            try:
                dset = grp.create_dataset(name='{}'.format(os.path.basename(fname)),
                                      shape=data.shape, data=data, dtype=np.float64)
            except Exception as e:
                pass
        else:
             dset = grp.create_dataset(name='{}'.format(os.path.basename(fname)),
                                      shape=data.shape, data=data, dtype=np.float64)
        for (key, val) in metadata.items():
            print(key, val)
            dset.attrs[key] = val


def format_data(data):
    keys = list(data.keys())
    values = list(data.values())
    data_out = np.asarray([keys, values])
    return data_out


def initialize_hdf5(instr, instr_cfg, subgrp_names):
    flist = instr_cfg['hdf5_files']
    for subgrp, f in zip(subgrp_names, flist):
        if not os.path.isdir(os.path.dirname(f)):
            os.mkdir(os.path.dirname(f))
        print('File structure: /{}/{}'.format(instr, subgrp))
        with h5py.File(f,'w') as fobj:
            grp = fobj.create_group(instr)
            subgrp = grp.create_group(subgrp)


def write_out_errors(fname, imgs ):
    print(imgs)
    with open(fname,'a+') as fobj:
        for img_name in imgs:
            fobj.write('{}\n'.format(img_name))


def find_files_to_download(instr):
    finder = FindData(instr)
    finder.get_date_ranges()
    return finder


def process_dataset(instr, flist):
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


def analyze_data(flist, instr, start, subgrp_names):
    num_cr_per_anneal = 0
    prefix = instr.split('_')[0]
    data_for_email = defaultdict(list)
    for f in flist:
        start_time = time.time()
        label_obj = CosmicRayLabel(f)

        label_obj.get_data()
        label_obj.get_label()
        stats_obj = ComputeStats(f, label_obj.label)
        cr_affected, cr_rate, sizes, anisotropy, deposition = \
            stats_obj.compute_stats()

        # Compute some data to supply in an email
        num_cr_per_anneal += len(sizes.values())
        sizes = format_data(sizes)
        anisotropy = format_data(anisotropy)
        data_for_email['filename'].append(os.path.basename(f))
        try:
            data_for_email['size [pix]'].append(np.nanmean(sizes[1]))
        except ValueError as e:
            data_for_email['size [pix]'].append(np.nan)
        try:
            data_for_email['shape [pix]'].append(np.nanmean(anisotropy[1]))
        except ValueError as e:
            data_for_email['shape [pix]'].append(np.nan)
        try:
            data_for_email['electron_deposition'].append(np.nanmedian(deposition[1]))
        except ValueError as e:
            data_for_email['electron_deposition'].append(np.nan)

        hdr = fits.getheader(f)
        if 'date-obs' in hdr:
            data_for_email['date-obs'].append(hdr['date-obs']+' '+
                                              hdr['time-obs'])
        elif 'tdateobs' in hdr:
            print(hdr['tdateobs'])
            data_for_email['date-obs'].append(hdr['tdateobs']+' '
                                                              +hdr['ttimeobs'])
        if 'exptime' in hdr:
            data_for_email['exptime'].append(hdr['exptime'])
        elif 'TEXPTIME' in hdr:
            data_for_email['exptime'].append(hdr['texptime'])
        else:
            data_for_email['exptime'].append('EXPTIME missing')
        # Package the files and data for writing out.
        if 'hrc' in instr.lower():
            path = prefix.upper()
            fs = instr.lower()
        else:
            path = prefix.upper()
            fs = prefix.lower()
        fout = [
            './../data/{}/{}_cr_affected_pixels.hdf5'.format(path,
                                                             fs),
            './../data/{}/{}_cr_shapes.hdf5'.format(path,
                                                 fs),
            './../data/{}/{}_cr_sizes.hdf5'.format(path,
                                                fs),
            './../data/{}/{}_cr_rate.hdf5'.format(path,
                                                  fs),
            './../data/{}/{}_cr_deposition.hdf5'.format(path,
                                                        fs)
        ]
        data_out = [
            cr_affected,
            anisotropy,
            sizes,
            cr_rate,
            deposition
        ]
        for (fname, data, subgrp) in zip(fout, data_out, subgrp_names):
            write_out(f,
                  fout=fname,
                  data=data,
                  grp=instr,
                  subgrp=subgrp,
                  update=True
                  )
        end_time = time.time()
        data_for_email['processing_time [min]'].append((end_time -
                                                        start_time)/60)
    gif_file = generate_gif(flist=flist, start_date=start, instr=instr)
    return gif_file, data_for_email


def clean_files(instr):
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
    with open('./../CONFIG/processed_dates_{}.txt'.format(instr),'a+') as fobj:
        fobj.write('{} {}\n'.format(start.iso, stop.iso))


def read_processed_ranges(instr):
    try:
        with open('./../CONFIG/processed_dates_{}.txt'.format(instr),'r') as fobj:
            lines = fobj.readlines()
            dates = [line.strip('\n') for line in lines]
    except FileNotFoundError as e:
        return []
    return dates


def main(instr, initialize):
    with open('./../CONFIG/pipeline_config.yaml', 'r') as fobj:
        cfg = yaml.load(fobj)

    if initialize:
        initialize_hdf5(instr, cfg[instr], cfg['subgrp_names'])
    finder = find_files_to_download(instr)
    search_pattern = cfg[instr]['search_pattern'][0]
    analyzed_dates = read_processed_ranges(instr)
    for (start, stop) in finder.dates:
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
            gif_file, data_for_email =\
                analyze_data(f_to_analyze, instr, start, cfg['subgrp_names'])
            subj = 'Finished analyzing darks from' \
                   ' {} to {}'.format(start.datetime.date(),
                                      stop.datetime.date())
            SendEmail(subj, data_for_email, gif_file, gif=False)
            write_processed_ranges(start, stop, instr)
        clean_files(instr)


if __name__ == '__main__':
    args = parser.parse_args()
    instr = args.instr.upper()
    main(args.instr.upper(), args.initialize)
    # main('ACS_WFC', True)
