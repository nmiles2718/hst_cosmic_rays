#!/usr/bin/env python

import argparse
from astropy.io import fits
import boto3
from botocore.exceptions import ClientError
from collections import defaultdict
import dask
from email.message import EmailMessage
from email.headerregistry import Address
from email.utils import make_msgid
import glob
import h5py
import itertools
import numpy as np
import os
import pandas as pd
import smtplib
import shutil
import sys
import time
import yaml

sys.path.append('/user/nmiles/animated_fits/lib')
# local imports
from mkAnimation import AnimationObj
from CosmicRayLabel import CosmicRayLabel
from ComputeStats import ComputeStats
from FindData import FindData
from ProcessData import ProcessData
from process_IR import ProcessIR
from GenerateMetadata import GenerateMetadata
from scipy import ndimage

enocding_file = '/grp/hst/acs7/nmiles/cr_repo/lib/encoding_errors.txt'

parser = argparse.ArgumentParser()

parser.add_argument('-aws',
                    default=False,
                    action='store_true',
                    help='Flag for running the pipeline on AWS.')

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
    flags = s > med + 1.25*std
    return ['background-color: #CD5C5CC' if a else '' for a in flags]

def SendEmailAWS(toSubj, data_for_email, times):
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
    df = df[df['size [pix]'].notnull()]
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

         .set_properties(**{'text-align': 'center'})
         .format({'shape [pix]': '{:.3f}', 'size [pix]': '{:.3f}'})
         .set_table_styles(css)
         )
    html_tb = s.render(index=False)
    # This address must be verified with Amazon SES.
    SENDER = "natemiles92@gmail.com"

    # Replace recipient@example.com with a "To" address. If your account
    # is still in the sandbox, this address must be verified.
    RECIPIENT = "nmiles@stsci.edu"

    # Specify a configuration set. If you do not want to use a configuration
    # set, comment the following variable, and the
    # ConfigurationSetName=CONFIGURATION_SET argument below.
    # CONFIGURATION_SET = "ConfigSet"

    # If necessary, replace us-west-2 with the AWS Region you're using for Amazon SES.
    AWS_REGION = "us-east-1"

    # The subject line for the email.
    SUBJECT = toSubj

    # The email body for recipients with non-HTML email clients.
    BODY_TEXT = "{}".format(df.to_string(index=True,
                                         header=True,
                                         justify='center'))

    # The HTML body of the email.
    BODY_HTML = """
                <html>
                    <head></head>
                    <body>
                    <h2>Processing Times</h2>
                        <ul>
                            <li>Downloading data: {:.3f} minutes </li>
                            <li>CR rejection: {:.3f} minutes</li>
                            <li>Labeling analysis: {:.3f} minutes </li>
                            <li>Total time: {:.3f} minutes </li>
                        </ul>
                    <h2> Cosmic Ray Statistics </h2>
                    <p><b> All cosmic ray statistics reported are averages for 
                            the entire image</b></p>
                            {}
                    </body>
                </html>
                """.format(times['download_time'],
                           times['rejection_time'],
                           times['analysis_time'],
                           times['total'],
                           html_tb)

    # The character encoding for the email.
    CHARSET = "UTF-8"

    # Create a new SES resource and specify a region.
    client = boto3.client('ses', region_name=AWS_REGION)
    # Try to send the email.
    try:
        # Provide the contents of the email.
        response = client.send_email(
            Destination={
                'ToAddresses': [
                    RECIPIENT,
                ],
            },
            Message={
                'Body': {
                    'Html': {
                        'Charset': CHARSET,
                        'Data': BODY_HTML,
                    },
                    'Text': {
                        'Charset': CHARSET,
                        'Data': BODY_TEXT,
                    },
                },
                'Subject': {
                    'Charset': CHARSET,
                    'Data': SUBJECT,
                },
            },
            Source=SENDER,
            # If you are not using a configuration set, comment or delete the
            # following line
            # ConfigurationSetName=CONFIGURATION_SET,
        )
    # Display an error if something goes wrong.
    except ClientError as e:
        print(e.response['Error']['Message'])
    else:
        print("Email sent! Message ID:"),
        print(response['MessageId'])

def SendEmail(toSubj, data_for_email, gif_file, times, gif=False):
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
    df = df[df['size [pix]'].notnull()]
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
                <h2>Processing Times</h2>
                <ul>
                    <li>Downloading data: {:.3f} minutes </li>
                    <li>CR rejection: {:.3f} minutes</li>
                    <li>Labeling analysis: {:.3f} minutes </li>
                    <li>Total time: {:.3f} minutes </li>
                </ul>
                <h2> Cosmic Ray Statistics </h2>
                <p><b> All cosmic ray statistics reported are averages for 
                the entire image</b></p>
                <p><b> All cosmic ray statistics reported are averages for 
                        the entire image</b></p>
                {}
                <img src="cid:{}">
            </body>
        </html>
        """.format(times['download_time'],
                   times['rejection_time'],
                   times['analysis_time'],
                   times['total'],
                   html_tb,
                   gif_cid[1:-1])
        msg.add_alternative(body_str, subtype='html',)
    else:
        body_str = """
                <html>
                    <head></head>
                    <body>
                    <h2>Processing Times</h2>
                        <ul>
                            <li>Downloading data: {:.3f} minutes </li>
                            <li>CR rejection: {:.3f} minutes</li>
                            <li>Labeling analysis: {:.3f} minutes </li>
                            <li>Total time: {:.3f} minutes </li>
                        </ul>
                    <h2> Cosmic Ray Statistics </h2>
                    <p><b> All cosmic ray statistics reported are averages for 
                            the entire image</b></p>
                            {}
                    </body>
                </html>
                """.format(times['download_time'],
                           times['rejection_time'],
                           times['analysis_time'],
                           times['total'],
                           html_tb)
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


def write_out(dset_name, fout, data, grp, subgrp, metadata):
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
    # print('Output filename: {}'.format(fout))
    # print('HDF5 structure: /{}/{}'.format(grp, subgrp))
    with h5py.File(fout,'a', libver='latest') as f:
        # print('/{}/{}'.format(grp, subgrp))
        grp = f['/{}/{}'.format(grp,subgrp)]
        print(os.path.basename(dset_name))
        dset = grp.create_dataset(name='{}'.format(os.path.basename(dset_name)),
                                      shape=data.shape,
                                      data=data,
                                      dtype=np.float64)
        for (key, val) in metadata.items():
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
        # print('File structure: /{}/{}'.format(instr, subgrp_names[i]))
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
            # print('Failed to process {}'.format(img_name))
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


def process_dataset(instr, flist, use_pipeline=True):
    """ Run cr crejection on the dataset

    Parameters
    ----------
    instr
    flist

    Returns
    -------
    failed : list of images that could not be CR rejected for various reasons
    """
    start_time = time.time()
    processor = ProcessData(instr, flist)
    # Sort the files by exposure time and chunk to smaller datasets
    processor.sort()
    if use_pipeline:
        processor.cr_reject()
    if 'failed' in processor.output.keys():
        processor.output['failed'] = list(itertools.chain.from_iterable(
            processor.output['failed']))
        f_out = './../crrejtab/{}/{}_failed_' \
                'to_process.txt'.format(instr.split('_')[0], instr)
        write_out_errors(f_out,
                         processor.output['failed'])
    failed = set(processor.output['failed'])
    end_time = time.time()

    return failed, (end_time - start_time)/60


def decompose(f):
    """ Decompose each IMA file into 16 different multi-ext fits files

    Parameters
    ----------
    flist

    Returns
    -------

    """
    p = ProcessIR(f)
    try:
        p.write_out()
    except Exception:
        return os.path.dirname(f)
    else:
        return None


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


def combine_separate_extensions(data):
    """ Convenience funciton for handling WFPC2 and WFC3/IR data

    Parameters
    ----------
    data

    Returns
    -------

    """
    # print('reformatting wfpc2 data')
    int_ids = []
    stats = []
    for chip_results in data:
        int_ids = int_ids + list(chip_results[0])
        stats = stats + list(chip_results[1])
    return np.asarray([int_ids, stats])

def analyze_reads(f0, f1):
    """ Analyze cosmic rays in IR data

    Since IR data is readout using non-destructive reads we need to subtract
    off the cr label from the previous read to ensure we only analyze cosmic
    rays from the specific read

    Parameters
    ----------
    f0
    f1

    Returns
    -------
    """

    # Generate the label from the previous read so we can determine what cosmic
    # rays hit only during read f1
    cr_label_f0 = CosmicRayLabel(f0)

    cr_label_f0.get_data(ext='dq')
    cr_label_f0.get_label(bit_flag=8192, threshold=1)
    # print(sigma_clipped_stats(cr_label_f0.sci[cr_label_f0.sci > 0]))
    # Turn the label from read f0 into a 2D array of 1's and 0's
    previous_label = np.where(cr_label_f0.label > 0, 1, 0)


    cr_label_f1 = CosmicRayLabel(f1)
    cr_label_f1.get_data(ext='dq')
    cr_label_f1.get_label(bit_flag=8192, threshold=1)
    # cr_label_f1.get_data(ext='sci')
    # cr_label_f1.get_label(bit_flag=8192)
    # print(sigma_clipped_stats(cr_label_f1.sci))

    # Turn the label from read f1 into a 2D array for 1's and 0's
    tmp = np.where(cr_label_f1.label > 0, 1, 0)
    # Subtract off the previous label, since they are both just 1's and 0's
    # and everything in f0 is also in f1, the difference will remove all
    # previously flagged cosmic rays
    cr_label_f1.dq = tmp - previous_label
    cr_label_f1.get_label(bit_comp=False)
    # Get science data to pass to stats object
    cr_label_f1.get_data(ext='sci')
    with fits.open(f1) as hdu:
        hdr = hdu[1].header
        # Each read is normalize by the TOTAL integration time for a SAMPLE
        # Hence we need to multiply by that factor to compute total e- deposited
        total_integration_time = hdr['samptime']

        # To determine the CR rate, we use the total time for the single sample
        sample_integration_time = hdr['deltatim']

    stats_obj = ComputeStats(f1,
                             cr_label_f1.label,
                             sci = total_integration_time*cr_label_f1.sci,
                             integration_time = sample_integration_time)
    cr_affected, cr_rate, sizes, shapes, deposition = \
        stats_obj.compute_stats()
    # We return the total number of cosmic rays so that we can
    # define variables for holding the computed statistics
    return cr_affected, cr_rate, sizes, shapes, deposition

def analyze_IR(ima_dir):
    """

    Parameters
    ----------
    imd_dir

    Returns
    -------

    """
    start_time = time.time()
    # print(ima_dir)

    reads = glob.glob(ima_dir + '/read*.fits')
    # print(reads)
    num = [int(f.split('_')[1].split('.')[0]) for f in reads]
    # Sort them by read number, 1 corresponds to 0th read, 2 corresponds
    # to the 1st read, .., N+1 corresponds to the Nth read (last read)
    tmp = list(zip(reads, num))
    tmp.sort(key=lambda val: val[1])
    # Unpack the sorted list to create pairs of sequential reads
    reads = list(zip(*tmp))[0]

    # Note that we skip the 0th read since each read has had the 0th read
    # subtracted off, including the 0th read itself
    sequential_reads = [(reads[i], reads[i + 1])
                             for i in range(1, len(reads) - 1, 1)]
    cr_affected = []
    cr_rate = []
    sizes = []
    shapes = []
    deposition = []
    # Loop through all of the reads and compute the stats
    for pair in sequential_reads:
        affected_tmp, rate_tmp, sizes_tmp, shapes_tmp, deposition_tmp =\
            analyze_reads(*pair)
        cr_affected.append(affected_tmp)
        cr_rate.append(rate_tmp)
        sizes.append(sizes_tmp)
        shapes.append(shapes_tmp)
        deposition.append(deposition_tmp)
    sizes = combine_separate_extensions(sizes)
    shapes = combine_separate_extensions(shapes)
    deposition = combine_separate_extensions(deposition)
    cr_affected = np.asarray([val for data in cr_affected for val in data])
    cr_rate_avg = np.asarray([np.mean(cr_rate)])
    print('Observed CR rate for {} samples is: {} cr/s'.format(len(cr_rate),
                                                            cr_rate_avg))
    end_time = time.time()
    processing_time = (end_time - start_time) / 60
    print(sizes.shape, shapes.shape, deposition.shape)
    return cr_affected, cr_rate_avg, sizes, shapes, deposition, processing_time


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

    if 'c0m.fits' in f:
        # WFPC2 data is SOOOO ANNOYING
        label_obj.label_wfpc2_data()
        cr_affected = []
        cr_rate = []
        sizes = []
        shapes = []
        deposition = []
        for label, sci in zip(label_obj.label,label_obj.sci):
            stats_obj = ComputeStats(f, label, sci, label_obj.integration_time)
            affected_tmp, rate_tmp, sizes_tmp, shapes_tmp, deposition_tmp = \
                stats_obj.compute_stats()

            cr_affected.append(affected_tmp)
            cr_rate.append(rate_tmp)
            sizes.append(sizes_tmp)
            shapes.append(shapes_tmp)
            deposition.append(deposition_tmp)
        # Combine data from each chip into single array
        sizes = combine_separate_extensions(sizes)
        shapes = combine_separate_extensions(shapes)
        deposition = combine_separate_extensions(deposition)
        # combined all pixel coords
        cr_affected = np.asarray([val for data in cr_affected for val in data])
        cr_rate = np.asarray([sum(cr_rate)])
    else:
        # Setting ext='sci' will use the threshold label procedure
        # Setting ext='dq' will use the DQ labeling procedure
        label_obj.get_data(ext='dq')
        label_obj.get_label()
        stats_obj = ComputeStats(f, label_obj.label)

        cr_affected, cr_rate, sizes, shapes, deposition = \
            stats_obj.compute_stats()
    end_time = time.time()
    processing_time = (end_time - start_time)/ 60
    # # print(len(sizes[0]), len(shapes[0]), len(deposition[0]))
    return cr_affected, cr_rate, sizes, shapes, deposition, processing_time


def analyze_data(flist, instr, start, subgrp_names, i, IR=False):
    """ Analyze the list of files that were successfully processed

    Parameters
    ----------
    flist : list of files to process
    instr : instr we are processing (e.g. ACS/WFC, ACS/HRC)
    start : start date of month
    subgrp_names : subgrp_names for stats (sizes, shapes, etc.)
    i : integer correspoding which hdf5 file to write data too (1 - 4)
    IR : boolean switch for processing IR data
    Returns
    -------

    """
    run_start = time.time()
    prefix = instr.split('_')[0]
    data_for_email = defaultdict(list)
    cr_data = defaultdict(list)

    if IR:
        # We pass the path to the directory containing all of the individual
        # reads that were created from the original IMA file
        ima_data = [os.path.dirname(f) for f in flist]
        delayed = [dask.delayed(analyze_IR)(ima) for ima in ima_data]
        results = list(dask.compute(*delayed, scheduler='processes'))


    # results = analyze_file(flist[0])
    else:
        # Process all images at once
        delayed = [dask.delayed(analyze_file)(f) for f in flist]
        results = list(dask.compute(*delayed, scheduler='processes'))


    if 'hrc' in instr.lower() or 'ir' in instr.lower():
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

    file_metadata = []
    # 0: affected, 1: rate, 2: sizes, 3: shapes, 4: deposition
    for i, result in enumerate(results):
        # format the data for writing out
        # Keys are the file and subgrp combos (e.g acs_cr_sizes_1.hdf5, sizes)
        cr_data[(fout[0], subgrp_names[0])].append(result[0])
        cr_data[(fout[1], subgrp_names[1])].append(result[1])
        cr_data[(fout[2], subgrp_names[2])].append(result[2])
        cr_data[(fout[3], subgrp_names[3])].append(result[3])
        cr_data[(fout[4], subgrp_names[4])].append(result[4])

        # Grab metadata for each file (exptime, pointing, lat/lon, etc..)
        # TODO: Need to figure out how to handle IR data
        # the issuse is the flist is a set of directory paths, and not filepaths
        metadata = get_metadata(flist[i])
        file_metadata.append(metadata)

        # grab data for email notification
        data_for_email['filename'].append(flist[i])
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
            # print(hdr['tdateobs'])
            data_for_email['date-obs'].append(hdr['tdateobs'] + ' '
                                              + hdr['ttimeobs'])
        if 'exptime' in hdr:
            data_for_email['exptime'].append(hdr['exptime'])
        elif 'TEXPTIME' in hdr:
            data_for_email['exptime'].append(hdr['texptime'])
        else:
            data_for_email['exptime'].append('EXPTIME missing')
    failed = []
    for j, key in enumerate(cr_data.keys()):
        for i, data in enumerate(cr_data[key]):
            # Don't bother to write out nan results
            if np.isnan(data_for_email['electron_deposition'][i]):
                # print('NaN dataset, skipping {}'.format(flist[i]))
                failed.append(flist[i])
                continue
            print(key[0], key[1], flist[i])
            write_out(dset_name=flist[i],
                      fout=key[0],
                      data=data,
                      grp=instr,
                      subgrp=key[1],
                      metadata=file_metadata[i])
    f_out = './../crrejtab/{}/{}_failed_to_process.txt'.format(
                                                        instr.split('_')[0],
                                                        instr)
    write_out_errors(f_out, failed)
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
    if crjs is not None:
        for a in crjs:
            os.remove(a)
            
    shutil.rmtree('./../crrejtab/{}/mastDownload'.format(val), ignore_errors=True)



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


def download_data(finder, start, stop, aws):
    """

    Parameters
    ----------
    finder
    start
    stop

    Returns
    -------

    """
    start_time = time.time()
    finder.query(range=(start, stop), aws=aws)
    finder.download(start.datetime.date().isoformat())
    end_time = time.time()
    return (end_time - start_time)/60

def main(instr, initialize, aws):
    """ Run the pipeline as whole

    1) Generate a range of dates to analyze
    2) If it hasn't been analyzed, query MAST for darks in that range
    3) Download all darks found
    4) Run CR rejection on all files
        - Combined images with similar apertures and exposure times
        - Images that fail processing will be excluded from further analysis
    5) Analyze the cosmic rays flagged in the processed images
    6) Save the results and send email notification

    To use pipeline CR rejection, set use_pipeline=True in the process()
    function call, as well as, setting ext='dq' in the labeling function calls

    To use threshold labeling, set use_pipeline=False and ext='sci' in the same
    places mentioned above.

    Parameters
    ----------
    instr : Instrument to analyze (STIS_CCD, ACS_WFC, ACS_HRC, WFC3_UVIS, etc..)
    initialize :  Whether or not to create a new batch of HDF5 files
    aws : Whether or not to use AWS services
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
            process_times = {'download_time': 0,
                             'rejection_time': 0,
                             'analysis_time': 0,
                             'total': 0}
            if '{} {}'.format(start.iso, stop.iso) in analyzed_dates:
                print('Already analyzed {} to {}'.format(start.iso, stop.iso))
                continue

            print('Analyzing data from {} to {}'.format(start.iso, stop.iso))
            # Download the data
            download_time = download_data(finder, start, stop, aws)
            process_times['download_time'] = download_time

            if instr == 'WFC3_IR' or instr == 'NICMOS':
                # Before proceeding we have to decompose each IMA into 16
                # separate files, one for each NDR
                flist = glob.glob(search_pattern)
                ima_dirs = [os.path.dirname(f) for f in flist]
                # If a file raises an error during decomposition it will return
                # the directory where the data is written too
                results = [dask.delayed(decompose)(f) for f in flist]
                results = dask.compute(*results, schedulers='processes')
                # Only analyze data that was successfully decomposed
                f_to_analyze = list(set(flist).difference(set(results)))
                if not f_to_analyze:
                    # print('No files to analyze, something happened with processing.')
                    continue
                gif_file, data_for_email, analysis_time = \
                    analyze_data(f_to_analyze,
                                 instr,
                                 start,
                                 cfg['subgrp_names'],
                                 i,
                                 IR=True)
            else:
                # Run cr rejection
                flist = glob.glob(search_pattern)
                if instr =='WFPC2':
                    # WFPC2 has to be handle separately
                    f_to_analyze = flist


                else:
                    failed, rejection_time = process_dataset(instr,
                                                             flist,
                                                             use_pipeline=False)
                    process_times['rejection_time'] = rejection_time
                    f_to_analyze = list(set(flist).difference(failed))

                if not f_to_analyze:
                    # print('No files to analyze, something happened with processing.')
                    continue

                # Run the analysis
                gif_file, data_for_email, analysis_time = \
                    analyze_data(f_to_analyze,
                                 instr,
                                 start,
                                 cfg['subgrp_names'],
                                 i)
            process_times['analysis_time'] = analysis_time

            process_times['total'] = sum(process_times.values())
            subj = 'Finished analyzing {} darks from {} to {}.' \
                   .format(instr, start.datetime.date(),stop.datetime.date())
            if aws:
                SendEmailAWS(subj, data_for_email, process_times)
            else:
                SendEmail(subj, data_for_email, gif_file, process_times, gif=False)
            write_processed_ranges(start, stop, instr)
            clean_files(instr)
            
            
if __name__ == '__main__':
    args = parser.parse_args()
    instr = args.instr.upper()
    main(args.instr.upper(), args.initialize)
    # main('WFPC2', True)
