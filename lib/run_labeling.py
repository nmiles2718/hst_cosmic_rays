#!/usr/bin/env python

import argparse
from astropy.io import fits
from collections import Iterable
import glob
import h5py
import numpy as np
import os
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import shutil
import yaml


# local imports
from CosmicRayLabel import CosmicRayLabel
from ComputeStats import ComputeStats
from FindData import FindData
from ProcessData import ProcessData
from GenerateMetadata import GenerateMetadata

enocding_file = '/grp/hst/acs7/nmiles/cr_repo/lib/encoding_errors.txt'

parser = argparse.ArgumentParser()

parser.add_argument('-instr',
                    default=None,
                    help='HST instrument to process (acs_wfc, wfc3_uvis, stis_ccd)')
parser.add_argument('-initialize',
                    action='store_true',
                    default=False,
                    help='Use this flag if this is the first time the '
                         'pipeline has been run. It will ensure all HDF5 files '
                         'are properly initialized with the correct structure.')


def SendEmail(toSubj, bodyStr):
    """
    Send out an email to user(s).

    Parameters
    ----------
    fromAddr: string
        Email address of sender.

    toAddrList: list of string
        Email address(es) of receipient(s).

    toSubj: string
        Subject line.

    bodyStr: string
        Email content.

    """
    toAddr = 'nmiles@stsci.edu'

    msg = MIMEMultipart()
    msg['Subject'] = toSubj
    msg['From'] = toAddr
    msg['To'] = toAddr
    msg.attach(MIMEText(bodyStr))

    s = smtplib.SMTP('smtp.stsci.edu')
    s.sendmail(toAddr, toAddr, msg.as_string())
    s.quit()


def get_metadata(fname):
    meta = GenerateMetadata(fname)
    # Get image specific data (date-obs, expstart, expend, etc..)
    meta.get_image_data()
    # Get HST specific data (latitude, longitude, altitude)
    meta.get_observatory_info()
    return meta.metadata


def write_out(fname, fout, data, grp, subgrp, update=False):
    print('HDF5 structure: /{}/{}'.format(grp, subgrp))
    with h5py.File(fout,'a', libver='latest') as f:
        print('/{}/{}'.format(grp, subgrp))
        grp = f['/{}/{}'.format(grp,subgrp)]
        subgrps = list(grp.keys())
        metadata = get_metadata(fname)
        if subgrp in ['sizes', 'shapes']:
            metadata['mean'] = np.nanmean(data[1])
            metadata['std'] = np.nanstd(data[1])
            metadata['max'] = np.nanmax(data[1])
            metadata['min'] = np.nanmin(data[1])

        elif subgrp in ['cr_affected_pixels']:
            # these statistics can't be computed on a per image basis
            metadata['number_affected'] = len(data)

        if os.path.basename(fname) in subgrps and update:
            dset = grp['{}'.format(os.path.basename(fname))]
            if isinstance(data, Iterable):
                dset[()] = data
            else:    
                dset[:] = data
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
    finder.query()
    return finder


def process_dataset(instr, flist):
    processor = ProcessData(instr, flist)
    # Sort the files by exposure time and chunk to smaller datasets
    processor.sort()
    processor.cr_reject()
    print(len(processor.output['failed']))
    if 'failed' in processor.output.keys():
        write_out_errors('{}_failed_to_process.txt'.format(instr),
                         processor.output['failed'])
    failed = set(processor.output['failed'])
    return failed


def analyze_data(flist, instr, subgrp_names):
    start_time = time.time()
    num_cr_per_anneal = 0

    prefix = instr.split('_')[0]
    sizes_avg = []
    shapes_avg = []
    cr_incident_rate = []
    cr_deposition = []
    for f in flist:
        label_obj = CosmicRayLabel(f)
        
        label_obj.get_data()
        label_obj.get_label()
        stats_obj = ComputeStats(f, label_obj.label)
        stats_obj.get_data()
        cr_affected, cr_rate, sizes, anisotropy, deposition = \
            stats_obj.compute_stats()

        # Compute some data to supply in an email
        num_cr_per_anneal += len(sizes.values())
        sizes = format_data(sizes)
        anisotropy = format_data(anisotropy)
        sizes_avg.append(np.nanmean(sizes[1]))
        shapes_avg.append(np.nanmean(anisotropy[1]))
        cr_incident_rate.append(cr_rate)
        cr_deposition.append(deposition[1])

        # Package the files and data for writing out.

        fout = [
            './../data/{}/{}_cr_affected_pixels.hdf5'.format(prefix.upper(),
                                                             prefix),
            './../data/{}/{}_shapes.hdf5'.format(prefix.upper(),
                                                 prefix),
            './../data/{}/{}_sizes.hdf5'.format(prefix.upper(),
                                                prefix),
            './../data/{}/{}_cr_rate.hdf5'.format(prefix.upper(),
                                                  prefix),
            './../data/{}/{}_cr_deposition.hdf5'.format(prefix.upper(),
                                                        prefix)
        ]
        data_out = [
            cr_affected,
            anisotropy,
            sizes,
            cr_rate,
            deposition
        ]
        for (fname, data, subgrp) in zip(fout, data_out, subgrp_names):
            print(fname, data, instr, subgrp)v
            write_out(f,
                      fout=fname,
                      data=data,
                      grp=instr,
                      subgrp=subgrp,
                      update=True
                      )

    end_time = time.time()
    processing_time = (end_time - start_time)/60
    subject = 'Finished processing {} files'.format(len(flist))
    msg = 'There was a total of {} cosmic rays ' \
           'in this anneal cycle\n'.format(num_cr_per_anneal)
    msg += 'The average cr incident rate is {}\n'.format(np.mean(cr_incident_rate))
    msg += 'The average anisotropy is {}\n'.format(np.mean(shapes_avg))
    msg += 'The average sigma-size is {} pixels\n'.format(np.mean(sizes_avg))
    msg += 'Processing time: {} minutes'.format(processing_time)
    print(msg)
    SendEmail(subject, msg)

def clean_files(instr):
    crjs = glob.glob('./tmp*')
    val = instr.split('_')[0]
    for a in crjs:
        os.remove(a)
    shutil.rmtree('./../crrejtab/{}/mastDownload'.format(val))


def main():
    args = parser.parse_args()
    instr = args.instr.upper()
    with open('./../CONFIG/pipeline_config.yaml', 'r') as fobj:
        cfg = yaml.load(fobj)

    if args.initialize:
        initialize_hdf5(instr, cfg[instr], cfg['subgrp_names'])

    finder = find_files_to_download(instr)
    search_pattern = cfg[instr]['search_pattern'][0]
    print(cfg['subgrp_names'])
    for key in finder._products.keys():
        # finder.download(key)
        flist = glob.glob(search_pattern)
        # failed = process_dataset(instr, flist)
        failed = {'a'}
        f_to_analyze = list(set(flist).difference(failed))
        if not f_to_analyze:
            print('No files to analyze, something happened with processing.')
        else:
            analyze_data(f_to_analyze, instr, cfg['subgrp_names'])
        clean_files(instr)
        

if __name__ == '__main__':
    main()