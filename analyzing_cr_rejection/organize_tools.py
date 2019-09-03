#!/usr/bin/env python

import argparse
from collections import defaultdict
import glob
import inspect
import json
import logging
import os
import shutil

from astropy.io import fits
from astropy.time import Time
import numpy as np
import pandas as pd
from tqdm import tqdm


logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s',
                    level=logging.INFO)
LOG = logging.getLogger('cr_rejection')
LOG.setLevel(logging.INFO)


parser = argparse.ArgumentParser()

parser.add_argument('-dir1',
                    type=str,
                    help='first directory containing files to process')

parser.add_argument('-dir2',
                    type=str,
                    help='second directory containing files to process')



def find_files(longdir=None, shortdir=None, suffix=None):
    """ Read in all files in longdir and shortdir with filetype suffix

    Parameters
    ----------
    longdir
    shortdir
    suffix

    Returns
    -------

    """
    flistlong = glob.glob(f'{longdir}/{suffix}')
    flistshort = glob.glob(f'{shortdir}/{suffix}')
    return flistlong, flistshort


def generate_record(
        outputdir,
        badinpdq=None,
        crmask="yes",
        crrejtab=None,
        crsigmas='6,5,4',
        crthresh=0.75,
        crradius=1.0,
        initgues='med',
        scalense=0,
        skysub="mode"
):
    """ Generate an output file containing a summary of the input parameters

    Parameters
    ----------
    outputdir : str

    crsigmas : list

    crthresh : float

    crradius : float


    Returns
    -------

    """
    frame = inspect.currentframe()
    arginfo = inspect.getargvalues(frame)
    args = arginfo.args
    values = [arginfo.locals[arg] for arg in args]
    fout_json = f"{outputdir}/_summary.json"
    LOG.info(f"Writing the input parameters to the following file:\n{fout_json}")
    data_dict = {}
    for key, val in zip(args, values):
        data_dict[key] = val
    json.dump(data_dict, open(fout_json, mode='w'))



# noinspection PyTypeChecker
def setup_output(flist1, flist2, dir_suffix=''):
    """ Setup output directories for each list of files

    Parameters
    ----------
    flist1 : list

    flist2 : list


    Returns
    -------

    """
    # Generate the output path for each input list
    dir1_path = os.path.dirname(flist1[0])
    dir1_name = dir1_path.split('/')[-1]
    outdir1 = dir1_path.replace(
        dir1_name,
        f"{dir1_name.split('_')[0]}_{dir_suffix}"
    )

    try:
        os.mkdir(outdir1)
    except FileExistsError:
        pass

    dir2_path = os.path.dirname(flist2[0])
    dir2_name = dir2_path.split('/')[-1]
    outdir2 = dir2_path.replace(
        dir2_name,
        f"{dir2_name.split('_')[0]}_{dir_suffix}"
    )
    try:
        os.mkdir(outdir2)
    except FileExistsError:
        pass
    LOG.info(
        f"Set up two output directories: \n{outdir1}\n{outdir2}\n {'-'*79}"
    )

    for f1 in tqdm(flist1, desc=f'Copying to {os.path.basename(outdir1)} '):
        fits.setval(f1, keyword='CRREJTAB', 
            value='/Users/nmiles/hst_cosmic_rays/j3m1403io_crr.fits')
        shutil.copy(f1, outdir1)
        shutil.copy(f1.replace('flt.fits','spt.fits'), outdir1)

    for f2 in tqdm(flist2,desc=f'Copying to {os.path.basename(outdir2)} '):
        fits.setval(f2, keyword='CRREJTAB', 
            value='/Users/nmiles/hst_cosmic_rays/j3m1403io_crr.fits')
        shutil.copy(f2, outdir2)
        shutil.copy(f2.replace('flt.fits','spt.fits'), outdir2)

    return outdir1, outdir2

def sort_flist(flist):
    date_time = []
    for f in flist:
        with fits.open(f) as hdu:
            hdr = hdu[0].header
            try:
                dateobs = hdr['DATE-OBS']
            except KeyError:
                dateobs = hdr['TDATEOBS']
            try:
                timeobs = hdr['TIME-OBS']
            except KeyError:
                timeobs = hdr['TTIMEOBS']

        date_time.append(Time(f"{dateobs} {timeobs}",
                              format='iso').to_datetime())
    data = list(zip(flist, date_time))
    data.sort(key=lambda val: val[1])
    flist, date_time = zip(*data)
    return flist



def initialize(dir1, dir2, nimages=None, dir_suffix=None):
    # Get string to use for recorded the start of processing
    if dir_suffix is None:
        tday = Time.now().to_datetime()
        dir_suffix = tday.strftime('%b%d')

    flist1 = sort_flist(glob.glob(dir1+'*flt.fits'))
    flist2 = sort_flist(glob.glob(dir2+'*flt.fits'))
    
    if nimages is not None:
        flist1 = flist1[:nimages]
        flist2 = flist2[:nimages]

    outdir1, outdir2 = setup_output(flist1, flist2, dir_suffix=dir_suffix)
    return outdir1, outdir2

def parse_inputs():
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':
    # args = vars(parser.parse_args())
    args = {
        'dir1':'/Users/nmiles/hst_cosmic_rays/'
               'analyzing_cr_rejection/1100.0_clean/',
        'dir2':'/Users/nmiles/hst_cosmic_rays/'
               'analyzing_cr_rejection/60.0_clean/'
    }
    main(**args)