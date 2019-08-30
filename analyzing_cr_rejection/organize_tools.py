#!/usr/bin/env python

import argparse
from collections import defaultdict
import glob
import inspect
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
    data = zip(values, args)
    fout = f"{outputdir}/_summary.txt"
    LOG.info(f"Writing the input parameters to the following file:\n {fout}")
    with open(fout, mode='w') as fobj:
        fobj.write('# Cosmic Ray Rejection Parameters\n')
        for val, label in data:
            fobj.write(f"{label} {val}\n")


# noinspection PyTypeChecker
def setup_output(flist1, flist2, date_str=None):
    """ Setup output directories for each list of files

    Parameters
    ----------
    flist1 : list

    flist2 : list


    Returns
    -------

    """
    if date_str is None:
        date_str = ''

    # Generate the output path for each input list
    dir1_path = os.path.dirname(flist1[0])
    dir1_name = dir1_path.split('/')[-1]
    outdir1 = dir1_path.replace(
        dir1_name,
        f"{dir1_name.split('_')[0]}_{date_str}"
    )

    try:
        os.mkdir(outdir1)
    except FileExistsError:
        pass

    dir2_path = os.path.dirname(flist2[0])
    dir2_name = dir2_path.split('/')[-1]
    outdir2 = dir2_path.replace(
        dir2_name,
        f"{dir2_name.split('_')[0]}_{date_str}"
    )
    try:
        os.mkdir(outdir2)
    except FileExistsError:
        pass
    LOG.info(
        f"\nSet up two output directories: \n{outdir1}\n {outdir2}\n {'-'*79}"
    )

    for f1 in tqdm(flist1, desc=f'Copying to {os.path.basename(outdir1)} '):
        shutil.copy(f1, outdir1)

    for f2 in tqdm(flist2,desc=f'Copying to {os.path.basename(outdir2)} '):
        shutil.copy(f2, outdir2)

    return outdir1, outdir2


# def main(dir1, dir2):
#     # Get string to use for recorded the start of processing
#     tday = Time.now().to_datetime()
#     date_str = tday.strftime('%b%d_%H:%M')
#     flist1 = glob.glob(dir1+'*flt.fits')
#     flist2 = glob.glob(dir2+'*flt.fits')
#     outdir1, outdir2 = setup_output(flist1, flist2, date_str=date_str)
#     generate_record(outdir1, crradius=1, crsigmas='8,6,4', crthresh=1)



def parse_inputs():
    args = vars(parser.parse_args())
    return args

if __name__ == '__main__':
    # args = vars(parser.parse_args())
    args = {
        'dir1':'/Users/nmiles/hst_cosmic_rays/'
               'analyzing_cr_rejection/long_clean/',
        'dir2':'/Users/nmiles/hst_cosmic_rays/'
               'analyzing_cr_rejection/short_clean/'
    }
    main(**args)