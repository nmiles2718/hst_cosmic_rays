#!/usr/bin/env python

import argparse
from collections import defaultdict
import glob
import logging
import os
import shutil

from astropy.io import fits
from astropy.time import Time
import numpy as np
import organize_tools as org_tools
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


def run_setup(dir1, dir2, date_str=None):

    flist1 = glob.glob(dir1 + '*flt.fits')
    flist2 = glob.glob(dir2 + '*flt.fits')
    outdir1, outdir2 = org_tools.setup_output(flist1, flist2,
                                              date_str=date_str)

    return outdir1, outdir2

def run_rejection(dir1, dir2, ):
    pass


def main(
        dir1=None,
        dir2=None,
        badinpdq=None,
        crmask="yes",
        crrejtab=None,
        crsigmas='6,5,4',
        crthresh=0.75,
        initgues='med',
        scalense=0,
        skysub="mode",

):
    tday = Time.now().to_datetime()
    date_str = tday.strftime('%b%d_%H:%M')
    outdir1, outdir2 = run_setup(dir1, dir2, date_str=date_str)

    org_tools.generate_record(outdir1,
                              badinpdq,
                              crmask,
                              crrejtab,
                              crsigmas,
                              crthresh,
                              initgues,
                              scalense,
                              skysub)

    flist1 = glob.glob(f"{outdir1}/*flt.fits")
    flist2 = glob.glob(f"{outdir2}/*flt.fits")




if __name__ == '__main__':
    # args = vars(parser.parse_args())
    args = {
        'dir1':'/Users/nmiles/hst_cosmic_rays/'
               'analyzing_cr_rejection/long_clean/',
        'dir2':'/Users/nmiles/hst_cosmic_rays/'
               'analyzing_cr_rejection/short_clean/'
    }
    main(**args)