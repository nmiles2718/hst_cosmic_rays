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
from stistools import ocrreject
from tqdm import tqdm

logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s',
                    )
LOG = logging.getLogger('cr_rejection')
LOG.setLevel(logging.INFO)


parser = argparse.ArgumentParser()

parser.add_argument('N', 
                    help='Number of images to process',
                    type=int,
                    default=20)

parser.add_argument('-dir1',
                    type=str,
                    help='first directory containing files to process',
                    default=None)

parser.add_argument('-dir2',
                    type=str,
                    help='second directory containing files to process',
                    default=None)

parser.add_argument('-crrejtab', 
                    type=str,
                    help='path to crrejtab reference file',
                    default=None
                    )

parser.add_argument('-crsigmas',
                    type=str,
                    help=("sigma thresholds as a comma separated "
                      "string. Default to CRREJTAB"),
                    default=None)

parser.add_argument('-crradius',
                    type=float,
                    help='propagation radius for marking pixels adjacent to \
                         ones identified as outliers by ocrreject. \
                         Default to CRREJTAB',
                    default=None)

parser.add_argument('-crthresh',
                    type=float,
                    help='sigma threshold to use for pixels within crradius \
                          of those identified as outliers by ocrreject. \
                          Default to CRREJTAB',
                    default=None)

parser.add_argument('-crmask',
                    type=str,
                    help="If 'yes', cosmic rays are marked in the DQ arrays of \
                          the input files. Default to CRREJTAB",
                    default=None)

parser.add_argument('-initgues',
                    type=str,
                    help="Method for computing the initial guess; \
                          either 'med' or 'min'. Default to CRREJTAB",
                    default=None)

parser.add_argument('-skysub',
                    type=str,
                    help="Method for estimating the sky background; \
                          either 'none' or 'mode. Default to CRREJTAB",
                    default=None)



def run_setup(dir1, dir2, date_str=None):

    flist1 = glob.glob(dir1 + '*flt.fits')
    flist2 = glob.glob(dir2 + '*flt.fits')
    outdir1, outdir2 = org_tools.setup_output(flist1, flist2,
                                              date_str=date_str)

    return outdir1, outdir2

def run_rejection(
  flist,
  dirname,
  crrejtab=None,
  crradius=None,
  crsigmas=None,
  crthresh=None,
  crmask=None,
  initgues=None,
  scalense=None,
  skysub=None,
  verbose=None,
):
    """

    Parameters
    ----------
    flist : TYPE
      Description
    dirname : TYPE
      Description
    crrejtab : None, optional
      Description
    verbose : None, optional
      Description
    crsigmas : None, optional
      Description
    crthresh : None, optional
      Description
    crmask : None, optional
      Description
    initgues : None, optional
      Description
    skysub : None, optional
      Description
    """
    os.chdir(dirname)
    flist = [os.path.basename(f) for f in flist]
    fout = 'combined_crj.fits'
    
    # Check to see if a file exists for this specified configuation
    try:
        os.remove(fout)
    except FileNotFoundError:
        pass
    try:
        os.remove(fout.replace('crj.fits','spt.fits'))
    except FileNotFoundError:
        pass

    ocrreject.ocrreject(
    ' '.join(flist),
    output=fout,
    crrejtab=crrejtab,
    crsigmas=crsigmas,
    crradius=crradius,
    crthresh=crthresh,
    crmask=crmask,
    initgues=initgues,
    skysub=skysub,
    verbose=True
    )

def main(
        dir1=None,
        dir2=None,
        badinpdq=None,
        crmask=None,
        crrejtab=None,
        crradius=None,
        crsigmas=None,
        crthresh=None,
        initgues=None,
        N=None,
        scalense=None,
        skysub=None,

):

    """
    Parameters
    ----------
    dir1 : None, optional
        Description
    dir2 : None, optional
        Description
    badinpdq : None, optional
        Description
    crmask : str, optional
        Description
    crrejtab : None, optional
        Description
    crradius : float, optional
        Description
    crsigmas : str, optional
        Description
    crthresh : float, optional
        Description
    initgues : str, optional
        Description
    N : None, optional
        Description
    scalense : int, optional
        Description
    skysub : str, optional
        Description
    """
    
    # Create output testing directories and create a record of the input
    # parameters passed to the CR Rejection routine
    LOG.info(f"{dir1}, {dir2}")
    outdir1, outdir2 = org_tools.initialize(
      dir1, dir2, nimages=N, dir_suffix=f"{initgues}_{crsigmas}"
    )

    # Get the lists of files to process
    flist1 = glob.glob(f"{outdir1}/*flt.fits")
    flist2 = glob.glob(f"{outdir2}/*flt.fits")

    for flist, outdir in zip([flist1, flist2],[outdir1, outdir2]):
        # Generate a JSON record for the input parameters
        org_tools.generate_record(outputdir=outdir,
                                  badinpdq=badinpdq,
                                  crmask=crmask,
                                  crrejtab=crrejtab,
                                  crsigmas=crsigmas,
                                  crthresh=crthresh,
                                  crradius=crradius,
                                  initgues=initgues,
                                  scalense=scalense,
                                  skysub=skysub)
         # Run occreject
        run_rejection(
          flist=flist, 
          dirname=outdir,
          crmask=crmask,
          crrejtab=crrejtab,
          crsigmas=crsigmas,
          crthresh=crthresh,
          crradius=crradius,
          initgues=initgues,
          scalense=scalense,
          skysub=skysub
        )

if __name__ == '__main__':
    args = vars(parser.parse_args())
    # Uncomment to set the values of dir1 and dir2
    # args['dir1'] = ("/Users/nmiles/hst_cosmic_rays/"
    #                 "analyzing_cr_rejection/1100.0_clean/")
    # args['dir2'] = ("/Users/nmiles/hst_cosmic_rays/"
    #                 "analyzing_cr_rejection/60.0_clean/")
    main(**args)