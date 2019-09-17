import argparse
from collections import defaultdict
import glob
import json
import logging
import os
import shutil
import sys
sys.path.append('/Users/nmiles/hst_cosmic_rays/pipeline')
import warnings
warnings.simplefilter("ignore")


from astropy.io import fits
from astropy.time import Time
import numpy as np
import organize_tools as org_tools
import pandas as pd
from pipeline import CosmicRayPipeline
from stistools import ocrreject
from utils import initialize





parser = argparse.ArgumentParser()

parser.add_argument('-dir1',
                    type=str,
                    help='first directory containing files to process',
                    default=None
                    )

parser.add_argument('-dir2',
                    type=str,
                    help='second directory containing files to process',
                    default=None
                    )
parser.add_argument('-initialize',
                    help=('Generate new HDF5 files to store the results in'
                          '(will overwrite existing files)'),
                    action='store_true',
                    default=False
                    )

logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s',
                    )
LOG = logging.getLogger('analyze_cr')
LOG.setLevel(logging.INFO)

def run_pipeline(flist=None, dirname=None, **kwargs):
    
    ocrreject_params = json.load(open(f"{dirname}_summary.json",mode='r'))
    fsuffix = f"{ocrreject_params['initgues']}_{ocrreject_params['crsigmas']}"
    LOG.info(fsuffix)

    # Create a CosmicRayPipeline Object to perform the labeling
    cr_pipe = CosmicRayPipeline(**kwargs)
    # Set the chunks attribute to a string to specify the output suffix
    # This ensures observations analyzed with the same initgues and crsigmas
    # are stored in the same HDF5 files.
    cr_pipe.chunks = fsuffix

    # Specify the list of files we are processing
    cr_pipe.flist = flist

    if cr_pipe.initialize:
      # Create an Initializer
      initializer_obj = initialize.Initializer(cr_pipe.instr, cr_pipe.cfg)

      # Set up HDF5 files for saving the data
      initializer_obj.initialize_HDF5(chunks=cr_pipe.chunks)
    
    # Run the pipeline across all the files
    cr_pipe.run_labeling_all(chunk_num=cr_pipe.chunks)
    
def analyze_cosmicrays(dir1=None, dir2=None, initialize=None, **kwargs):

    # Grab the file sets we processed with stistools.occreject
    flist1 = glob.glob(f"{dir1}/*flt.fits")
    flist2 = glob.glob(f"{dir2}/*flt.fits")
    cr_args = {
       'aws': False,
       'download': False,
       'process': False,
       'ccd': True,
       'ir': False,
       'chunks': None,
       'analyze': True,
       'use_dq': True,
       'instr': 'stis_ccd',
       'initialize': initialize,
       'store_downloads': False,
       'test': True
    }
    results = defaultdict(list)
    for i, (dirname, flist) in enumerate(zip([dir1, dir2], [flist1, flist2])):
        LOG.info(f"Processing files in {dirname}")
        keyname = dirname.split('/')[-1]
        # if i == 0:
            # run_pipeline(flist=flist, dirname=dirname, **cr_args)

        run_pipeline(flist=flist, dirname=dirname, **cr_args)

if __name__ == '__main__':
    args = vars(parser.parse_args())
    # Uncomment to set the values of dir1 and dir2
    # args['dir1'] = ("/Users/nmiles/hst_cosmic_rays/"
    #                 "analyzing_cr_rejection/60.0_med_6.5,5.5,4.5/")
    # args['dir2'] = ("/Users/nmiles/hst_cosmic_rays/"
    #                 "analyzing_cr_rejection/60.0_min_6.5,5.5,4.5/")
    analyze_cosmicrays(**args)
