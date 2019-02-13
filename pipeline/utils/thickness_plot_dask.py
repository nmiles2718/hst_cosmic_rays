#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
import dask


parser = argparse.ArgumentParser()
parser.add_argument('-instr',
                    default=None,
                    help='HST instrument to process (acs_wfc, '
                         'wfc3_uvis, stis_ccd, wfpc2/wfc)')

def main(instr):
    print('Starting')
    with open('./../CONFIG/pipeline_config.yaml', 'r') as fobj:
        cfg = yaml.load(fobj)
    fname = cfg[instr]['hdf5_files'][0]

    with h5py.File(fname,mode='r') as f:
        dsets = f['/{}/cr_affected_pixels']
        arrays = [da.from_array(dset, chunks=(10000)) for dset in dsets]


if __name__ == '__main__':
    args = parser.parse_args()
    instr = args.instr.upper()
    main(instr)