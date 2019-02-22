#!/usr/bin/env python

import argparse
import boto3
import glob
import os
import yaml

parser = argparse.ArgumentParser()

parser.add_argument('-instr',
                    default='acs_wfc',
                    help='HST instrument to process (acs_wfc, '
                         'wfc3_uvis, stis_ccd, acs_hrc)')

def transfer(instr):
    """ Transfer data files from EC2 instance to S3 storage for download

    Parameters
    ----------
    instr


    Returns
    -------

    """
    with open('./../CONFIG/pipeline_config.yaml', 'r') as fobj:
        cfg = yaml.load(fobj)
    fnames = cfg[instr]['hdf5_files']
    hdf5_files = []
    for f in fnames:
        search_pattern = f.replace('.hdf5','*')
        flist = glob.glob(search_pattern)
        hdf5_files = hdf5_files + flist

    session = boto3.Session(profile_name='nmiles')
    client = session.client('s3', region_name='us-east-1')
    for f in hdf5_files:
        client.upload_file(f, 'hstcosmicraydata','{}'.format(os.path.basename(f)))

if __name__ == '__main__':
    args = parser.parse_args()
    instr = args.instr.upper()
    transfer(instr)