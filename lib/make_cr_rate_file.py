#!/usr/bin/env python

import argparse
from astropy.io import fits
from astropy.time import Time
import h5py
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-instr', help='ACS_WFC, WFC3_UVIS, STIS_CCD')


def make_file(instr, subgrp):
    if instr == 'ACS_WFC':
        f_out = './acs_cr_rate_test.hdf5'
    else:
        f_out = './{}_cr_rate_test.hdf5'.format(instr.split('_')[0].lower())
    with h5py.File(f_out,'w') as f:
        grp = f.create_group(instr)
        subgrp = grp.create_group(subgrp)
    return f_out

def write_data(f_data, f_out, instr):
    f_data = h5py.File(f_data, 'r')
    f_out = h5py.File(f_out, 'a')
    grp_out = f_out['/{}/incident_cr_rate'.format(instr)]
    subgrp = f_data['/{}/sizes'.format(instr)]
    for key in subgrp.keys():
        dset_sizes = subgrp[key]
        num_cr = len(dset_sizes[:][1])
        expstart = Time(dset_sizes.attrs['expstart'], format='iso')
        expend = Time(dset_sizes.attrs['expend'], format='iso')
        delta = expend - expstart
        cr_rate = num_cr / delta.sec
        dset = grp_out.create_dataset(name=key,
                                  data=cr_rate,
                                  dtype=np.float64)
        for (key, value) in dset_sizes.attrs.items():
            if key in ['mean','std','max','min']:
                continue
            dset.attrs[key] = value
    f_data.close()
    f_out.close()


def main(instr):
    name = instr.split('_')[0]
    fname = './../data/{}/{}_cr_sizes.hdf5'.format(name, name.lower())
    f_out = make_file(instr, 'incident_cr_rate')
    write_data(fname, f_out, instr)





if __name__ == '__main__':
    args = parser.parse_args()
    main(args.instr)
