#!/usr/bin/env python

from astropy.io import fits
import h5py
import numpy as np


def make_file(instr, subgrp):
    if instr == 'ACS_WFC':
        f_out = './acs_cr_rate_test.hdf5'
    else:
        f_out = './{}_cr_rate_test.hdf5'.format(instr.lower())
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
        dset = grp_out.create_dataset(name=key,
                                  data=num_cr,
                                  dtype=np.float64)
        for (key, value) in dset_sizes.attrs.items():
            if key in ['mean','std','max','min']:
                continue
            dset.attrs[key] = value
    f_data.close()
    f_out.close()


def main(instr='ACS_WFC'):
    fname = '/Users/nmiles/hstcosmicrays/data/ACS/acs_sizes.hdf5'
    f_out = make_file(instr, 'incident_cr_rate')
    write_data(fname, f_out, instr)





if __name__ == '__main__':
    main()