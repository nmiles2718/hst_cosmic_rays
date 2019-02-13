#!/usr/bin/env python

import argparse
from astropy.io import fits
import h5py
from collections import Counter
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import time
import yaml

# Empirical thresholds derived by analyzing the data interactively
typical_size = 0.6332
sigma_size = 0.0625
size_thresh = 3

typical_symmetry = 0.4143
sigma_symmetry = 0.0472
sym_thresh = 3

parser = argparse.ArgumentParser()
parser.add_argument('-instr',
                    default=None,
                    help='HST instrument to process (acs_wfc, '
                         'wfc3_uvis, stis_ccd, wfpc2/wfc)')

def get_data(fname, instr):
    start_time = time.time()
    c = Counter()
    file_counter = Counter()
    # with open('./../data/bad_files.txt','r') as fobj:
    #     bad_files = fobj.readlines()
    #     bad_files = [f.strip('\n').replace('raw.fits','blv_tmp.fits')
    #                  for f in bad_files]

    instr_name = instr.split('_')[0].lower()
    fname = fname.replace('_pixes.hdf5','_pixels_master.hdf5')
    print(fname)
    i = 0
    with h5py.File(fname,mode='r') as f:
        grp = f['/{}'.format(instr)]
        subgrp = grp['cr_affected_pixels']
        size_file =  h5py.File('./../data/ACS/{}_sizes_master.hdf5'.format(instr_name),'r')
        shape_file = h5py.File('./../data/ACS/{}_shapes_master.hdf5'.format(instr_name),'r')
        for dset in subgrp.keys():
            # if dset in bad_files:
            #     print('skipping {}'.format(dset))
            #     file_counter['bad'] += 1
            # else:
            size_grp = size_file['/{}'.format(instr.upper())]
            avg_size = np.nanmean(size_grp['sizes'][dset][:][1])
            shape_grp = shape_file['/{}'.format(instr.upper())]
            avg_symmetry = np.nanmean(shape_grp['shapes'][dset][:][1])
            print(dset, avg_size, avg_symmetry)
            if np.absolute(avg_symmetry - typical_symmetry) <= sym_thresh*sigma_symmetry \
            and np.absolute(avg_size - typical_size) <= size_thresh*sigma_size:
                try:
                    coords = subgrp[dset][:]
                except Exception as e:
                    file_counter['bad']+=1
                else:
                    file_counter['good']+=1
                    for coord in coords:
                        c['{},{}'.format(coord[0],coord[1])]+=1
                        i +=1

    #
    size_file.close()
    shape_file.close()
    end_time = time.time()
    duration = end_time - start_time
    if duration > 3600:
        duration /= 3600
        units = 'hours'
    elif duration < 3600:
        duration /= 660
        units = 'seconds'
    print('It took {} {} to read {} cr-affected pixels'.format(duration,
                                                        units,
                                                        i))
    return c, i, duration, units, file_counter

def main(instr):
    print('Starting')
    with open('./../CONFIG/pipeline_config.yaml', 'r') as fobj:
        cfg = yaml.load(fobj)
    fname = cfg[instr]['hdf5_files'][0]
    c, num_datapoints, duration, units, file_counter = \
        get_data(fname, instr=instr)
    counter_array = np.zeros([1024, 1024])
    hdr = fits.Header()

    for key, val in c.items():
        coords = key.split(',')
        coords = (int(float(coords[0])), int(float(coords[1])))
        counter_array[coords[0]][coords[1]] = val

    try:
        tmp = instr.split('_')
        print(file_counter)
        hdr.set(keyword='GOODIMGS',
         value =file_counter['good'],
         comment='Total number of {} {} images used in '\
                 'this analysis'.format(tmp[0], tmp[1]))
        hdr.set(keyword='BADIMGS',
                value=file_counter['bad'],
                comment='Total number of {} {} images '\
                        'excluded in this analysis'.format(tmp[0], tmp[1]))
        hdr.set(keyword='STRIKES',
            value=num_datapoints,
            comment='Number of CR strikes')
        hdr.set(keyword='READTIME',
            value=duration,
            comment='time  to read in data [{}]'.format(units))
        hdu = fits.HDUList()
        hdu.append(fits.ImageHDU(header=hdr, data=counter_array))
        hdu.writeto('cosmic_ray_incidence_map_{}.fits'.format(instr),
                    overwrite=True)
    except Exception as e:
        print(e)
    return counter_array

if __name__ == '__main__':
    args = parser.parse_args()
    instr = args.instr.upper()
    main(instr)