#!/usr/bin/env python

import argparse
from astropy.io import fits
import h5py
from collections import Counter
from collections import namedtuple
import glob
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

def get_data(flist, instr, stats, cfg):
    start_time = time.time()
    c = Counter()
    file_counter = Counter()

    instr_name = instr.split('_')[0].lower()
    fname_size = cfg[instr]['hdf5_files'][2].replace('.hdf5','_{}.hdf5')
    fname_shape = cfg[instr]['hdf5_files'][3].replace('.hdf5','_{}.hdf5')
    i = 0
    for fname in flist:   
        print(fname)
        fnum = fname.split('_')[-1].split('.')[0]
        with h5py.File(fname,mode='r') as f:
            grp = f['/{}'.format(instr)]
            subgrp = grp['cr_affected_pixels']
            size_file =  h5py.File(fname_size.format(fnum),'r')
            shape_file = h5py.File(fname_shape.format(fnum),'r')
            for dset in subgrp.keys():
                # if dset in bad_files:
                #     print('skipping {}'.format(dset))
                #     file_counter['bad'] += 1
                # else:
                size_grp = size_file['/{}'.format(instr.upper())]
                try:
                    avg_size = np.nanmean(size_grp['sizes'][dset][:][1])
                except KeyError:
                    continue
                shape_grp = shape_file['/{}'.format(instr.upper())]
                try:
                    avg_symmetry = np.nanmean(shape_grp['shapes'][dset][:][1])
                except KeyError:
                    continue
                print(dset, avg_size, avg_symmetry)
                if np.absolute(avg_symmetry - stats.shape_mean) <= sym_thresh*stats.shape_std \
                and np.absolute(avg_size - stats.size_mean) <= size_thresh*stats.size_std:
                    try:
                        coords = subgrp[dset][:]
                    except Exception as e:
                        file_counter['bad']+=1
                    else:
                        file_counter['good']+=1
                        for coord in coords:
                            c['{},{}'.format(coord[0],coord[1])]+=1
                            i +=1
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
    stats = namedtuple('stats',['size_mean','size_std', 'shape_mean','shape_std'])
    
    _CFG = {
            'ACS_WFC': {'stat': stats(size_mean=0.513, size_std=0.0625, shape_mean=0.41, shape_std=0.0625),
                        'detector_size':[4096, 4096]},
            'ACS_HRC': {'stat':stats(size_mean=0.513, size_std = 0.0625, shape_mean=0.41, shape_std=0.0625),
                        'detector_size':[1024, 1024]},
            'STIS_CCD': {'stat':stats(size_mean=0.513, size_std = 0.0625, shape_mean=0.41, shape_std=0.0625),
                         'detector_size':[1024,1024]},
            'WFC3_UVIS': {'stat':stats(size_mean=0.513, size_std=0.03125, shape_mean=0.41, shape_std=0.03125),
                          'detector_size':[4102, 4096]}                        
          }
   

    print('Starting')
    with open('./../CONFIG/pipeline_config.yaml', 'r') as fobj:
        cfg = yaml.load(fobj)
    search_pattern = cfg[instr]['hdf5_files'][0]
    search_pattern = search_pattern.replace('.hdf5','_?.hdf5')
    flist = glob.glob(search_pattern)
    c, num_datapoints, duration, units, file_counter = \
        get_data(flist, instr=instr, cfg=cfg, stats=_CFG[instr]['stat'])
    # Make a blank map for CR incidence
    counter_array = np.zeros(_CFG[instr]['detector_size'])
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
