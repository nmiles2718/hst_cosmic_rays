#!/usr/bin/env python

import argparse
from collections import defaultdict
import glob
import sys


from astropy.time import Time
import datahandler as dh
import h5py
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('-instr',
                    type=str,
                    default='ACS_WFC')

def main(instr):
    # Desired information
    metadata_keywords = [
        'date',
        'expend',
        'integration_time',
        'altitude',
        'latitude',
        'longitude',
    ]

    keyword_out = [
        'date_start',
        'date_end'
        'mjd_start',
        'mjd_end',
        'altitude_start',
        'altitude_end',
        'latitude_start',
        'latitude_end',
        'longitude_start',
        'longitude_end',
        'integration_time',
        'incident_cr_rate',
        'cumulative_energy_deposited',
        'energy_per_area',
        'energy_per_area_per_time',
        'average_size_pixels',
        'obs_id'
    ]

    output_data = defaultdict(list)

    rates = dh.DataReader(instr=instr.upper(), statistic='incident_cr_rate')
    size = dh.DataReader(instr=instr.upper(), statistic='sizes')
    energy = dh.DataReader(instr=instr.upper(), statistic='energy_deposited')

    for r in [rates, size, energy]:
        r.find_hdf5()
    if 'STIS' in instr.upper():
        for r in [rates, size, energy]:
            for i,f in enumerate(r.hdf5_files):
                r.hdf5_files[i] = f.replace('STIS','STIS_crrejtab_CRSIGMAS')
    
    print(rates.hdf5_files)
    print(size.hdf5_files)
    print(energy.hdf5_files)
    #sys.exit()
    area = rates.instr_cfg['instr_params']['detector_size']

    flist_tuple = zip(rates.hdf5_files, size.hdf5_files, energy.hdf5_files)
    for (f1, f2, f3) in flist_tuple:
        fobj1 = h5py.File(f1, mode='r')
        grp1 = fobj1['/incident_cr_rate']
        
        fobj2 = h5py.File(f2, mode='r')
        grp2 = fobj2['sizes']
        
        fobj3 = h5py.File(f3, mode='r')
        grp3 = fobj3['energy_deposited']
        for key in grp1.keys():

            rate_dset = grp1[key]
            missing = False
            try:
                size_dset = grp2[key]
            except Exception as e:
                print(e)
                missing=True
        
            try:
                energy_dset = grp3[key]
            except Exception as e:
                print(e)
                missing=True
        
            if missing:
                print(missing)
                continue
        
            output_data['obs_id'].append(key)
            # get the meta data for the dataset
            metadata = rate_dset.attrs
            start_date = Time(metadata['date'], format='iso')
            end_date = Time(metadata['expend'], format='iso')
            output_data['date_start'].append(start_date)
            output_data['mjd_start'].append(start_date.mjd)

            output_data['date_end'].append(end_date)
            output_data['mjd_end'].append(end_date.mjd)

            output_data['integration_time'].append(metadata['integration_time'])


            altitude = metadata['altitude']
            try:
                output_data['altitude_start'].append(altitude[0])
                output_data['altitude_end'].append(altitude[-1])
            except IndexError:
                output_data['altitude_start'].append(altitude)
                output_data['altitude_end'].append(altitude)


            latitude = metadata['latitude']
            try:
                output_data['latitude_start'].append(latitude[0])
                output_data['latitude_end'].append(latitude[-1])
            except IndexError:
                output_data['latitude_start'].append(latitude)
                output_data['latitude_end'].append(latitude)

            longitude = metadata['longitude']
            try:
                output_data['longitude_start'].append(longitude[0])
                output_data['longitude_end'].append(longitude[-1])
            except IndexError:
                output_data['longitude_start'].append(longitude)
                output_data['longitude_end'].append(longitude)

            output_data['incident_cr_rate'].append(rate_dset[()])

            output_data['cumulative_energy'].append(energy_dset.value.sum())
            output_data['cumulative_energy_per_area'].append(
                energy_dset.value.sum()/area
            )
            output_data['cumulative_energy_per_area_per_time'].append(
                energy_dset.value.sum()/area/metadata['integration_time']
            )

            output_data['mean_size_pixels'].append(np.nanmean(size_dset[:][1]))
            output_data['median_size_pixels'].append(np.nanmedian(size_dset[:][1]))

    print('Number of datasets: {}'.format(len(output_data['date_start'])))
    date_index = pd.DatetimeIndex(
        [val.iso for val in output_data['date_start']]
    )
    df = pd.DataFrame(output_data, index=date_index)
    df.sort_index(inplace=True)
    df.to_csv('{}_catalog.txt'.format(instr), header=True, index=False)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.instr)
