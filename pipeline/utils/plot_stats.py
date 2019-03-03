#!/usr/bin/env python

from collections import defaultdict
from itertools import chain
from astropy.time import Time
from astropy.visualization import ImageNormalize, LinearStretch, \
    LogStretch, ZScaleInterval, SqrtStretch
import dask.array as da
import costools
from collections import Iterable
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
rc('font', weight='bold', family='sans-serif')
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.basemap import Basemap
import pandas as pd
import sunpy
import sunpy.timeseries
import sunpy.data.sample

plt.style.use('ggplot')


class PlotData(object):
    def __init__(self, fname, instr, subgrp, flist):
        self.fname = fname
        self.instr = instr.upper()
        self.flist = flist
        self.data = None
        self.data_no_saa = None
        self.data_df = None
        self.subgrp = subgrp
        self.num_images = 0
        self.map = None
        self.ax = None
        self.fig = None
        # Detector sizes in cm^2
        self.detector_size ={'ACS_WFC': 37.748,
                             'ACS_HRC': 4.624 ,
                             'WFC3_UVIS': 37.804,
                             'WFC3_IR': 3.331,
                             'STIS_CCD': 4.624,
                             'WFPC2':5.76}

        self.detector_readtime = {'ACS_WFC': 1,
                                  'ACS_HRC': 30,
                                  'WFC3_UVIS': 1.,
                                  'WFC3_IR' : None,
                                  'STIS_CCD': 29.0/2.,
                                  'WFPC2': 60./2.}

    def read_rate(self):
        data_out = defaultdict(list)
        for f in self.flist:
            print('Analyzing {}'.format(f))
            fobj = h5py.File(f, mode='r')
            sgrp = fobj[self.instr.upper()+'/'+self.subgrp]
            for key in sgrp.keys():
                data_out['obs_name'].append(key)
                dset = sgrp[key].value
                attrs = sgrp[key].attrs
                exptime = attrs['exptime']
                factor = (exptime + self.detector_readtime[self.instr.upper()]) \
                        / exptime
                if isinstance(dset, Iterable):
                    dset = np.nanmedian(dset)

                # Multiply by correction factor to account for various instrument
                # readouts
                data_out[self.subgrp].append(factor * dset /
                                             self.detector_size[self.instr])
                for at_key in attrs.keys():
                    val = attrs[at_key]
                    if at_key == 'date':
                        val = Time(val, format='iso')
                    elif at_key == 'latitude' or at_key == 'longitude' \
                        or at_key == 'altitude':
                        try:
                            data_out['start_{}'.format(at_key)].append(val[0])
                        except IndexError as e:
                            data_out['start_{}'.format(at_key)].append(val)
                        try:
                            data_out['end_{}'.format(at_key)].append(val[-1])
                        except IndexError as e:
                            data_out['end_{}'.format(at_key)].append(val)
                        continue
                    data_out['{}'.format(at_key)].append(val)
        data_out['mjd'] = [val.mjd for val in data_out['date']]
        date_index = pd.DatetimeIndex([val.iso for val in data_out['date']])
        self.data_df = pd.DataFrame(data_out, index = date_index)
        self.data_df.sort_index(inplace=True)
    

    def perform_SAA_cut(self):
        saa = [list(t) for t in zip(*costools.saamodel.saaModel(5))]
        saa[0].append(saa[0][0])
        saa[1].append(saa[1][0])
        saa = np.asarray(saa)
        saa_eastern = (39.0, -30.0) # lon/lat
        saa_western = (267.0, -20.0)
        saa_northern = (312.0, 1.0)

        mask = (self.data_df['longitude'] > saa_eastern[0]) &\
               (self.data_df['longitude'] < saa_western[0]) &\
               (self.data_df['latitude'] > saa_northern[1])
        cut = self.data_df[mask]
        self.data_no_saa = cut
        return mask


    def read_data(self):
        data_out = defaultdict(list)
        for f in self.flist:
            fobj = h5py.File(f, mode='r')
            sgrp = fobj[self.instr.upper()+'/'+self.subgrp]
            for key in sgrp.keys():
                dset = sgrp[key][:][1]
                attrs = sgrp[key].attrs
                data_out['var_average'].append(np.nanmedian(dset))
                for at_key in attrs.keys():
                    val = attrs[at_key]
                    if at_key == 'date':
                        val = Time(val, format='iso')
                    data_out['{}'.format(at_key)].append(val)
        date_index = pd.DatetimeIndex([val.iso for val in data_out['date']])
        self.data_df = pd.DataFrame(data_out, index = date_index)
    


    def plot_data(self, ax=None, bins=30, logx=True, logy=True,
                  range=[0, 3], fill_value=-1,c='r'):
        """ plot a histogram of data, defaults are set for sizes

        Parameters
        ----------
        subgrp
        bins
        range

        Returns
        -------

        """
        # Read in the data if it doesn't exist already
        tmp = []
        if '_' in self.instr:
            label = self.instr.replace('_','/')
        else:
            label = self.instr
        if self.data is None:
            for f in self.flist:
                print('Analyzing file {}'.format(f))
                with h5py.File(f, mode='r') as fobj:
                    subgrp_ = fobj[self.instr.upper()+'/'+self.subgrp]
                    keys = list(subgrp_.keys())
                    self.num_images += len(keys)
                    for name in keys:
                        if logx:
                            dset = np.log10(subgrp_[name][:][1])
                        else:
                            dset = subgrp_[name][:][1]
                        tmp.append(da.from_array(dset,chunks=(20000)))

            x = da.concatenate(tmp, axis=0)
            print(x.shape)
            self.data = da.ma.fix_invalid(x, fill_value=fill_value)

        h, edges = da.histogram(self.data, bins=bins,
                                range=range)
        hist = h.compute()
        # Create an axis if it doesnt exists
        lw = 1.75
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(7,5),
                                      nrows=1,
                                      ncols=1)
            # Normalize by the highest count
            # self.data[subgrp] /= np.nanmax(self.data[subgrp])
            if logy:
            # self.ax.step(edges[:-1], h.compute(), color='r')

                self.ax.semilogy(edges[:-1], hist,
                                 label=label,
                                 drawstyle='steps-mid', color=c, lw=lw)
            else:
                self.ax.step(edges[:-1], hist,
                                 label=label,
                                 where='mid', color=c,lw=lw)
        else:
            self.ax = ax
            if logy:
            # self.ax.step(edges[:-1], h.compute(), color='r')
                self.ax.semilogy(edges[:-1], hist,
                                 label=label,
                                 drawstyle='steps-mid', color=c,lw=lw)
            else:
                self.ax.step(edges[:-1], hist,
                                 label=label,
                                 where='mid', color=c,lw=lw)
        self.ax.tick_params(axis='both', which='major',
                            labelsize=10, width=2)
        self.ax.legend(loc='best')


    def plot_rate_vs_time(self, ax= None, ptype='rolling',
                          window='20D', min_periods=20, i=0, saa_exclude=False):
        if self.data_df is None:
            self.read_rate()

        if saa_exclude:
            flags = self.data_no_saa.exptime.gt(200)
            df1 = self.data_no_saa[['incident_cr_rate','mjd']][flags]
        else:
            flags = self.data_df.exptime.gt(200)
            df1 = self.data_df[['incident_cr_rate','mjd']][flags]

        if ptype == 'rolling':
            averaged = df1.rolling(window=window, min_periods=min_periods).median()
        elif ptype == 'resample':
            averaged = df1.resample(rule=window).median()
        else:
            averaged = df1

        avg_no_nan = averaged.dropna()
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(7,5),
                                       nrows=1,
                                       ncols=1)
        else:
            self.ax = ax
        # Normalize the CR rate by the detector size

        CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                          '#f781bf', '#a65628', '#984ea3',
                          '#999999', '#e41a1c', '#dede00']
        self.ax.scatter([Time(val, format='mjd').to_datetime()
                         for val in avg_no_nan['mjd']],
                        avg_no_nan['incident_cr_rate'],
                        label=self.instr.replace('_','/'),
                        s=2,
                        color=CB_color_cycle[i])
        # self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Cosmic Ray Rate [CR/s/cm^2]')
        self.ax.set_title('Smoothed Cosmic Ray Rate')
       # plt.savefig('cr_incidence_rolling_average.png',)
       # plt.show()



    def plot_solar_cycle(self, variable=None, ax = None, smoothed=False):
        noaa = sunpy.timeseries.TimeSeries(sunpy.data.sample.NOAAINDICES_TIMESERIES,
                                           source='NOAAIndices')

        print(noaa.columns)
        if variable is None:
            noaa.peek(type='sunspot RI', ax=ax)
        else:
            noaa.peek(type=variable, ax=ax)
        return noaa

    def draw_map(self, map=None, scale=0.2):

        if map is None:
            pass
        else:
            self.map=map
        # Set the background map up
        self.map.shadedrelief(scale=scale)

        # Draw the meridians
        # lats and longs are returned as a dictionary
        lats = self.map.drawparallels(np.linspace(-90, 90, 13),
                                      labels=[True, True, False, False])

        lons = self.map.drawmeridians(np.linspace(-180, 180, 13),
                                      labels=[False, False, False, True])

        # keys contain the plt.Line2D instances
        lat_lines = chain(*(tup[1][0] for tup in lats.items()))
        lon_lines = chain(*(tup[1][0] for tup in lons.items()))
        all_lines = chain(lat_lines, lon_lines)
        # cycle through these lines and set the desired style
        for line in all_lines:
             line.set(linestyle='-', alpha=0.3, color='w')

    def plot_hst_loc(self, i = 5, df = None, key='start', save=False):

        self.fig = plt.figure(figsize=(9, 7))
        # Get the model for the SAA
        self.map = Basemap(projection='cyl')

        self.draw_map()

        # Generate an SAA contour
        saa = [list(t) for t in zip(*costools.saamodel.saaModel(i))]
        # Ensure the polygon representing the SAA is a closed curve by adding
        # the starting points to the end of the list of lat/lon coords
        saa[0].append(saa[0][0])
        saa[1].append(saa[1][0])
        self.map.plot(saa[1], saa[0],
               c='r',
               latlon=True,
               label='SAA contour {}'.format(i))
        if df is None:
            lat, lon, rate = self.data_df['{}_latitude'.format(key)], \
                             self.data_df['{}_longitude'.format(key)], \
                             self.data_df['incident_cr_rate']
        else:
            lat, lon, rate = df['{}_latitude'.format(key)], \
                             df['{}_longitude'.format(key)], \
                             df['incident_cr_rate']

        norm = ImageNormalize(rate,
                              stretch=LinearStretch(),
                              vmin=0.75, vmax=2)


        scat = self.map.scatter(lon, lat,
                         marker='o',
                         s=10,
                         latlon=True,
                         c=rate, alpha=0.5,
                         norm = norm,
                         cmap='Reds')

        cax = self.fig.add_axes([0.1, 0.1, 0.8, 0.05])
        cbar = self.fig.colorbar(scat, cax=cax, orientation='horizontal')
        cbar.set_label('cosmic rays/s/cm^2', fontweight='bold')
        cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)

        if save:
            self.fig.savefig('lat_lon_{}.png'.format(key),
                             format='png',
                             dpi=300)
        plt.show()

    def get_full_path(self, obsnames):
        data_out = defaultdict(list)
        for f in self.flist:
            print('Analyzing {}'.format(f))
            fobj = h5py.File(f, mode='r')
            sgrp = fobj[self.instr.upper()+'/'+self.subgrp]
            for key in sgrp.keys():
                if key not in obsnames:
                    continue
                dset = sgrp[key].value
                attrs = sgrp[key].attrs
                num_intervals = len(attrs['latitude'])
                data_out['obs_name'].append([key]*num_intervals)
                data_out['exptime'].append([attrs['exptime']]*num_intervals)
                exptime = attrs['exptime']
                factor = (exptime + self.detector_readtime[self.instr.upper()]) \
                         / exptime
                if isinstance(dset, Iterable):
                    dset = np.nanmedian(dset)

                # Multiply by correction factor to account for various instrument
                # readouts
                data_out[self.subgrp].append([factor * dset /
                                             self.detector_size[self.instr]] *\
                                             num_intervals)
                data_out['latitude'].append(attrs['latitude'])
                data_out['longitude'].append(attrs['longitude'])
                data_out['altitude'].append(attrs['altitude'])

        for key in data_out.keys():
            data_out[key] = [val for dset in data_out[key] for val in dset]
        return data_out



    def plot_full_path(self, df, i):
        self.fig = plt.figure(figsize=(9, 7))
        # Get the model for the SAA
        self.map = Basemap(projection='cyl')

        self.draw_map()

        # Generate an SAA contour
        saa = [list(t) for t in zip(*costools.saamodel.saaModel(i))]
        # Ensure the polygon representing the SAA is a closed curve by adding
        # the starting points to the end of the list of lat/lon coords
        saa[0].append(saa[0][0])
        saa[1].append(saa[1][0])
        self.map.plot(saa[1], saa[0],
                      c='r',
                      latlon=True,
                      label='SAA contour {}'.format(i))
        self.map.scatter(df['longitude'],df['latitude'], latlon=True)
        plt.show()




if __name__ == '__main__':
    main()
