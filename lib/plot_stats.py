#!/usr/bin/env python

from collections import defaultdict

from astropy.time import Time
import dask.array as da
import costools
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc,rcParams
rc('font', weight='bold', family='sans-serif')
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.basemap import Basemap
import pandas as pd

plt.style.use('ggplot')


class PlotData(object):
    def __init__(self, fname, instr, subgrp, flist):
        self.fname = fname
        self.instr = instr.upper()
        self.flist = flist
        self.data = None
        self.data_df = None
        self.subgrp = subgrp
        self.num_images = 0
        self.ax = None
        self.fig = None
        # Detector sizes in cm^2
        self.detector_size ={'ACS_WFC': 37.748,
                             'ACS_HRC': 4.624 ,
                             'WFC3_UVIS': 37.804,
                             'WFC3_IR': 3.331,
                             'STIS_CCD': 4.624,
                             'WFPC2':5.76}

    def read_rate(self):
        data_out = defaultdict(list)
        for f in self.flist:
            fobj = h5py.File(f, mode='r')
            sgrp = fobj[self.instr.upper()+'/'+self.subgrp]
            for key in sgrp.keys():
                dset = sgrp[key].value
                attrs = sgrp[key].attrs
                data_out[self.subgrp].append(np.nanmedian(dset))
                for at_key in attrs.keys():
                    val = attrs[at_key]
                    if at_key == 'date':
                        val = Time(val, format='iso')
                    data_out['{}'.format(at_key)].append(val)
        data_out['mjd'] = [val.mjd for val in data_out['date']]
        date_index = pd.DatetimeIndex([val.iso for val in data_out['date']])
        self.data_df = pd.DataFrame(data_out, index = date_index)
        self.data_df.sort_index(inplace=True)
    

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


    def plot_rate_vs_time(self, ax= None, ptype='rolling', window='20D', min_periods=20):
        if self.data_df is None:
            self.read_rate()
        flags = self.data_df.exptime.gt(800)
        df1 = self.data_df[['incident_cr_rate','mjd']][flags]

        if ptype == 'rolling':
            averaged = df1.rolling(window=window, min_periods=min_periods).mean()
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
        avg_no_nan.loc[:,'incident_cr_rate'] = avg_no_nan['incident_cr_rate']/\
                                               self.detector_size[self.instr]

        self.ax.scatter([Time(val, format='mjd').to_datetime()
                         for val in avg_no_nan['mjd']],
                        avg_no_nan['incident_cr_rate'],
                        label=self.instr)
        self.ax.set_xlabel('Date')
        self.ax.set_ylabel('Cosmic Ray Rate [CR/s/cm^2]')
        self.ax.set_title('Smoothed Cosmic Ray Rate')
       # plt.savefig('cr_incidence_rolling_average.png',)
       # plt.show()


    def plot_hst_loc(self):
        with h5py.File(self.fname, mode='r') as fobj:
            subgrp_ = fobj[self.instr+'/'+self.subgrp]
            for name in subgrp_.keys():
                dset = subgrp_[name]
                attrs = dset.attrs
                self.data['lat'].append(attrs['latitude'])
                self.data['lon'].append(attrs['longitude'])
                self.data['altitude'].append(attrs['altitude'])
                self.data['expstart'].append(attrs['expstart'])

        self.fig = plt.figure(figsize=(7, 5))
        # Get the model for the SAA
        m = Basemap(projection='cyl')
        m.bluemarble()
        for i in range(32):
            saa = [list(t) for t in zip(*costools.saamodel.saaModel(i))]
            saa[0].append(saa[0][0])
            saa[1].append(saa[1][0])
            m.plot(saa[1], saa[0],
                   c='r',
                   latlon=True,
                   label='SAA contour {}'.format(i))

        # m.drawcoastlines()
        # m.fillcontinents(color='oldlace', lake_color='#c7d7e5')
        # draw parallels and meridians.
        m.drawparallels(np.arange(-90., 91., 30.))
        m.drawmeridians(np.arange(-180., 181., 45))
        # m.drawmapboundary(fill_color='#c7d7e5')
        m.plot(saa[1],saa[0],
               c='r',
               latlon=True,
               label='SAA contour')
        generator = zip(self.data['lat'],
                        self.data['lon'],
                        self.data['altitude'])
        for lat_, lon_, alt_ in generator:
            m.plot(lon_, lat_,
                   marker='+',
                   markersize=10,
                   latlon=True,
                   c='w',label='z={:.2f}km'.format(alt_))
        # plt.legend(bbox_to_anchor=(0., 0.95, 1., .102), loc=3, ncol=4,
        #            mode="expand", borderaxespad=0.)
        plt.show()





if __name__ == '__main__':
    main()
