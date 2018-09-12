#!/usr/bin/env python

from collections import defaultdict

from astropy.time import Time
import dask.array as da
import costools
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.basemap import Basemap
import pandas as pd

plt.style.use('ggplot')


class PlotData(object):
    def __init__(self, fname, instr, subgrp):
        self.fname = fname
        self.instr = instr
        self.data = []
        self.subgrp = subgrp
        self.ax = None
        self.fig = None

    def plot_data(self, ax=None, bins=30, range=[0, 3], fill_value=-1,c='r'):
        """ plot a histogram of data, defaults are set for sizes

        Parameters
        ----------
        subgrp
        bins
        range

        Returns
        -------

        """
        tmp = []
        with h5py.File(self.fname, mode='r') as fobj:
            subgrp_ = fobj[self.instr.upper()+'/'+self.subgrp]
            print(subgrp_)
            for name in subgrp_.keys():
                dset = np.log10(subgrp_[name][:][1])
                tmp.append(da.from_array(dset,chunks=(20000)))
        x = da.concatenate(tmp, axis=0)
        print(x.shape)
        self.data = da.ma.fix_invalid(x, fill_value=fill_value)
        h, edges = da.histogram(self.data, bins=bins,
                                range=range, density=True)

        # Create an axis if it doesnt exists
        if ax is None:
            self.fig, self.ax = plt.subplots(figsize=(5,5),
                                      nrows=1,
                                      ncols=1)
            # Normalize by the highest count
            # self.data[subgrp] /= np.nanmax(self.data[subgrp])

            # self.ax.step(edges[:-1], h.compute(), color='r')
            self.ax.semilogy(edges[:-1], h.compute(),
                             label='{}/{}'.format(*self.instr.split('_')),
                             drawstyle='steps-mid', color=c)
        else:
            ax.semilogy(edges[:-1], h.compute(),
                        label='{}/{}'.format(*self.instr.split('_')),
                        drawstyle='steps-mid', color=c)
        # plt.show()

    def plot_rate_vs_time(self):
        data=defaultdict(list)
        with h5py.File(self.fname, mode='r') as fobj:
            subgrp_ = fobj[self.instr+'/'+self.subgrp]
            for name in subgrp_.keys():
                dset = subgrp_[name]
                if dset.value == 0:
                    continue
                    print('Error analyzing {}'.format(key))
                data['rate'].append(dset.value/37.748)
                d = dset.attrs['date']
                t = dset.attrs['time']
                date_obs = Time('{} {}'.format(d, t),
                                format='iso')
                data['date'].append(date_obs)

        data['mjd'] = [date.mjd for date in data['date']]
        index = pd.DatetimeIndex([val.iso for val in data['date']])
        df = pd.DataFrame(data, index=index)
        df.sort_index(inplace=True)
        df1 = df[['rate','mjd']]
        averaged = df1.rolling(window='60D', min_periods=60).mean()
        print(averaged)
        averaged_no_nan = averaged.dropna()
        plt.scatter([Time(val, format='mjd').to_datetime() for val in averaged_no_nan['mjd']]
                        ,averaged_no_nan['rate'])
        plt.ylabel('Cosmic Ray Rate [CR/s/cm^2]')
        plt.title('ACS/WFC Smoothed Cosmic Ray Rate')
        plt.savefig('cr_incidence_rolling_average.png',)
        plt.show()


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
                   latlon=10,
                   c='w',label='z={:.2f}km'.format(alt_))
        # plt.legend(bbox_to_anchor=(0., 0.95, 1., .102), loc=3, ncol=4,
        #            mode="expand", borderaxespad=0.)
        plt.show()





if __name__ == '__main__':
    main()