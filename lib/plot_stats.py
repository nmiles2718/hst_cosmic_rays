#!/usr/bin/env python

from collections import defaultdict

from astropy.io import fits
import costools
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.basemap import Basemap

plt.style.use('ggplot')


class PlotData(object):
    def __init__(self, fname, instr, subgrp):
        self.fname = fname
        self.instr = instr
        self.data = defaultdict(list)
        self.subgrp = subgrp
        self.ax = None
        self.fig = None

    def plot_data(self, bins=30, range=(0,3), log=False):
        """ plot a histogram of data, defaults are set for sizes

        Parameters
        ----------
        subgrp
        bins
        range

        Returns
        -------

        """
        with h5py.File(self.fname, mode='r') as fobj:
            subgrp_ = fobj[self.instr+'/'+self.subgrp]
            for name in subgrp_.keys():
                dset = subgrp_[name]
                self.data[self.subgrp] = np.append(self.data[self.subgrp],
                                              dset[:][1])

        # Create an axis if it doesnt exists
        self.fig, self.ax = plt.subplots(figsize=(5,5),
                                  nrows=1,
                                  ncols=1)
        # Normalize by the highest count
        # self.data[subgrp] /= np.nanmax(self.data[subgrp])
        hist, edges = np.histogram(self.data[self.subgrp],
                                  bins=bins,
                                  range=range)
        hist = hist / hist.sum()
        self.ax.step(edges[:-1], hist, color='r')
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