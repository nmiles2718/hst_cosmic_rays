#!/usr/bin/env python

import matplotlib.pyplot as plt
plt.style.use('ggplot')
import dask.array as da
import h5py


def read_data(fname,subgrp):
     tmp =[]
     with h5py.File(fname, mode='r') as fobj:
         subgrp_ = fobj[subgrp]
         print('Total number of datasets: {}'.format(len(subgrp_.keys())))
         for name in subgrp_.keys():
             dset = subgrp_[name][:][1]
             tmp.append(da.from_array(dset,chunks=(20000)))
     x = da.concatenate(tmp, axis=0)
     x = x.rechunk('auto')
     masked_x = da.ma.masked_invalid(x)
     masked_x = da.ma.fix_invalid(masked_x, fill_value=-99)
     return masked_x

def plot_hist(size_data, shape_data,range=(0,10),bins=100):
    size_hist, size_edges = da.histogram(size_data, bins=bins, range=range)
    shape_hist, shape_edges = da.histogram(shape_data, bins=80, range=(0,0.9))
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2,figsize=(7,4))
    ax1.semilogy(size_edges[:-1], size_hist.compute(), drawstyle='steps-mid')
    ax1.set_xlabel('FWHM/(2*sqrt(2*ln(2))) [pixels]')
    ax2.set_xlabel('Symmetry')
    ax2.semilogy(shape_edges[:-1], shape_hist.compute(), drawstyle='steps-mid')
    fig.savefig('size_symmetry_hist.png',format='png',dpi=300)
    plt.show()


if __name__ == '__main__':
    main()