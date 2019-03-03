#!/usr/bin/env python

from astropy.io import fits
import os
import numpy as np


class ProcessIR(object):
    def __init__(self, fname):
        self.fname = fname
        self.dirname = os.path.dirname(fname)
        self.nsamp = None
        self.exts = None

    def make_exts(self):
        with fits.open(self.fname) as hdu:
            self.nsamp = hdu[0].header['nsamp']
            print('The number of non-destructive reads is: {}'.format(self.nsamp))
        sci_exts = [('sci', n) for n in range(1, self.nsamp + 1)]
        err_exts = [('err', n) for n in range(1, self.nsamp + 1)]
        dq_exts = [('dq', n) for n in range(1, self.nsamp + 1)]
        samp_exts = [('samp', n) for n in range(1, self.nsamp + 1)]
        time_exts = [('time', n) for n in range(1, self.nsamp + 1)]
        self.exts = zip(sci_exts, err_exts, dq_exts, samp_exts, time_exts)

    def edit_hdr(self, hdr):
        hdr['extver'] = 1
        return hdr

    def write_ext(self, f_out='', exts=None):
        sci, err, dq, samp, time = next(exts)
        with fits.open(self.fname) as hdu:
            sci_ext = hdu[hdu.index_of(sci)]
            err_ext = hdu[hdu.index_of(err)]
            dq_ext = hdu[hdu.index_of(dq)]
            samp_ext = hdu[hdu.index_of(samp)]
            time_ext = hdu[hdu.index_of(time)]
            hdu_list = fits.HDUList()
            hdu_list.append(fits.PrimaryHDU(header=hdu[0].header))
            hdu_list.append(fits.ImageHDU(data=sci_ext.data,
                                          header=self.edit_hdr(sci_ext.header)))
            hdu_list.append(fits.ImageHDU(data=err_ext.data,
                                          header=self.edit_hdr(err_ext.header)))
            hdu_list.append(fits.ImageHDU(data=dq_ext.data,
                                          header=self.edit_hdr(dq_ext.header)))
            hdu_list.append(fits.ImageHDU(data=samp_ext.data,
                                          header=self.edit_hdr(samp_ext.header)))
            hdu_list.append(fits.ImageHDU(data=time_ext.data,
                                          header=self.edit_hdr(time_ext.header)))
        hdu_list.writeto(f_out, overwrite=True)

    def write_out(self):
        self.make_exts()
        for i in reversed(range(1, self.nsamp + 1)):
            self.write_ext(f_out=self.dirname + '/read_{}.fits'.format(i),
                           exts=self.exts)


if __name__ == '__main__':
    main()