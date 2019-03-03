#!/usr/bin/env python

import argparse
from astropy.io import fits
import os

parser = argparse.ArgumentParser()
parser.add_argument('-fname',help='file to process')


def edit_hdr(hdr):
    hdr['extver'] = 1
    return hdr


def write_out(f_in='', f_out='', exts=None):
    sci, err, dq, samp, time = next(exts)
    with fits.open(f_in) as hdu:
        sci_ext = hdu[hdu.index_of(sci)]
        err_ext = hdu[hdu.index_of(err)]
        dq_ext = hdu[hdu.index_of(dq)]
        samp_ext = hdu[hdu.index_of(samp)]
        time_ext = hdu[hdu.index_of(time)]
        hdu_list = fits.HDUList()
        hdu_list.append(fits.PrimaryHDU(header=hdu[0].header))
        hdu_list.append(fits.ImageHDU(data = sci_ext.data,
                                      header = edit_hdr(sci_ext.header)))
        hdu_list.append(fits.ImageHDU(data = err_ext.data,
                                      header = edit_hdr(err_ext.header)))
        hdu_list.append(fits.ImageHDU(data = dq_ext.data,
                                      header = edit_hdr(dq_ext.header)))
        hdu_list.append(fits.ImageHDU(data = samp_ext.data,
                                      header = edit_hdr(samp_ext.header)))
        hdu_list.append(fits.ImageHDU(data = time_ext.data,
                                      header = edit_hdr(time_ext.header)))
    hdu_list.writeto(f_out,overwrite=True)


def mkdir(fname):
    dirname = os.path.dirname(fname)
    basename = os.path.basename(fname).split('_')[0]
    new_dir = dirname+'/'+basename
    try:
        os.mkdir(new_dir)
    except Exception as e:
        print(e)
        r = input('Overwrite current data? (y/n) ')
        if r == 'y':
            return new_dir
        return -1
    else:
        return new_dir


def main(fname):
    dirname = mkdir(fname)
    sci_exts = [('sci', n) for n in range(1, 17)]
    err_exts = [('err', n) for n in range(1, 17)]
    dq_exts = [('dq', n) for n in range(1, 17)]
    samp_exts = [('samp', n) for n in range(1, 17)]
    time_exts = [('time', n) for n in range(1, 17)]
    exts = zip(sci_exts, err_exts, dq_exts, samp_exts, time_exts)
    if isinstance(dirname, str):
        for i in reversed(range(1, 17)): # [16, 15, ..., 1]
            write_out(f_in=fname, f_out='{}/read_{}.fits'.format(dirname, i),
                      exts=exts)
    else:
        print('check directory name')

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.fname)