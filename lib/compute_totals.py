#!/usr/bin/env python

from collections import defaultdict
from collections import namedtuple
import dask

import h5py
import glob
import pandas as pd



def tally_stats(fname):
    Stat = namedtuple('Stat', ['cr_count',
                                  'img_count',
                                  'total_exptime'])

    with h5py.File(fname,mode='r') as f:
        instr = list(f.keys())[0]
        print(instr)
        grp = f['/{}/sizes'.format(instr)]
        num_images = 0
        num_cr = 0
        total_exptime = 0
        for key in grp.keys():
            dset = grp[key][...]
            attrs = grp[key].attrs
            # print(list(attrs.items()))
            num_cr += dset.shape[1]
            num_images += 1
            total_exptime += attrs['exptime']

    result = Stat(cr_count=num_cr,
                  img_count=num_images,
                  total_exptime=total_exptime)
    print(result)

    return instr, result


def compile_global_stats(data_dir='./../data/*/*cr_sizes_?.hdf5'):
    flist = glob.glob(data_dir)
    output = defaultdict(list)
    flist = [f for f in flist if 'nicmos' not in f]
    print(flist)
    flist.append('./../data/STIS/stis_cr_sizes.hdf5')
    results = [dask.delayed(tally_stats)(f) for f in flist]
    results = list(dask.compute(*results, scheduler='processes'))

    for instr, data in results:
        output[instr].append(data)

    for key in output.keys():
        cr_count = 0
        img_count = 0
        total_exptime = 0
        for val in output[key]:
            cr_count += val.cr_count
            img_count += val.img_count
            total_exptime += val.total_exptime
        output[key] = [cr_count, img_count, total_exptime]

    df = pd.DataFrame(output, index=['cr_count', 'img_count', 'total_exptime'])
    print(df)
    print('Total CR count: {}'.format(df.loc['cr_count', :].sum()))
    print('Total number of images analyzed: {}'.format(df.loc['img_count', :].sum()))
    print('Cumulative exposure time: {}'.format(df.loc['total_exptime', :].sum()))

if __name__ == '__main__':
    compile_global_stats()
