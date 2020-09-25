import argparse
from collections import defaultdict
import glob
import sys

from astropy.time import Time
import datahandler as dh
import dask.array as da
import h5py
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()

parser.add_argument('-instr',
                    type=str,
                    default='ACS_WFC')


def compute_percentile(dset):
    percentiles = da.percentile(dset, q=[25, 50, 75]).compute()
    return percentiles


def main(instr):
    shapes = dh.DataReader(instr=instr.upper(), statistic='shapes')
    sizes = dh.DataReader(instr=instr.upper(), statistic='sizes')
    energies = dh.DataReader(instr=instr.upper(), statistic='energy_deposited')


    for r in [shapes, sizes, energies]:
        r.find_hdf5()
    shapes.read_cr_stat(units=None, min_exptime=200)
    shape_stats = compute_percentile(shapes._shape)

    sizes.read_cr_stat(units='pixels', min_exptime=200)
    size_stats_pixels = compute_percentile(sizes._size_pixels)

    sizes.read_cr_stat(units='sigmas', min_exptime=200)
    size_stats_sigmas = compute_percentile(sizes._size_sigmas)

    energies.read_cr_stat(units=None, min_exptime=200)
    energy_stats = compute_percentile(energies._energy_deposited)

    data = defaultdict(list)
    labels = ['shape','size_pix','size_sig','energy']
    loop = zip(
        labels, [shape_stats, size_stats_pixels, size_stats_sigmas, energy_stats]
    )
    for l, st in loop:
        data[f"{l}_25%"].append(st[0])
        data[f"{l}_50%"].append(st[1])
        data[f"{l}_75%"].append(st[2])

    df = pd.DataFrame(data)
    df.to_csv(f'{instr}_summary_statistics.txt', header=True, index=True)


def run_all():
    instrs = ['ACS_WFC','ACS_HRC','WFPC2','WFC3_UVIS','STIS_CCD']
    for instr in instrs:
        print(instr)
        main(instr)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.instr)
    #run_all()

