#!/usr/bin/env python

import argparse
from astropy.io import fits
from astropy.stats import sigma_clip
import numpy as np
import pandas as pd
import sys
import os

parser = argparse.ArgumentParser()

parser.add_argument('-fname',
                    help='/path/to/flshfile/',
                    type=str)
parser.add_argument('-fout',
                    help='filename to write to (default smoothed.fits)',
                    default='smoothed.fits')
parser.add_argument('-sigma_low',
                    help='Number of sigma to clip on the '
                         'low end (default is 5)',
                    type=int,
                    default=4)
parser.add_argument('-sigma_high',
                    help='Number of sigma to clip on'
                         ' the high end (default is 5)',
                    type=int,
                    default=5)
parser.add_argument('-box_size',
                    help='size of box used in smoothing (default is 20x20)',
                    type=int,
                    default=20)
parser.add_argument('-num_iter',
                    help='Number of iterations for smoothing '
                         '(default is 0)',
                    default=None,
                    type=int)




def first_pass(chip, threshold=17):
    tmp = np.where(chip > threshold)
    coords = list(zip(tmp[1], tmp[0]))
    return coords


def smooth(chip, coords, lowthres=4, highthres=5, dx=10, dy=10):
    locally_high_values = []
    locally_low_values = []
    y_max, x_max = chip.shape
    for (x, y) in coords:
        # Handle edge cases for bottom left corner of image
        if y < dy and x < dx:
            chip_slice = chip[y:y + 2 * dy, x:x + 2 * dx]
        # Handle edge cases for bottom right corner of image
        elif x + dx > x_max and y < dy:
            chip_slice = chip[y:y + 2 * dy, x:x - 2 * dx]
        # Handle edge cases for top left corner of image
        elif x < dx and y + dy > y_max:
            chip_slice = chip[y:y - 2 * dy, x:x + 2 * dx]
        # Handle edge cases for top right corner
        elif x + dx > x_max and y + dy > y_max:
            chip_slice = chip[y:y - 2 * dy, x:x - 2 * dx]
        # Handle edge cases for top border of image
        elif x + dx < x_max and x > dx and y + dy > y_max:
            chip_slice = chip[y:y - 2 * dy, x - dx:x + dx]
        # Handle edge cases for bottom border of image
        elif x + dx < x_max and x > dx and y < dy:
            chip_slice = chip[y:y + 2 * dy, x - dx:x + dx]
        # Handle edge cases for left side of image
        elif x < dx and y + dy < y_max:
            chip_slice = chip[y - dy:y + dy, x:x + 2 * dx]
        # Handle edge cases for right side of image
        elif x + dx > x_max and y + dy < y_max:
            chip_slice = chip[y - dy: y + dy, x:x - 2 * dx]
        # Normal pixels
        else:
            chip_slice = chip[y - dy:y + dy, x - dx:x + dx]

        median = np.median(chip_slice)
        std = np.std(chip_slice.flatten())
        # print(x, y, median, std)

        if chip[y][x] > median + highthres * std:
            locally_high_values.append((x, y))
            chip[y][x] = median
        elif chip[y][x] < median - lowthres * std:
            locally_low_values.append((x, y))
            chip[y][x] = median
    return locally_high_values, locally_low_values, chip


def mkRegion(coords, fname, color, dq_flag=1):
    gr.makeRegion_numpy(coords, os.getcwd() + os.sep + fname,
                        dq_flag, chip, color=color)


def iteratively_smooth(num_iter, data, coords, lowthres=4,
                       highthres=5, dx=10, dy=10):
    i = 0
    while i < num_iter:
        if i == 0:
            high_coords, low_coords, data_smooth = smooth(data, coords,
                                                          lowthres,highthres,
                                                          dx, dy)
        else:

            high_coords, low_coords, data_smooth = smooth(data_smooth,
                                                               coords,
                                                               lowthres,
                                                               highthres,
                                                               dx, dy)
        i += 1
    return high_coords, low_coords, data_smooth


def main(fname, sigma_low, sigma_high, box_size, num_iter, fout):
    with fits.open(fname) as hdu:
        data = hdu[0].data

    # generate a list of coordinates, start with all possible values
    coords = np.where(data > 0)
    coords = list(zip(coords[1], coords[0]))
    if not num_iter:
        high_coords, low_coords, chip_smooth = \
            smooth(data, coords=coords, lowthres=sigma_low,
                   highthres=sigma_high,
                   dx=round(box_size / 2),
                   dy=round(box_size / 2))
    else:
        high_coords, low_coords, chip_smooth = \
            iteratively_smooth(num_iter=num_iter,
                               data=data,
                               coords=coords,
                               lowthres=sigma_low,
                               highthres=sigma_high,
                               dx=round(box_size / 2),
                               dy=round(box_size / 2))
    # mkRegion(high_coords, 'smoothed_high_sigma_coords.fits',
    #          color='red')
    # mkRegion(low_coords, 'smoothed_low_sigma_coords.fits',
    #          color='green')
    print('Number of smoothed coords {}'.format(len(high_coords + low_coords)))
    hdu_o = fits.HDUList()
    hdu_o.append(fits.ImageHDU(chip_smooth))
    hdu_o[0].header['sigma_l'] = sigma_low
    hdu_o[0].header['sigma_h'] = sigma_high
    hdu_o[0].header['box_size'] = box_size
    hdu_o.writeto(fout,overwrite=True)


if __name__ == '__main__':
    args = parser.parse_args()
    fname = args.fname
    fout = args.fout
    sigma_low = args.sigma_low
    sigma_high = args.sigma_high
    box_size = args.box_size
    num_iter = args.num_iter
    main(fname, sigma_low, sigma_high, box_size, num_iter, fout)
