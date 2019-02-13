#!/usr/bin/env python

from astropy.io import fits
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from CosmicRayLabel import CosmicRayLabel
from ComputeStats import ComputeStats
import glob
import itertools
import scipy.ndimage as ndimage
import sys
sys.path.append('/Users/nmiles/animated_fits/lib')
from animate_data.animate_data import AnimateData

def get_deposition(read1, read2, diff = True):
    # grab read1
    ('Getting data from {}'.format(read1))
    cr_label_previous = CosmicRayLabel(read1)
    cr_label_previous.get_data()
    print('DQ Flags present in {}: {}'.format(read1,np.unique(cr_label_previous.dq)))
    cr_label_previous.dq = np.bitwise_and(cr_label_previous.dq, 8192)

    # Grab read2
    # print('Getting data from {}'.format(read2))
    cr_label = CosmicRayLabel(read2)
    cr_label.get_data()
    # print('DQ Flags present in {}: {}'.format(read2,np.unique(cr_label.dq)))
    cr_label.dq = np.bitwise_and(cr_label.dq, 8192)
    # Subtract previous read from current read
    if diff:
        print('Differencing DQ arrays')
        print('CR flags before: {}'.format(len(cr_label.dq[cr_label.dq == 8192])))
        cr_label.dq = cr_label.dq - cr_label_previous.dq
        print('CR flags after: {}'.format(len(cr_label.dq[cr_label.dq == 8192])))
    cr_label.get_label()
    if not diff:
        unique_crs = compare_label(cr_label.label, cr_label_previous.label)
        cr_label.label = unique_crs
    stats2 = ComputeStats(read2,cr_label.label)
    stats1 = ComputeStats(read1, cr_label.label)
    stats1.get_data()
    stats2.get_data()
    stats2.sci = stats2.sci - stats1.sci
    total_deposition = stats2.compute_total_cr_deposition()
    total_deposition*=fits.getval(read2,keyword='samptime',ext=('sci',1)) # scale by the samples integration time
    return total_deposition

def main(dirname, save=True):
    reads = glob.glob('{}/read*.fits'.format(dirname))
    num = [int(f.split('.')[0].split('_')[-1]) for f in reads]
    pairs = list(zip(reads, num))
    pairs.sort(key=lambda val: val[1])
    reads = list(zip(*pairs))[0]

    num_cr = []
    hist_data = []
    for i, read1 in enumerate(reads):
        if i == len(reads) - 1:
            continue
        depo = get_deposition(read1, reads[i + 1])
        read_num = fits.getval(reads[i + 1], keyword='sampnum',
                                ext=('sci', 1)) + 1
        hist, edges = np.histogram(np.log10(depo), bins=20,
                                    range=(1.25, 5.25))
        num_cr.append(num)
        hist_data.append((edges[:-1], hist))
    with fits.open(reads[0]) as hdu:
        hdr = hdu[0].header

    date = fits.getval(reads[0],'date-obs')
    args = [np.asarray(hist_data), 'hist', 'hist_{}.gif'.format(date), save]
    hist_obj = AnimateData(*args)
    hist_obj.xlim = (1, 6)
    hist_obj.ylim = (0, 150)
    hist_obj.xlabel = 'log10(electron deposition)'
    hist_obj.title = 'EXPTIME: {:.3f}, ' \
                     'SAMP_SEQ: {}, DATE-OBS: {}'.format(hdr['exptime'],
                                                        hdr['samp_seq'],
                                                         hdr['date-obs'])
    hist_obj.hist_plot()


if __name__ == '__main__':
    main()