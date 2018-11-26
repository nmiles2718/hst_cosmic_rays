#!/usr/bin/env python

from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.stats import sigma_clipped_stats
from astropy.visualization import ImageNormalize, LinearStretch, ZScaleInterval

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from photutils import detect_sources, deblend_sources
from scipy import ndimage


class CosmicRayLabel(object):
    def __init__(self, fname):
        self.fname = fname
        self.dq = None
        self.sci = None
        self.label = None
        self.integration_time = 0

    def get_data(self, ext='dq'):
        """ Grab the DQ extensions from fits file
        """

        dq1 = (ext,1) # Chip 2
        dq2 = (ext,2) # Chip 1
        with fits.open(self.fname) as hdu:
            try:
                ext1 = hdu.index_of(dq1)
                ext1_data = hdu[ext1].data
            except KeyError:
                print('{1} is missing for {0}'.format(self.fname, dq1))
                ext1 = None
            try:
                ext2 = hdu.index_of(dq2)
                ext2_data = hdu[ext2].data
            except KeyError:
                print('{1} is missing for {0}'.format(self.fname, dq2))
                ext2 = None
        # If second DQ ext is missing, only work with the first
        # Otherwise combine each DQ ext to make full-frame
        if not ext2 and ext=='dq':
            self.dq = ext1_data
        elif ext=='dq':
            self.dq = np.concatenate([ext1_data,ext2_data], axis=0)
        elif not ext2 and ext=='sci':
            self.sci = ext1_data
        elif ext=='sci':
            self.sci = np.concatenate([ext1_data, ext2_data], axis=0)

    def get_label(self, bit_flag=8192, bit_comp=True, samptime=None,
                  threshold_l=2, threshold_u = 1000,
                  structure_element=np.ones((3, 3)), wfpc2=False):
        """ Generate a label based of a dq array using ndimage.label()

        Parameters
        ----------
        dq : DQ data extension
        bit_flag : BIT value to search for
        structure_element : structure element for labeling

        Returns
        -------
        label
        num_feat
        """
        if self.dq is None:
            if samptime is None:
                mean, median, std = sigma_clipped_stats(self.sci[self.sci > 0],
                                                        sigma_lower=3,
                                                        sigma_upper=3)
            else:
                # Nicmos avg dark current is ~0.1 e/s
                # We use this information to ensure the amplifier glow does not
                # contaminate the sigma clipped statistics

                # Do the first round of sigma clipping
                mean, median, std = sigma_clipped_stats(self.sci[self.sci > 0],
                                                        sigma_lower=3,
                                                        sigma_upper=3)

                # Do the second round
                mean, median, std = sigma_clipped_stats(
                    self.sci[(self.sci >= median - 3*std) &
                             (self.sci <= median + 3 * std )],
                    sigma_lower=3,
                    sigma_upper=3
                )

            print('mean: {}, median: {}, std: {}'.format(mean, median, std))
            self.dq = np.where(self.sci > median + 5 * std, 1, 0)

        elif bit_comp:
            bad_pixels = np.bitwise_and(self.dq, 4)
            unstable_pixels = np.bitwise_and(self.dq, 32)
            crs = np.bitwise_and(self.dq, bit_flag)
            self.dq = np.where((crs > 0) &
                               (bad_pixels==0) &
                               (unstable_pixels==0), bit_flag, 0)

        label, num_feat = ndimage.label(self.dq,
                                             structure=structure_element)
        print('A total of {} objects were identified'.format(num_feat))

        cr_labels = label.ravel()  # Returns a flattened label
        # Count up the number of pixels associated with each unique label
        sizes = np.bincount(cr_labels)
        arg_max = np.argmax(sizes)
        sizes[arg_max] = 0
        large_CRs = (sizes > threshold_l) & (sizes < threshold_u)

        # Create a 2-D mask from the 1-D array of large cosmic rays, and set all
        # labels of cosmic rays smaller than threshold to 0 so they are ignored.
        label_mask = large_CRs[label]
        self.dq[~label_mask] = 0
        label, num_feat = ndimage.label(self.dq,
                                             structure=structure_element)
        print('After thresholding there are {} objects'.format(num_feat))
        if not wfpc2:
            self.label = label
        else:
            return label

    def get_wfpc2_data(self):
        """ Grab the SCI extensions from WFPC2 fits file """

        pc = ('sci', 1)  # Chip 2
        wf2 = ('sci', 2)  # Chip 1
        wf3 = ('sci', 3)
        wf4 = ('sci', 4)
        detector_data = []
        with fits.open(self.fname) as hdu:
            prhdr = hdu[0].header
            scihdr = hdu[1].header
            gain = prhdr['ATODGAIN']
            if 'exptime' in prhdr:
                self.integration_time += prhdr['exptime']  # add ~113/2 for readout
                if 'flashdur' in prhdr:
                    self.integration_time += prhdr['flashdur']
            elif 'exptime' in scihdr:
                self.integration_time += scihdr['exptime']
                if 'flashdur' in scihdr:
                    self.integration_time += scihdr['flashdur']

            for ext in [pc, wf2, wf3, wf4]:
                try:
                    ext = hdu.index_of(ext)
                    ext_data = gain * hdu[ext].data
                except KeyError:
                    print('{1} is missing for {0}'.format(self.fname, ext))
                    ext1 = None
                else:
                    detector_data.append(ext_data)
        # If second ext is missing, only work with the first
        # Otherwise combine each DQ ext to make full-frame
        self.sci = detector_data

    def label_wfpc2_data(self):
        """ Generate a label for each CCD chip in the WFPC2 detector

        Parameters
        ----------
        deblend

        Returns
        -------
        list of segmentation labels
        """
        self.get_wfpc2_data()
        self.label = []
        for sci_data in self.sci:

            # Generate some stats to use for the source detection
            mean, median, std = sigma_clipped_stats(sci_data, sigma_lower=3,
                                                    sigma_upper=3)
            print('mean: {}, median: {}, std: {}'.format(mean, median, std))

            self.dq = np.where(sci_data > np.absolute(median) + 3*std, 1, 0)
            self.label.append(self.get_label(wfpc2=True,
                                             bit_comp=False,
                                             threshold_l=2, threshold_u=500))

    def generate_label(self, use_dq=True, threshold_l=None, threshold_u=None):
        if use_dq:
            self.get_data(ext='dq')
        else:
            self.get_data(ext='sci')
        self.get_label(threshold_l=threshold_l, threshold_u=threshold_u)

    def mk_fig(self):
        """ Generate a figure with two axes for plotting the label

        Returns
        -------
        fig
        ax1
        ax2
        """
        grid = plt.GridSpec(1, 2, wspace=0.2, hspace=0.15)
        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot(grid[0, 0])
        ax2 = fig.add_subplot(grid[0, 1], sharex=ax1, sharey=ax1)
        for ax in [ax1, ax2]:
            ax.grid(False)
            ax.tick_params(axis='both', bottom=False,
                           labelbottom=False,
                           left=False,
                           labelleft=False)
        return fig, ax1, ax2

    def plot(self):
        """ Plot the label

        Returns
        -------

        """

        fig, ax1, ax2 = self.mk_fig()
        ncolors = np.max(self.label) + 1
        prng = np.random.RandomState(1234)
        h = prng.uniform(low=0.0, high=1.0, size=ncolors)
        s = prng.uniform(low=0.2, high=0.7, size=ncolors)
        v = prng.uniform(low=0.5, high=1.0, size=ncolors)
        hsv = np.dstack((h, s, v))

        rgb = np.squeeze(colors.hsv_to_rgb(hsv))
        rgb[0] = (0,0,0)
        cmap = colors.ListedColormap(rgb)
        norm = ImageNormalize(self.sci,
                              stretch=LinearStretch(),
                              interval=ZScaleInterval())
        ax1.imshow(self.sci, norm=norm, cmap='gray', origin='lower')
        ax2.imshow(self.label, cmap=cmap, origin='lower')
        plt.show()

    def generate_label(self):
        self.get_data()
        self.get_label()


def main():
    pass


if __name__ == '__main__':
    main()
