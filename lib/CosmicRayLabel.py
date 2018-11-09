#!/usr/bin/env python

from astropy.io import fits
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma
from astropy.stats import sigma_clipped_stats
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
        if not ext2:
            self.dq = ext1_data
        else:
            self.dq = np.concatenate([ext1_data,ext2_data], axis=0)
     

    def get_label(self, bit_flag=8192, bit_comp=True,threshold=2,
                  structure_element=np.ones((3, 3)), wfpc2=False):
        """ Generate a label based of a dq array using ndimage.label()

        Parameters
        ----------
        dq -- DQ data extension
        bit_flag -- BIT value to search for
        structure_element -- structure element for labeling

        Returns
        -------
        label
        num_feat
        """
        if bit_comp:
            dq_bit = np.bitwise_and(self.dq , bit_flag)
        else:
            dq_bit = self.dq
        label, num_feat = ndimage.label(dq_bit,
                                             structure=structure_element)
        print('A total of {} objects were identified'.format(num_feat))

        cr_labels = label.ravel()  # Returns a flattened label
        # Count up the number of pixels associated with each unique label
        sizes = np.bincount(cr_labels)
        arg_max = np.argmax(sizes)
        sizes[arg_max] = 0
        large_CRs = sizes > threshold

        # Create a 2-D mask from the 1-D array of large cosmic rays, and set all
        # labels of cosmic rays smaller than threshold to 0 so they are ignored.
        label_mask = large_CRs[label]
        dq_bit[~label_mask] = 0
        label, num_feat = ndimage.label(dq_bit,
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

    def label_wfpc2_data(self, deblend=False):
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


        # Generate a kernel for use in the segmentation mapping, normalize it's value to 1
        # sigma = 2. * gaussian_fwhm_to_sigma  # convert FWHM of 2.75 pix to sigma
        # kernel = Gaussian2DKernel(sigma, x_size=5, y_size=5)
        # kernel.normalize()
        for sci_data in self.sci:

            # Generate some stats to use for the source detection
            mean, median, std = sigma_clipped_stats(sci_data, sigma_lower=3,
                                                    sigma_upper=3)
            print('mean: {}, median: {}, std: {}'.format(mean, median, std))

            self.dq = np.where(sci_data > median + 5*std, 1, 0)
            self.label.append(self.get_label(wfpc2=True,
                                             bit_comp=False,
                                             threshold=2))
            # Generate a segmentation map based on identified sources
            # segm = detect_sources(sci_data - median,
            #                       threshold=median + 10 * std,
            #                       npixels=4,
            #                       filter_kernel=kernel,
            #                       connectivity=8)
            # if deblend:
            #     # Deblend sources
            #     print('deblending')
            #     segm_deblend = deblend_sources(sci_data - median,
            #                                    segm.data,
            #                                    npixels=4,
            #                                    nlevels=32,
            #                                    filter_kernel=kernel,
            #                                    contrast=0.1,
            #                                    connectivity=8,
            #                                    )
            #     segm = segm_deblend
            # self.label.append(segm.data)



    def generate_label(self):
        self.get_data()
        self.get_label()


def main():
    pass


if __name__ == '__main__':
    main()
