#!/usr/bin/env python

from astropy.io import fits
import numpy as np
from scipy import ndimage

class CosmicRayLabel(object):
    def __init__(self, fname):
        self.fname = fname
        self.dq = None
        self.label = None

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
                  structure_element=np.ones((3, 3))):
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
        self.label, num_feat = ndimage.label(dq_bit,
                                             structure=structure_element)
        print('After thresholding there are {} objects'.format(num_feat))

    def generate_label(self):
        self.get_data()
        self.get_label()


def main():
    pass


if __name__ == '__main__':
    main()