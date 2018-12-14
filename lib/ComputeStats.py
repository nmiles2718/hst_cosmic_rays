#!/usr/bin/env python

from astropy.io import fits
import numpy as np
from scipy import ndimage
import time

class ComputeStats(object):
    """
    Origin of coordinates (0,0)
    """
    def __init__(self, fname, instr, label, sci=None, integration_time=0):
        self.fname = fname
        self.instr = instr
        self.label = label
        self.max_x = label.shape[1]
        self.max_y = label.shape[0]
        self.int_ids = np.unique(label)[1:]
        self.cr_locs = np.asarray(ndimage.find_objects(label))
        self.sci = sci
        self.integration_time = integration_time
        self.shapes = {}
        self.sizes = {}
        self.cr_affected_pixels = []
        self.cr_deposition = None
        self.detector_readtime = {'ACS_WFC': 100./2.,
                                  'ACS_HRC': 30.0 /2,
                                  'WFC3_UVIS': 100./2.,
                                  'WFC3_IR': None,
                                  'STIS_CCD': 30.0 / 2.,
                                  'WFPC2': 60. / 2.}

    def get_data(self):
        """ Grab the SCI extensions from fits file
        """
        sci1 = ('sci',1) # Chip 2
        sci2 = ('sci',2) # Chip 1
        with fits.open(self.fname) as hdu:
            prhdr = hdu[0].header
            scihdr = hdu[1].header
            if 'exptime' in prhdr:
                self.integration_time += prhdr['exptime'] # add ~113/2 for readout
                if 'flashdur' in prhdr:
                    self.integration_time += prhdr['flashdur']
            elif 'exptime' in scihdr:
                self.integration_time += scihdr['exptime']
                if 'flashdur' in scihdr:
                    self.integration_time += scihdr['flashdur']
            self.integration_time += self.detector_readtime[self.instr]
            try:
                ext1 = hdu.index_of(sci1)
                ext1_data = hdu[ext1].data
            except KeyError:
                print('{1} is missing for {0}'.format(self.fname, sci1))
                ext1 = None
            try:
                ext2 = hdu.index_of(sci2)
                ext2_data = hdu[ext2].data
            except KeyError:
                print('{1} is missing for {0}'.format(self.fname, sci2))
                ext2 = None
        # If second DQ ext is missing, only work with the first
        # Otherwise combine each DQ ext to make full-frame
        if not ext2:
            self.sci = ext1_data
        else:
            self.sci = np.concatenate([ext1_data,ext2_data], axis=0)


    def compute_total_cr_deposition(self):
        """Apply image label for a single chip to its corresponding science
        extension.

        Parameters
        ----------
        label -- image label with cr information
        sci -- science extension

        Returns
        -------
        cr_sum -- total energy deposited by cosmic rays
        """

        cr_sum = ndimage.sum(self.sci, labels=self.label, index=self.int_ids)
        return cr_sum

    def compute_first_moment(self):
        """

        Parameters
        ----------
        sci
        label

        Returns
        -------

        """

        r_cm = ndimage.measurements.center_of_mass(self.sci,
                                                   labels=self.label,
                                                   index=self.int_ids)
        return np.asarray(r_cm)

    def compute_higher_moments(self, I_0, I_ci, grid_coords, index):
        """

        Parameters
        ----------
        I_0
        I_ci
        grid_coords
        index

        Returns
        -------

        """
        I_rr = [0, 0] # I_xx, I_yy
        I_xy = 0
        cr_coords = []
        for r_i in grid_coords:
            if self.label[r_i[0]][r_i[1]] == index:
                cr_coords.append(r_i)
                I_rr += (1 / I_0) * self.sci[r_i[0]][r_i[1]] \
                                 * (np.asarray(r_i) - np.asarray(I_ci)) ** 2
                I_xy += (1 / I_0) * self.sci[r_i[0]][r_i[1]] * \
                        (r_i[0] - I_ci[0]) * (r_i[1] - I_ci[1])
        return I_rr, I_xy, cr_coords

    def compute_second_moment(self, I_0, I_ci, grid_coords, index):
        """

        Parameters
        ----------
        I_0
        p_i
        r_i
        I_ci

        Returns
        -------

        """
        second_moment = [0, 0]
        for r_i in grid_coords:
            if self.label[r_i[0]][r_i[1]] == index:
                second_moment += (1 / I_0) * self.sci[r_i[0]][r_i[1]] \
                                 * (np.asarray(r_i) - np.asarray(I_ci)) ** 2

        return np.asarray(second_moment)

    def compute_cross_moment(self, I_0, I_ci, grid_coords, index):
        """

        Parameters
        ----------
        I_0
        sci
        label
        I_ci
        grid_coords
        index

        Returns
        -------

        """
        I_xy = 0
        for r_i in grid_coords:
            if self.label[r_i[0]][r_i[1]] == index:
                I_xy += (1 / I_0) * self.sci[r_i[0]][r_i[1]] * \
                        (r_i[0] - I_ci[0]) * (r_i[1] - I_ci[1])
        return I_xy

    def mk_grid(self, slice_tuple):
        """Build a meshgrid from a tuple of python slice objects

        Parameters
        ----------
        slice_tuple

        Returns
        -------
        (row, col) --> (y, x)
        """

        y_slice = slice_tuple[0]
        x_slice = slice_tuple[1]

        if int(x_slice.stop) == self.max_x and int(y_slice.stop) == self.max_y:
            # print('Cosmic ray struck the edge!!!!!!!!!!!!')
            y_coords = np.linspace(y_slice.start, y_slice.stop,
                                   int(y_slice.stop - y_slice.start) + 1,
                                   endpoint=False)

            x_coords = np.linspace(x_slice.start, x_slice.stop,
                                   int(x_slice.stop - x_slice.start) + 1,
                                   endpoint=False)

        elif int(y_slice.stop) == self.max_y:
            # print('Cosmic ray struck the edge!!!!!!!!!!!!')
            y_coords = np.linspace(y_slice.start, y_slice.stop,
                                   int(y_slice.stop - y_slice.start) + 1,
                                   endpoint=False)
            x_coords = np.linspace(x_slice.start, x_slice.stop,
                                   int(x_slice.stop - x_slice.start) + 1)
        elif int(x_slice.stop) == self.max_x:
            # print('Cosmic ray struck the edge!!!!!!!!!!!!')
            y_coords = np.linspace(y_slice.start, y_slice.stop,
                                   int(y_slice.stop - y_slice.start) + 1)
            x_coords = np.linspace(x_slice.start, x_slice.stop,
                                   int(x_slice.stop - x_slice.start) + 1,
                                   endpoint=False)
        else:
            y_coords = np.linspace(y_slice.start, y_slice.stop,
                                   int(y_slice.stop - y_slice.start) + 1)

            x_coords = np.linspace(x_slice.start, x_slice.stop,
                                   int(x_slice.stop - x_slice.start) + 1)

        xx, yy = np.meshgrid(x_coords, y_coords)
        positions = np.vstack([yy.ravel(), xx.ravel()])
        grid_coords = list(zip(map(int, positions[0]), map(int, positions[1])))
        return grid_coords

    def compute_size(self):
        """ Compute the size in "sigmas" of the cosmic ray.

        Parameters
        ----------
        label
        sci

        Returns
        -------

        """

        sizes = {}
        R_cm = self.compute_first_moment()
        cr_deposition = self.compute_total_cr_deposition()
        loop_gen = zip(self.int_ids, R_cm, cr_deposition, self.cr_locs)
        for int_id, r_cm, I_0, loc in loop_gen:
            grid_coords = self.mk_grid(loc)
            second_moment = self.compute_second_moment(I_0, r_cm,
                                                       grid_coords, int_id)
            sizes[int_id] = np.sqrt(second_moment.sum() / 2)
        return sizes

    def compute_shapes(self, int_it, r_cm, I_0, loc):
        """

        Parameters
        ----------
        label
        sci
        R_cm
        cr_deposition
        cr_locs

        Returns
        -------

        """

        shapes = {}
        R_cm = self.compute_first_moment()
        cr_deposition = self.compute_total_cr_deposition()
        loop_gen = zip(self.int_ids, R_cm, cr_deposition, self.cr_locs)
        for int_id, r_cm, I_0, loc in loop_gen:
            grid_coords = self.mk_grid(loc)
            second_moment = self.compute_second_moment(I_0, r_cm,
                                                  grid_coords, int_id)
            cross_moment = self.compute_cross_moment(I_0, r_cm,
                                                grid_coords, int_id)
            shapes[int_id] = np.sqrt(((second_moment[0] - second_moment[1])**2
                                    + 4*cross_moment**2)/(second_moment.sum())**2)

        return shapes

    def run_computation(self, int_id, r_cm, I_0, loc):
        # print('Analyzing {} at ({}, {})'.format(int_id, r_cm[1], r_cm[0]))
        grid_coords = self.mk_grid(loc)
        I_rr, I_xy, cr_coords = self.compute_higher_moments(I_0, r_cm,
                                                            grid_coords,
                                                            int_id)
        self.cr_affected_pixels.append(cr_coords)
        self.shapes[int_id] = np.sqrt(((I_rr[0] - I_rr[1]) ** 2
                                      + 4 * I_xy ** 2) / (I_rr.sum()) ** 2)
        self.sizes[int_id] = np.sqrt(I_rr.sum() / 2)

    def format_data(self, data):
        keys = list(data.keys())
        values = list(data.values())
        data_out = np.asarray([keys, values])
        return data_out

    def compute_stats(self):
        if self.sci is None:
            self.get_data()
        try:
            cr_incident_rate = float(len(self.int_ids))/self.integration_time
        except ZeroDivisionError:
            cr_incident_rate = np.nan

        R_cm = self.compute_first_moment()
        self.cr_deposition = self.compute_total_cr_deposition()

        loop_gen = zip(self.int_ids, R_cm, self.cr_deposition, self.cr_locs)
        # results = []
        start_time = time.time()
        for int_id, r_cm, I_0, loc in loop_gen:
            grid_coords = self.mk_grid(loc)
            I_rr, I_xy, cr_coords = self.compute_higher_moments(I_0, r_cm,
                                                    grid_coords, int_id)
            self.cr_affected_pixels.append(cr_coords)
            self.shapes[int_id] = np.sqrt(((I_rr[0] - I_rr[1]) ** 2
                                         + 4 * I_xy ** 2) / (I_rr.sum()) ** 2)
            self.sizes[int_id] = np.sqrt(I_rr.sum()/2)
            # datum = dask.delayed(self.run_computation)(int_id, r_cm, I_0, loc)
            # results.append(datum)
        # dask.compute(*results)
        end_time = time.time()
        print((end_time - start_time)/60)
        self.cr_affected_pixels = [a for data in self.cr_affected_pixels for a in data]
        self.sizes = self.format_data(self.sizes)
        self.shapes = self.format_data(self.shapes)
        return np.asarray(self.cr_affected_pixels), \
               np.asarray(cr_incident_rate),\
               self.sizes, \
               self.shapes, np.asarray([self.int_ids, self.cr_deposition])

