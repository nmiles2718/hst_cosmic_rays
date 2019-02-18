import logging

import numpy as np
from scipy import ndimage

logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s')

LOG = logging.getLogger()

LOG.setLevel(logging.INFO)


class ComputeStats(object):
    """
    Origin of coordinates (0,0)
    """

    def __init__(self, CosmicRayLabel):
        """

        Parameters
        ----------
        CosmicRayLabel : :py:class:`~label.labeler.CosmicRayLabel`
            Instance of the `CosmicRayLabel` class

        """
        self._fname = CosmicRayLabel.fname
        self._label = CosmicRayLabel.label
        self._instr_cfg = CosmicRayLabel.instr_cfg
        self._integration_time = CosmicRayLabel.integration_time
        self._sci = CosmicRayLabel.sci

        self._cr_incident_rate = None
        self._max_x = self._label.shape[1]
        self._max_y = self._label.shape[0]
        self._label_ids = np.unique(self._label)[1:]
        self._cr_positions = np.asarray(ndimage.find_objects(self._label))

        self._centroids = None
        self._energy_deposition = None
        self._shapes = []
        self._size_in_sigmas = []
        self._size_in_pixels = []
        self._cr_affected_pixels = []


    @property
    def cr_incident_rate(self):
        return self._cr_incident_rate

    @cr_incident_rate.getter
    def cr_incident_rate(self):
        """Compute cosmic ray incidence rate"""
        return self._cr_incident_rate

    @cr_incident_rate.setter
    def cr_incident_rate(self, value):
        self._cr_incident_rate = value

    @property
    def centroids(self):
        return self._centroids

    @centroids.getter
    def centroids(self):
        """Centroids of each cosmic ray"""
        return self._centroids

    @centroids.setter
    def centroids(self, value):
        self._centroids = value

    @property
    def cr_affected_pixels(self):
        return self._cr_affected_pixels

    @cr_affected_pixels.getter
    def cr_affected_pixels(self):
        return self._cr_affected_pixels

    @cr_affected_pixels.setter
    def cr_affected_pixels(self, value):
        self._cr_affected_pixels = value

    @property
    def energy_deposition(self):
        return self._energy_deposition

    @energy_deposition.getter
    def energy_deposition(self):
        """Total energy deposited by cosmic rays"""
        return self._energy_deposition

    @energy_deposition.setter
    def energy_deposition(self, value):
        self._energy_deposition = value

    @property
    def fname(self):
        return self._fname

    @fname.getter
    def fname(self):
        """Name of FITS file"""
        return self._fname

    @property
    def label(self):
        return self._label

    @label.getter
    def label(self):
        """Generated label"""
        return self._label

    @property
    def instr_cfg(self):
        return self._instr_cfg

    @instr_cfg.getter
    def instr_cfg(self):
        return self._instr_cfg

    @property
    def integration_time(self):
        return self._integration_time

    @integration_time.getter
    def integration_time(self):
        """Total integration time of the observation"""
        return self._integration_time

    @property
    def label_ids(self):
        return self._label_ids

    @label_ids.getter
    def label_ids(self):
        """List of integer IDs used in the label"""
        return self._label_ids

    @property
    def sci(self):
        return self._sci

    @sci.getter
    def sci(self):
        """A concatenated version of all SCI extensions in :py:attr:`fname`"""
        return self._sci

    @property
    def size_in_pixels(self):
        return self._size_in_pixels

    @size_in_pixels.getter
    def size_in_pixels(self):
        return self._size_in_pixels

    @size_in_pixels.setter
    def size_in_pixels(self, value):
        self._size_in_pixels = value

    @property
    def size_in_sigmas(self):
        return self._size_in_sigmas

    @size_in_sigmas.getter
    def size_in_sigmas(self):
        """Width of cosmic rays energy distribution in units of sigmas"""
        return self._size_in_sigmas

    @size_in_sigmas.setter
    def size_in_sigmas(self, value):
        self._size_in_sigmas = value

    @property
    def shapes(self):
        return self._shapes

    @shapes.getter
    def shapes(self):
        """Shapes of cosmic ray energy distribution"""
        return self._shapes

    @shapes.setter
    def shapes(self, value):
        self._shapes = value

    def compute_total_cr_deposition(self):
        """Compute the total number of electrons deposited at each label.

        Returns
        -------
        cr_sum : numpy.ndarray
            Sum of all pixels at each label in :py:attr:`label_ids`

        """

        cr_sum = ndimage.sum(self.sci,
                             labels=self.label,
                             index=self.label_ids)
        self.energy_deposition = np.asarray(cr_sum)

    def compute_first_moment(self, sci=None):
        """Compute the first moment of energy deposited

        This corresponds to the flux-weighted centroid of the cosmic ray

        Parameters
        ----------
        sci : numpy.ndarray [py:attr:`CosmicRayLabel.sci`]
            SCI extension of the FITS image.

        Returns
        -------

        """
        if sci is None:
            data = self.sci
        else:
            data = sci

        r_cm = ndimage.measurements.center_of_mass(data,
                                                   labels=self.label,
                                                   index=self.label_ids)
        self.centroids = np.asarray(r_cm)

    def compute_higher_moments(self, energy_deposited, centroid, label_idx):
        """

        Parameters
        ----------
        energy_deposited : float
            Total energy deposited by a single cosmic ray

        centroid : tuple
            (x,y) position of the centroid of the cosmic ray

        label_idx : int
            Integer label of a given cosmic ray

        Returns
        -------

        """
        I_rr = [0, 0]  # I_xx, I_yy
        I_xy = 0
        label_coords = np.where(self.label == label_idx)
        label_coords = list(zip(label_coords[0], label_coords[1]))

        for r_i in label_coords:
            I_rr += (1 / energy_deposited) * self.sci[r_i[0]][r_i[1]] \
                    * (np.asarray(r_i) - np.asarray(centroid)) ** 2

            I_xy += (1 / energy_deposited) * self.sci[r_i[0]][r_i[1]] * \
                    (r_i[0] - centroid[0]) * (r_i[1] - centroid[1])

        return I_rr, I_xy, label_coords

    def compute_shape(self, I_rr, I_xy):
        """

        Parameters
        ----------
        I_rr : tuple
            Second moment of energy distribution (I_xx, I_yy)

        I_xy : int
            Cross moment of energy distribution

        Returns
        -------

        """
        shape = np.sqrt(
            ((I_rr[0] - I_rr[1]) ** 2 + 4 * I_xy ** 2) / (I_rr.sum()) ** 2
        )
        return shape

    def compute_size(self, I_rr, label_coords):
        """ Compute the size of the cosmic ray

        This will compute the sigma-size based of the cosmic energy distribution
        and the size in pixels based of the labeled array.

        Parameters
        ----------
        I_rr

        Returns
        -------

        """
        size_sigmas = np.sqrt(I_rr.sum() / 2)
        size_pixels = len(label_coords)
        return size_sigmas, size_pixels

    def compute_cr_statistics(self):
        """ Compute the cosmic ray statistics

        """
        msg = ('Computing statistics\n '
               'fname: {}\n '
               'number of cosmic rays: {}'.format(self.fname,
                                                  len(self.label_ids)))
        LOG.info(msg)

        try:
            self.cr_incident_rate = float(len(self.label_ids)) \
                               / self.integration_time
        except ZeroDivisionError as e:
            msg = ('{}\n {} has an undefined integration time.\n '
                   'Setting cosmic ray rate to NaN'.format(e, self.fname))
            LOG.error(msg)
            self.cr_incident_rate = np.nan

        # Compute the centroids
        self.compute_first_moment()

        # Compute the total energy deposited
        self.compute_total_cr_deposition()

        # Create a generator to loop over
        loop_gen = zip(self.label_ids,
                       self.centroids,
                       self.energy_deposition)

        for label_idx, centroid, energy in loop_gen:
            # Compute the second moments of the energy distribution
            I_rr, I_xy, label_coords = self.compute_higher_moments(
                energy_deposited=energy, centroid=centroid, label_idx=label_idx
            )

            # Compute the width of the distribution of energy and size in pixels
            size_sigmas, size_pixels = self.compute_size(I_rr, label_coords)

            # Compute the symmetry of the distribution
            shapes = self.compute_shape(I_rr, I_xy)

            self.cr_affected_pixels.append(label_coords)
            self.size_in_sigmas.append(size_sigmas)
            self.size_in_pixels.append(size_pixels)
            self.shapes.append(shapes)

        self.cr_affected_pixels = [
            a for data in self.cr_affected_pixels for a in data
        ]
