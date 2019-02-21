"""
For each cosmic ray this module will compute the following statistics:

  * Size in pixels
  * Size in sigmas (i.e. width of the energy distribution deposited by the cosmic ray)
  * Shape (a measure of symmetry of the energy distribution deposited cosmic ray)
  * Incidence rate (i.e. number of cosmic rays per second)
  * Total energy deposited by each cosmic ray
  * A list of all the pixels affected by cosmic rays

"""

import logging

import numpy as np
from scipy import ndimage
from scipy.sparse import csr_matrix


logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s')

LOG = logging.getLogger()

LOG.setLevel(logging.INFO)


class Stats(object):
    """ Class for computing statistics about each cosmic ray

    Parameters
    ----------
    cr_label : :py:class:`~label.labeler.CosmicRayLabel`
        The resulting :py:class:`~label.labeler.CosmicRayLabel`
        object after executing the
        :py:meth:`~label.labeler.CosmicRayLabel.run_ccd_label` method.

    """

    def __init__(self, cr_label, integration_time):

        self._fname = cr_label.fname
        self._label = cr_label.label
        self._sci = cr_label.sci
        self._integration_time = integration_time

        self._incident_cr_rate = None
        self._max_x = self._label.shape[1]
        self._max_y = self._label.shape[0]
        self._label_ids = np.unique(self._label)[1:]
        self._cr_positions = np.asarray(ndimage.find_objects(self._label))

        self._centroids = None
        self._energy_deposited = None
        self._shapes = []
        self._size_in_sigmas = []
        self._size_in_pixels = []
        self._cr_affected_pixels = []


    @property
    def incident_cr_rate(self):
        return self._incident_cr_rate

    @incident_cr_rate.getter
    def incident_cr_rate(self):
        """Compute cosmic ray incidence rate"""
        return self._incident_cr_rate

    @incident_cr_rate.setter
    def incident_cr_rate(self, value):
        self._incident_cr_rate = value

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
    def energy_deposited(self):
        return self._energy_deposited

    @energy_deposited.getter
    def energy_deposited(self):
        """Total energy deposited by cosmic rays"""
        return self._energy_deposited

    @energy_deposited.setter
    def energy_deposited(self, value):
        self._energy_deposited = value

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

    def compute_cr_energy_deposited(self):
        """Compute the total number of electrons deposited at each label.

        This is simply a sum of all pixel values identified as belonging to a
        given cosmic ray.

        * :math:`I_0 = \sum_{i} p_i`

        Returns
        -------
        cr_sum : numpy.ndarray
            Sum of all pixels at each label in :py:attr:`label_ids`

        """

        cr_sum = ndimage.sum(self.sci,
                             labels=self.label,
                             index=self.label_ids)
        self.energy_deposited = np.asarray(cr_sum)

    def compute_first_moment(self, sci=None):
        """Compute the first moment of energy deposited by a given cosmic ray.

        This corresponds to the flux-weighted centroid of the cosmic ray denoted
        by :math:`(I_x, I_y).`

        * :math:`I_x = \\frac{1}{I_0} \\sum_{i}p_i * x_i`
        * :math:`I_y = \\frac{1}{I_0} \\sum_{i}p_i * y_i`

        Parameters
        ----------
        sci : numpy.ndarray [:py:attr:`~label.base.Label.sci`]
            The :py:attr:`~label.base.Label.sci` attribute of the
            :py:class:`~label.labeler.CosmicRayLabel` object.

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

    def compute_higher_moments(self, energy_deposited, centroid, label_coords):
        """ Compute all second moments of the distribution

        * :math:`I_{xx} = \\frac{1}{I_0} \\sum_{i}p_i(x_i - I_x)^2`

        * :math:`I_{yy} = \\frac{1}{I_0} \\sum_{i}p_i(y_i - I_y)^2`

        * :math:`I_{xy} = \\frac{1}{I_0} \\sum_{i}p_i(x_i - I_x)*(y_i - I_y)`

        Parameters
        ----------
        energy_deposited : float
            Total energy deposited by a single cosmic ray

        centroid : tuple
            (x,y) position of the centroid of the cosmic ray

        label_coords : tuple
            tuple of cosmic ray x coordinates and y coordinates

        Returns
        -------
        I_rr : tuple
            The second moments of the energy distribution in x and y, (:math:`I_{xx}`, :math:`I_{yy}`)

        I_xy : int
            The cross moment of the energy distribution, :math:`I_{xy}`


        """
        I_rr = [0, 0]  # I_xx, I_yy
        I_xy = 0
        # coords = list(zip(label_coords[1], label_coords[0]))
        for r_i in label_coords:
            I_rr += (1 / energy_deposited) * self.sci[r_i[0]][r_i[1]] \
                    * (np.asarray(r_i) - np.asarray(centroid)) ** 2

            I_xy += (1 / energy_deposited) * self.sci[r_i[0]][r_i[1]] * \
                    (r_i[0] - centroid[0]) * (r_i[1] - centroid[1])

        return I_rr, I_xy

    def _compute_M(self):
        """"""
        cols = np.arange(self.label.size)
        return csr_matrix((cols, (self.label.ravel(), cols)),
                          shape=(self.label.max() + 1, self.label.size))

    def _get_indices_sparse(self):
        """"""
        M = self._compute_M(self.label)
        return [np.unravel_index(row.data, self.label.shape) for row in M]

    def compute_shape(self, I_rr, I_xy):
        """ Compute the "shape" of the distribution of the energy deposited

        This is effectively a measure of symmetry of the distribution that is
        defined as follows:

            :math:`shape = \\sqrt{\\frac{(I_{xx} - I_{yy})^2 + 4I^2_{xy}}{(I_{xx} + I_{yy})^2}}`

        Parameters
        ----------
        I_rr : tuple
            Second moment of energy distribution :math:`(I_{xx}, I_{yy})`

        I_xy : int
            Cross moment of energy distribution

        Returns
        -------
        shape : int
            The computed shape

        """
        shape = np.sqrt(
            ((I_rr[0] - I_rr[1]) ** 2 + 4 * I_xy ** 2) / (I_rr.sum()) ** 2
        )
        return shape

    def compute_size(self, I_rr, label_coords):
        """ Compute the size of the cosmic ray in two ways

        #. Compute the width or size of the cosmic energy distribution
           using the previously computed second moments.

            :math:`size = \\sqrt{\\frac{I_{xx} + I_{yy}}{2}}`

        #. Compute the size of the cosmic ray in terms of the number of
           pixels that it affects.

        Parameters
        ----------
        I_rr : tuple
            Second moment of energy distribution :math:`(I_{xx}, I_{yy})`

        Returns
        -------
        size_sigmas : float
            Width of the cosmic ray energy distribution

        size_pixels : float
            Total number of pixels affected by the given cosmic ray

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
            self.incident_cr_rate = float(len(self.label_ids)) \
                               / self.integration_time
        except ZeroDivisionError as e:
            msg = ('{}\n {} has an undefined integration time.\n '
                   'Setting cosmic ray rate to NaN'.format(e, self.fname))
            LOG.error(msg)
            self.incident_cr_rate = np.nan

        # Compute the centroids
        self.compute_first_moment()

        # Compute the total energy deposited
        self.compute_cr_energy_deposited()

        # Get the cosmic ray coords
        sparse_m = self._compute_M()
        cr_coords = [
            np.unravel_index(row.data, self.label.shape) for row in sparse_m
        ]

        # Create a generator to loop over
        loop_gen = zip(self.centroids,
                       self.energy_deposited,
                       cr_coords[1:])


        for centroid, energy, coords in loop_gen:
            label_coords = list(zip(coords[0], coords[1]))
            # Compute the second moments of the energy distribution
            I_rr, I_xy = self.compute_higher_moments(
                energy_deposited=energy,
                centroid=centroid,
                label_coords=label_coords
            )

            # Compute the width of the distribution of energy and size in pixels
            size_sigmas, size_pixels = self.compute_size(I_rr, label_coords)

            # Compute the symmetry of the distribution
            shapes = self.compute_shape(I_rr, I_xy)

            self.cr_affected_pixels.append(label_coords)
            self.size_in_sigmas.append(size_sigmas)
            self.size_in_pixels.append(size_pixels)
            self.shapes.append(shapes)

        # Collapse the list of lists into a single flattened list
        self.cr_affected_pixels = [
            a for data in self.cr_affected_pixels for a in data
        ]


def debug():
    fname = '/Users/nmiles/hst_cosmic_rays/data/STIS/CCD/mastDownload/HST/o3st05eaq/o3st05eaq_flt.fits'