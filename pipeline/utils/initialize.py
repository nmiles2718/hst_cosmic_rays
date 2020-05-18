#!/usr/bin/env python
"""
This module contains a class :py:class:`~initialize.initialize.Initializer`
that contains a series of methods for performing basic initialization of the
pipeline. In short it will perform the following:

    * Generate a series of empty HDF5 files for each statistic recorded.
    * Generate a list of the previously processed date ranges.
    * Generate a list of date intervals spanning one month periods to be used
      in the downloading process.

"""

from collections import defaultdict
import logging
import os
import warnings
import yaml

from astropy.time import Time
import h5py
from numpy import array
from pandas import date_range


logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)


class Initializer(object):
    """Class for initializing common things used by the entire pipeline

    Parameters
    ----------
    instr : str
        Instrument to process

    cfg : dict
        Pipeline configuration object

    """
    def __init__(self, instr, cfg=None, instr_cfg=None):

        self._instr = instr
        self._mod_dir = os.path.dirname(os.path.abspath(__file__))
        self._base = os.path.join('/',
                                  *self._mod_dir.split('/')[:-2])
        if cfg is None:
            cfg_file = os.path.join(self._base,
                                    'CONFIG',
                                    'pipeline_config.yaml')

            with open(cfg_file, 'r') as fobj:
                cfg = yaml.load(fobj)
            self._cfg = cfg
        else:
            self._cfg = cfg

        if instr_cfg is None:
            self._instr_cfg = cfg[instr]
        else:
            self._instr_cfg = instr_cfg

        self._dates = None
        self._inactive_range = {
            'ACS_WFC': [
                Time('2007-01-27', format='iso'),
                Time('2009-05-01', format='iso')
            ],
            'ACS_HRC':[
                Time('2007-01-27', format='iso'),
                Time.now()
            ],
            'STIS_CCD': [
                Time('2004-08-03', format='iso'),
                Time('2009-05-01', format='iso')
            ]
        }
        self._previously_analyzed = None
        self._start_date = None
        self._stop_date = None

    @property
    def base(self):
        return self._base

    @base.getter
    def base(self):
        """Base path of the pipleine repository `~/hst_cosmic_rays/`"""
        return self._base

    @property
    def cfg(self):
        return self._cfg

    @cfg.getter
    def cfg(self):
        """Configuration object

        Corresponds to the configuration object stored in the
         :py:attr:`~pipeline_updated.CosmicRayPipeline.cfg` attribute

         """
        return self._cfg

    @property
    def instr_cfg(self):
        return self._instr_cfg

    @instr_cfg.getter
    def instr_cfg(self):
        """Instrument specific configuration object

        Corresponds to one of the `dict` configuration object stored in the
        :py:attr:`~pipeline_updated.CosmicRayPipeline.cfg` attribute

         """
        return self._instr_cfg

    @property
    def dates(self):
        return self._dates

    @dates.getter
    def dates(self):
        """A list of one month date intervals"""
        return self._dates

    @dates.setter
    def dates(self, value):
        self._dates = value

    @property
    def inactive_range(self):
        return self._inactive_range

    @inactive_range.getter
    def inactive_range(self):
        """Periods of inactivity for each instrument if they exists"""
        return self._inactive_range

    @property
    def instr(self):
        return self._instr

    @instr.getter
    def instr(self):
        """Name of the instrument that is going to be analyzed"""
        return self._instr

    @property
    def previously_analyzed(self):
        return self._previously_analyzed

    @previously_analyzed.getter
    def previously_analyzed(self):
        """A list of previously analyzed date ranges"""
        return self._previously_analyzed

    @previously_analyzed.setter
    def previously_analyzed(self, value):
        self._previously_analyzed = value

    @property
    def start_date(self):
        return self._start_date

    @start_date.getter
    def start_date(self):
        """Earliest possible date for any observations taken by instrument"""
        return self._start_date

    @start_date.setter
    def start_date(self, value):
        self._start_date = value

    @property
    def stop_date(self):
        return self._stop_date

    @stop_date.getter
    def stop_date(self):
        """Latest possible date for any observations taken by instrument"""
        return self._stop_date

    @stop_date.setter
    def stop_date(self, value):
        self._stop_date = value

    def initialize_dates(self):
        """Determine the start and stop dates of all observations

        The ranges of dates we wish to analyze will depend on whether or not
        the instrument being analyzed is an active or legacy instrument.

            - Active instruments have no defined end date, so we set this to
              the current date using :py:meth:`astropy.time.Time.now`
              method.

            - Legacy instruments have a defined end date that corresponds to
              a date when the instrument either failed or was shutdown.

        Initializes the :py:attr:`start_date` and
        :py:attr:`stop_date` attributes
        """
        cfg_dates = self.instr_cfg['astroquery']['date_range']
        if isinstance(cfg_dates, list):
            # Inactive instrument have defined date ranges
            self.start_date = Time(cfg_dates[0], format='iso')
            self.stop_date = Time(cfg_dates[1], format='iso')
        else:
            self.start_date = Time(cfg_dates, format='iso')
            self.stop_date = Time.now()


    def get_date_ranges(self):
        """ Generate a list of tuples containing one month intervals

        For instruments that experienced failures, intervals that fall in a
        period of inactivity will be automatically removed.
        """
        self.initialize_dates()
        pd_range = date_range(start=self.start_date.iso,
                                   end=self.stop_date.iso,
                                   freq='1MS')
        dates = [Time(date.date().isoformat(), format='iso')
                 for date in pd_range]
        date_ranges_even = list(zip(dates[::2], dates[1::2]))
        date_ranges_odd = list(zip(dates[1::2], dates[2:-2:2]))


        date_ranges = sorted(date_ranges_even + date_ranges_odd,
                              key=lambda x: x[0])

        # Check if the instrument had any failures
        # instr = self.instr.split('/')[0]
        if self.instr in self.inactive_range.keys():
            start_failure = self.inactive_range[self.instr][0]
            stop_failure = self.inactive_range[self.instr][1]
            keep = []
            # Remove dates falling in the period of inactivity
            for range in date_ranges:

                if range[0] >= start_failure and range[1] <= stop_failure:
                    continue
                keep.append(range)

            # Update the list of date intervals
            self.dates = array(keep)
        else:
            self.dates = array(date_ranges)

    def initialize_HDF5(self, chunks=4):
        """ Create the required hdf5 files that we will write to

        Each statistics stored in the HDF5 files will vary in size image to
        image. This makes it impossible to assign a single structure and
        utilize a chunked-dataset approach when writing to an HDF5 file.
        Additionally, because of the large number of datasets we require many
        open/close operations on each HDF5 to prevent loss of data.
        For example, if we were to keep our HDF5 file open during the entire
        pipeline and it faile

        """
        hdf5_files = self.instr_cfg['hdf5_files']
        new_flist = defaultdict(list)
        for key in hdf5_files.keys():
            rel_path = hdf5_files[key]
            full_path = os.path.join(self.base, *rel_path.split('/'))
            if isinstance(chunks, str):
                fnew = full_path.replace('.hdf5', '_{}.hdf5'.format(chunks))
                new_flist[key].append(fnew)
                continue
            i = 0
            while i < chunks:
                fnew = full_path.replace('.hdf5', '_{}.hdf5'.format(i + 1))
                i += 1
                new_flist[key].append(fnew)

        for key in new_flist.keys():
            for f in new_flist[key]:
                if not os.path.isdir(os.path.dirname(f)):
                    os.makedirs(os.path.dirname(f), exist_ok=True)

                LOG.info(
                    'File structure: /{}'.format(self.cfg['grp_names'][key])
                )
                with h5py.File(f, 'w') as fobj:
                    grp = fobj.create_group(self.cfg['grp_names'][key])

    def get_processed_ranges(self):
        """ Get the previously processed date ranges

        This will be used to check if the date range in question has already
        been analyzed, if it has then it will be skipped and the next
        range will be tried.

        Parameters
        ----------
        instr

        Returns
        -------

        """
        fname = os.path.join(self.base,'CONFIG',
                             'processed_dates_{}.txt'.format(self.instr))
        try:
            with open(fname, 'r') as fobj:
                lines = fobj.readlines()
                dates = [line.strip('\n') for line in lines]
        except FileNotFoundError as e:
            LOG.warning("Found 0 previously processed date ranges")
            self.previously_analyzed = []
        else:
            self.previously_analyzed = dates
            LOG.info(
                'Found {} previously processed'
                ' date ranges'.format(len(self.previously_analyzed))
            )
