#!/usr/bin/env python

import logging
import os
import warnings

from astropy.time import Time
from astroquery.mast import Observations
import yaml



__taskname__ = "download"
__author__ = "Nathan Miles"
__version__ = "1.0"
__vdate__ = "22-Jan-2019"


logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger('Downloader')
LOG.setLevel(logging.INFO)



class Downloader(object):

    def __init__(self, instr, instr_cfg=None):

        self._mod_dir = os.path.dirname(os.path.abspath(__file__))

        self._base = os.path.join('/',
                                  *self._mod_dir.split('/')[:-2])

        if instr_cfg is None:
            cfg_file = os.path.join(self._base,
                                    'CONFIG',
                                    'pipeline_config.yaml')

            with open(cfg_file, 'r') as fobj:
                cfg = yaml.load(fobj)

            self._instr_cfg = cfg[instr]
        else:
            self._instr_cfg = instr_cfg


        self._dates = None

        self._download_dir = os.path.join(
            self._base,
            *self._instr_cfg['astroquery']['download_dir'].split('/')
        )

        self._inactive_range = {
            'ACS': [
                Time('2007-01-27', format='iso'),
                Time('2009-05-01', format='iso')
            ],
            'STIS': [
                Time('2004-08-03', format='iso'),
                Time('2009-05-01', format='iso')
            ]
        }
        self._filtered_table = None
        self._instr = instr.replace('_', '/') # put into format for astroquery
        self._msg_div = '-' * 79
        self._obstype = 'calibration'
        self._product_type = ['image', 'spectrum'] # SPECTRUM is for STIS
        self._products = {}
        self._project = 'HST'
        self._start_date = None
        self._stop_date = None
        self._SubGroupDescription = self._instr_cfg['astroquery']['SubGroup' \
                                                            'Description']
        self._target_name = 'DARK*'
        self._t_exptime = [0.5, 10000] # Exposure times to include


    @property
    def instr_cfg(self):
        return self._instr_cfg

    @instr_cfg.getter
    def instr_cfg(self):
        """Configuration object

        Corresponds to the configuration object stored in the
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
    def download_dir(self):
        return self._download_dir

    @download_dir.getter
    def download_dir(self):
        """Download directory for the instrument being analyzed

        Corresponds to the value stored in the given instruments info stored in
        the :py:attr:`~pipeline_updated.CosmicRayPipeline.cfg` attribute

        """
        return self._download_dir

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
    def obstype(self):
        return self._obstype

    @obstype.getter
    def obstype(self):
        """`obstype` kwarg for MAST query (str)"""
        return self._obstype

    @property
    def products(self):
        return self._products

    @products.getter
    def products(self):
        """Filtered version of data products returned by MAST query"""
        return self._products

    @products.setter
    def products(self, value):
        self._products = value

    @property
    def product_type(self):
        return self._product_type

    @product_type.getter
    def product_type(self):
        """Product types to download """
        return self._product_type

    @property
    def project(self):
        return self._project

    @project.getter
    def project(self):
        """`project` kwarg for MAST query (str)"""
        return self._project

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

    @property
    def SubGroupDescription(self):
        return self._SubGroupDescription

    @SubGroupDescription.getter
    def SubGroupDescription(self):
        """`SubGroupDescription` kwarg for filtering MAST products (list)"""
        return self._SubGroupDescription

    @property
    def t_exptime(self):
        return self._t_exptime

    @t_exptime.getter
    def t_exptime(self):
        """Range of exposure times to download"""
        return self._t_exptime

    @property
    def target_name(self):
        return self._target_name

    @target_name.getter
    def target_name(self):
        """Name of target to download"""
        return self._target_name

    def query(self, range, aws=False):
        """ Submit a query to MAST for observations in the date range

        Parameters
        ----------
        range : tuple
            Tuple of `astropy.time.Time` objects correspond to the beginning
            and end of a one month interval
        aws : bool
            If True, query returns references to data hosted in S3.

        """
        LOG.info('Submitting query to MAST')
        if aws:
            Observations.enable_s3_hst_dataset()
        start, stop = range
        # there shouldn't be any data taken after the most recent file
        query_params = {
            'project': self.project,
            'dataproduct_type': self.product_type,
            'intentType': self.obstype,
            'target_name': self.target_name,
            'instrument_name': self.instr,
            't_min': [start.mjd, stop.mjd],
            't_exptime': self.t_exptime
        }
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            try:
                obsTable = Observations.query_criteria(**query_params)
            except Exception as e:
                msg = ('{}\n Date range [{}, {}]\n {}'.format(e,
                                                              start.iso,
                                                              stop.iso,
                                                              self._msg_div))
                LOG.error(msg)
            else:
                LOG.info('Filtering observations...')
                products = Observations.get_product_list(obsTable)
                filter_params = {
                    'mrp_only': False,
                    'productSubGroupDescription':self.SubGroupDescription
                }

                filt_products = Observations.filter_products(products,
                                                             **filter_params)

                key = start.datetime.date().isoformat() # 'YYYY-MM-DD'
                self.products[key] = filt_products

    def download(self, key):
        """Download the data

        Only download the observations contained in the interval specified by
        the `key`. The `key` argument must correspond to one of the keys in the
        :py:attr:`~download.Downloader.products` attribute. If it is not,
        then a KeyError will be raised and the download will be skipped.


        Parameters
        ----------
        key : str
            Date in ISO format (YYYY-MM-DD) of a given intervals start time

        Returns
        -------
        None
            Downloaded data will be stored in directory specified by the
            :py:attr:`~download.Downloader.download_dir` attribute
        """
        msg = ('Downloading data...\n '
               'Download Directory: {}\n {}'.format(self.download_dir,
                                                    self._msg_div))
        LOG.info(msg)
        download_params = {
            'download_dir': self.download_dir,
            'mrp_only': False,
            'dataproduct_type': self.product_type,
            'productSubGroupDescription': self.SubGroupDescription
        }
        try:
            download_list = self.products[key]['obsID'].tolist()
        except KeyError as e:
            LOG.error('{}\n{}'.format(e, self._msg_div))
        else:
            Observations.download_products(download_list, **download_params)

# def main():
#     import yaml
#     cfg_file = '/Users/nmiles/hst_cosmic_rays/CONFIG/pipeline_config.yaml'
#     with open(cfg_file) as fobj:
#         cfg = yaml.load(fobj)
#     instr = 'WFC3_UVIS'
#     d = Downloader(instr=instr, instr_cfg=cfg[instr])
#     d.initialize_dates()
#     d.get_date_ranges()
#     d.query(range=d.dates[0], aws=False)
#     d.download(d.dates[0][0].datetime.date().isoformat())
#
# if __name__ == "__main__":
#     main()



