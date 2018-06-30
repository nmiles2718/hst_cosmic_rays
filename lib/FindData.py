#!/usr/bin/env python

from datetime import date
import itertools
import warnings

from astropy.time import Time
from astroquery.mast import Observations
import pandas as pd

_CFG = {'ACS':'2002-03-01',
        'WFC3':'2009-05-01',
        'STIS':'1997-02-01'}

class FindData(object):

    def __init__(self, instr):
        # Query parameters
        self._instr = instr.replace('_','/') # Format the instrument name
        self._collection = 'HST'
        self._product_type = ['image','spectrum'] # DOES NOT WORK FOR STIS, MUST BE SPECTRUM
        self._obstype = 'cal'
        self._target_name = 'DARK'
        self._start = Time(_CFG[self._instr.split('/')[0]], format='iso')
        self._stop = Time(date.today().isoformat(), format='iso')
        self._SubGroupDescription = ['FLT', 'SPT']
        # Storing the results
        self._products = {}
        self._filtered_table = None
        self.dates = None
        self.t_exptime = [800, 1200]

    def get_date_ranges(self):
        """
        We search for all files and to find the oldest and newest obs dates.
        We use those dates to generate a list of date ranges that we will use
        to query MAST for 1 month chunks of darks.

        """


        # very roundabout way of generating a list of MJD dates separated by a month
        pd_range = pd.date_range(start=self._start.iso,
                                   end=self._stop.iso,
                                   freq='1MS')
        dates = [Time(date.date().isoformat(), format='iso')
                 for date in pd_range]
        date_ranges_even = zip(dates[::2], dates[1::2])
        date_ranges_odd = zip(dates[1::2], dates[2:-2:2])
        date_ranges = itertools.chain(date_ranges_even, date_ranges_odd))
        self.dates = date_ranges

    def query(self, range, aws=False):
        """
        Using the date list generated above, query for 1 month intervals of
        darks and save them in a dictionary where the key corresponds to the
        interval start date.
        Returns
        -------

        """

        # date_ranges_odd = zip(self.dates[::2], self.dates[1::2])
        # date_ranges_even = zip(self.dates[1::2],self.dates[2:-2:2])
        # date_ranges = itertools.chain(date_ranges_even, date_ranges_odd)
        if aws:
            Observations.enable_s3_hst_dataset()
        start, stop = range
        # there shouldn't be any data taken after the most recent file
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            try:
                obsTable = Observations.query_criteria(
                    obs_collection=self._collection,
                    dataproduct_type=self._product_type,
                    obstype=self._obstype,
                    target_name=self._target_name,
                    instrument_name=self._instr,
                    t_min = [start.mjd, stop.mjd]
                )
            except Exception as e:
                print(e,'Date range [{}, {}]'.format(start.iso, stop.iso))
            else:
                products = Observations.get_product_list(obsTable)
                filtered_products = Observations.filter_products(products,
                                                                 mrp_only=False,
                                                                 productSubGroupDescription=self._SubGroupDescription
                                                                 )
                key = start.datetime.date().isoformat() # 'YYYY-MM-DD'
                self._products[key] = filtered_products

    def download(self, key):
        """
        Download the one month chunk of data corresponding to the specified
        key. This will allow us to download only one month at a time, process it,
        then run the analysis on.

        """
        data_dir = './../crrejtab/{}/'.format(self._instr.split('/')[0])
        download_list = self._products[key]['obsID'].tolist()
        Observations.download_products(download_list,
                                       download_dir=data_dir,
                                       mrp_only=False,
                                       dataproduct_type = self._product_type,
                                       productSubGroupDescription=self._SubGroupDescription)


