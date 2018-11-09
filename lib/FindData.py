#!/usr/bin/env python

from datetime import date
import warnings

from astropy.time import Time
from astroquery.mast import Observations
from numpy import array
from pandas import date_range



_CFG = {'ACS':'2002-03-01',
        'WFC3':'2009-06-01',
        'STIS':'1997-02-01',
        'WFPC2':['1994-01-01','2009-05-30']}

class FindData(object):

    def __init__(self, instr):
        # Query parameters
        if instr == 'WFPC2':
            self._instr = instr
            self._SubGroupDescription = ['C0M', 'SHM']
            self._start = Time(_CFG[self._instr][0], format='iso')
            self._stop = Time(_CFG[self._instr][1], format='iso')
        elif 'IR' in instr:
            self._instr = instr.replace('_', '/')
            self._SubGroupDescription = ['IMA', 'SPT']
            self._start = Time(_CFG[self._instr.split('/')[0]], format='iso')
            self._stop = Time(date.today().isoformat(), format='iso')
        else:
            self._instr = instr.replace('_','/') # Format the instrument name
            self._start = Time(_CFG[self._instr.split('/')[0]], format='iso')
            self._stop = Time(date.today().isoformat(), format='iso')
            self._SubGroupDescription = ['FLT', 'SPT']
        self._collection = 'HST'
        self._product_type = ['image','spectrum'] # DOES NOT WORK FOR STIS, MUST BE SPECTRUM
        self._obstype = 'cal'
        self._target_name = 'DARK'

        # Uncomment for STIS debugging
        # self._stop = Time('2000-01-01', format='iso')

        # Storing the results
        self._products = {}
        self._filtered_table = None
        self.dates = None
        self.t_exptime = [200, 10000]

    def get_date_ranges(self):
        """
        We search for all files and to find the oldest and newest obs dates.
        We use those dates to generate a list of date ranges that we will use
        to query MAST for 1 month chunks of darks.

        """


        # very roundabout way of generating a list of MJD dates separated by a month
        pd_range = date_range(start=self._start.iso,
                                   end=self._stop.iso,
                                   freq='1MS')
        dates = [Time(date.date().isoformat(), format='iso')
                 for date in pd_range]
        date_ranges_even = list(zip(dates[::2], dates[1::2]))
        date_ranges_odd = list(zip(dates[1::2], dates[2:-2:2]))


        date_ranges = sorted(date_ranges_even + date_ranges_odd,
                              key=lambda x: x[0])
        if 'ACS' in self._instr:
            print(len(date_ranges))
            failure = Time('2007-01-27', format='iso')
            sm4 = Time('2009-05-01', format='iso')
            keep = []
            for range in date_ranges:
                if range[0] >= failure and range[1] <= sm4:
                    continue
                keep.append(range)
            self.dates = array(keep)
        else:
            self.dates = array(date_ranges)

    def query(self, range, aws=False):
        """
        Using the date list generated above, query for 1 month intervals of
        darks and save them in a dictionary where the key corresponds to the
        interval start date.
        Returns
        -------

        """

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
                    t_min = [start.mjd, stop.mjd],
                    t_exptime = self.t_exptime
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
        try:
            download_list = self._products[key]['obsID'].tolist()
        except KeyError as e:
            print(e)
        else:    
            Observations.download_products(download_list,
                                           download_dir=data_dir,
                                           mrp_only=False,
                                           dataproduct_type = self._product_type,
                                           productSubGroupDescription=self._SubGroupDescription)


