#!/usr/bin/env python

from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.constants import R_earth
from calcos import orbit
from calcos.timeline import gmst, DEGtoRAD, rectToSph
import numpy as np
import os


class GenerateMetadata(object):
    def __init__(self, fname):
        self.flt = fname # file name will always be the FLT
        self.spt = fname.replace('_flt.fits','_spt.fits')
        self.date = None
        self.metadata = {}

    def get_image_data(self):
        with fits.open(self.flt) as hdu:
            prhdr = hdu[0].header
            scihdr = hdu[1].header
            self.date = Time('{} {}'.format(scihdr['date-obs'],
                                            scihdr['time-obs']), format='iso')
            self.metadata['date'] = self.date.iso
            self.metadata['expstart'] = Time(scihdr['expstart'], format='mjd').iso
            self.metadata['expend'] = Time(scihdr['expend'], format='mjd').iso

    def get_observatory_info(self):
        """
        Compute the lat/lon and altitude of HST

        """
        pos_ = {'postnstx': None,
                'postnsty': None,
                'postnstz': None}
        if os.path.isfile(self.spt):
            with fits.open(self.spt) as hdu:
                hdr = hdu[0].header
                for key in pos_.keys():
                    pos_[key] = hdr[key]
            # Compute altitude in km's from center of earth
            altitude = np.sqrt(pos_['postnstx']**2 +
                               pos_['postnsty']**2 +
                               pos_['postnstz']**2)
            altitude -= R_earth.to('km').value
            orbital_params = orbit.HSTOrbit(self.spt)
            rect_, vel_ = orbital_params.getPos(self.date.mjd)
            r, ra_, dec_ = rectToSph(rect_)
            lat_ = dec_
            lon_ = ra_ - 2 * np.pi * gmst(self.date.mjd)
            if lon_ < 0:
                lon_ += 2 * np.pi
            lon_ /= DEGtoRAD
            lat_ /= DEGtoRAD
            print(lat_, lon_)
        else:
            altitude = np.nan
            lat_ = np.nan
            lon_ = np.nan
        self.metadata['altitude'] = altitude
        self.metadata['latitude'] = lat_
        self.metadata['longitude'] = lon_





