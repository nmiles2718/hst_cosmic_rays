#!/usr/bin/env python

from astropy.io import fits
from astropy.time import Time, TimeDelta
from astropy.wcs import WCS
from astropy.constants import R_earth
from calcos import orbit
from calcos.timeline import gmst, DEGtoRAD, rectToSph
import numpy as np
import os


class GenerateMetadata(object):
    def __init__(self, fname):
        self.flt = fname # file name will always be the FLT
        if 'c0m.fits' in fname:
            self.spt = fname.replace('_c0m.fits','_shm.fits')
        else:
            self.spt = fname.replace('_flt.fits','_spt.fits')
        self.date = None

        self.metadata = {}

    def get_wcs_info(self):
        with fits.open(self.flt) as hdu:
            scihdr = hdu[1].header
            try:
                wcs_obj = WCS(scihdr)
            except (MemoryError, ValueError, KeyError) as e:
                print(e)
            else:
                wcs_header = wcs_obj.to_header()
                for key in wcs_header.keys():
                    self.metadata[key] = wcs_header[key]

    def get_image_data(self):
        integration_time = 0
        with fits.open(self.flt) as hdu:
            prhdr = hdu[0].header
            scihdr = hdu[1].header
            wcs_obj = WCS(scihdr)
            if 'date-obs' in prhdr and 'time-obs' in prhdr:
                date_obs = prhdr['date-obs']
                time_obs = prhdr['time-obs']
            else:
                date_obs = scihdr['date-obs']
                time_obs = scihdr['time-obs']
            if 'expstart' in prhdr and 'expend' in prhdr:
                expstart = prhdr['expstart']
                expend = prhdr['expend']
            else:
                expstart = scihdr['expstart']
                expend = scihdr['expend']
            if 'exptime' in prhdr:
                integration_time += prhdr['exptime'] # add ~113/2 for readout
                if 'flashdur' in prhdr:
                    integration_time += prhdr['flashdur']
            elif 'exptime' in scihdr:
                integration_time += scihdr['exptime']
                if 'flashdur' in scihdr:
                    integration_time += scihdr['flashdur']

            self.date = Time('{} {}'.format(date_obs,
                                            time_obs), format='iso')
            self.metadata['date'] = self.date.iso
            self.metadata['expstart'] = Time(expstart, format='mjd').iso
            self.metadata['expend'] = Time(expend, format='mjd').iso
            self.metadata['exptime'] = integration_time

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
        else:
            altitude = np.nan
            lat_ = np.nan
            lon_ = np.nan
        self.metadata['altitude'] = altitude
        self.metadata['latitude'] = lat_
        self.metadata['longitude'] = lon_





