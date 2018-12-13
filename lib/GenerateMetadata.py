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
            self.spt = fname.replace('_c0m.fits', '_shm.fits')
        elif 'ima.fits' in fname:
            self.spt = fname.replace('_ima.fits', '_shm.fits')
            if not os.path.exists(self.spt):
                self.spt = fname.replace('_ima.fits','_spt.fits')
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
            self.metadata['expstart'] = Time(expstart, format='mjd')
            self.metadata['expend'] = Time(expend, format='mjd')
            self.metadata['exptime'] = integration_time

    def get_observatory_info(self):
        """
        Compute the lat/lon and altitude of HST.

        Using the expstart and expend, generate a series MJD dates that correspond
        to 1 minute intervals. Calculate the lat/lon and altitude at each
        time step.

        """

        altitude_ = []
        lat_ = []
        lon_ = []

        # Break up the exposure into one minute intervals
        time_delta = self.metadata['expend'] - self.metadata['expstart']
        num_intervals = int(time_delta.to('minute').value)

        # If the number of one minute intervals is less than 2, set the number
        # of intervals to 5 (arbitrarily chosen)
        if num_intervals < 2:
            num_intervals = 5
        # Generate MJD dates correspond to these one minute intervals
        time_intervals = np.linspace(self.metadata['expstart'].mjd,
                                     self.metadata['expend'].mjd,
                                     num_intervals,
                                     endpoint=True)
        # Using the telemetry data for the SPT file, compute HST (lon, lat, z)
        if os.path.isfile(self.spt):
            orbital_params = orbit.HSTOrbit(self.spt)
            # compute coords at beginning and end of exposure
            for t in time_intervals:
                rect_, vel_ = orbital_params.getPos(t)
                r, ra_, dec_ = rectToSph(rect_)
                altitude_.append(r - R_earth.to('km').value)
                lat = dec_
                lon = ra_ - 2 * np.pi * gmst(t)
                if lon < 0:
                    lon += 2 * np.pi
                lon /= DEGtoRAD
                lat /= DEGtoRAD
                lat_.append(lat)
                lon_.append(lon)
            self.metadata['latitude'] = np.asarray(lat_)
            self.metadata['longitude'] = np.asarray(lon_)
            self.metadata['altitude'] = np.asarray(altitude_)
            self.metadata['time_intervals'] = time_intervals
        else:
            # If the SPT file for some reason doesn't exist, save NaNs
            self.metadata['altitude'] = np.nan
            self.metadata['latitude'] = np.nan
            self.metadata['longitude'] = np.nan
            self.metadata['time_intervals'] = np.nan








