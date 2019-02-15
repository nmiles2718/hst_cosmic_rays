#!/usr/bin/env python
"""
The `pipeline.process.process` module contains two classes that are used to
process the data. The first class :py:class:`~process.process.ProcessCCD` is
used for active CCD imagers (ACS, STIS, and WFC3). Each of the
active CCD imagers has a pythonic interface to their calibration pipelines,
which includes their cosmic ray rejection routines listed below:

    * ACS

      - `acstools.acsrej <https://acstools.readthedocs.io/en/latest/acsrej.html>`_

    * STIS

      - `stistools.occreject <https://stistools.readthedocs.io/en/latest/ocrreject.html>`_

    * WFC3

      - `wfc3tools.wf3rej <https://wfc3tools.readthedocs.io/en/latest/wfc3tools/wf3rej.html>`_

The second class :py:class:`~process.process.ProcessIR` is used for both active
and inactive IR imagers. The IR imagers used on HST utitilize the method of
"up-the-ramp" sampling to identify cosmic rays which are subsequently marked in
the data quality (DQ) arrays of the IMA files. For more information, see
`Chapter 3.3.10 <http://www.stsci.edu/hst/wfc3/documents/handbooks/currentDHB/wfc3_dhb.pdf#page=71>`_ in the WFC3 DataHandbook. Since IMA files already have the cosmic
rays identified, the function of the :py:class:`~process.process.ProcessIR` is
to decompose the IMA into a series of individual FITS files corresponding to
each read.


"""

from collections import defaultdict
import glob
import itertools
import logging
import os
import time

# # non native imports
from astropy.io import fits
import dask
from numpy import array
from numpy import array_split
from numpy import concatenate
from numpy import where
import numpy.random as random

from acstools import acsrej
from stistools import ocrreject
from wfc3tools import wf3rej


logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s:%(funcName)s:%(lineno)d]'
                           ' %(message)s')
LOG = logging.getLogger()
LOG.setLevel(logging.INFO)

class ProcessCCD(object):
    """ Class for processing CCD data

    Parameters
    ----------
    instr : str
        Instrument to process

    flist : list
        List of files to process

    """
    def __init__(self, instr, flist):


        self._instr = instr
        self._flist = flist
        self._num = len(flist)

        self._mod_dir = os.path.dirname(os.path.abspath(__file__))

        self._base = os.path.join('/',
                                  *self._mod_dir.split('/')[:-2])

        self._crrejtab_dir = os.path.join(self._base,
                                          'crrejtab',
                                          self.instr.split('_')[0],
                                          '')
        self._input = {}
        self._output = defaultdict(list)
        self._i = 0
        self._msg_div = '-' * 79

    @property
    def instr(self):
        return self._instr

    @instr.getter
    def instr(self):
        """Instrument to analyze"""
        return self._instr

    @property
    def flist(self):
        return self._flist

    @flist.getter
    def flist(self):
        """List of filenames to process"""
        return self._flist

    @flist.setter
    def flist(self, value):
        self._flist = value

    @property
    def input(self):
        return self._input

    @input.getter
    def input(self):
        """Container for storing sorted inputs"""
        return self._input

    @input.setter
    def input(self, value):
        self._input = value

    @property
    def output(self):
        return self._output

    @output.getter
    def output(self):
        """Container for storing which files failed or passed processing"""
        return self._output

    @output.setter
    def output(self, value):
        self._output = value

    @property
    def num(self):
        return self._num

    @num.getter
    def num(self):
        """Number of files to process"""
        return self._num

    def check_for_artifact(self, f):
        """ Scan the DQ extension for compression artifacts

        In early ACS images when the option for compressing data was available,
        there is a possibility of Reed-Solomon decoding errors being incorrectly
        classified as cosmic rays during the cosmic ray rejection step.
        """
        dq1 = ('dq', 1)  # Chip 2
        dq2 = ('dq', 2)  # Chip 1
        dq = None
        with fits.open(f) as hdu:
            prhdr = hdu[0].header
            scihdr = hdu[1].header
            if 'exptime' in prhdr:
                expt = prhdr['exptime']
            else:
                expt = scihdr['exptime']

            try:
                ext1 = hdu.index_of(dq1)
                ext1_data = hdu[ext1].data
            except KeyError as e:
                LOG.warning('{}\n {} is missing for {}'.format(e, dq1, f))
                ext1 = None

            try:
                ext2 = hdu.index_of(dq2)
                ext2_data = hdu[ext2].data
            except KeyError as e:
                LOG.warning('{}\n {} is missing for {}'.format(e, dq2, f))
                ext2 = None

        # If second DQ ext is missing, only work with the first
        # Otherwise combine each DQ ext to make full-frame
        if not ext2:
            dq = ext1_data
        else:
            dq = concatenate([ext1_data, ext2_data], axis=0)
        artifacts = where(dq == 2)[0]

        if artifacts.size > 0 or expt < 0.1:
            return True
        else:
            return False


    def ACS(self, input):
        """ Run ACS cosmic ray rejection

        Parameters
        ----------
        input : list
            List of files to process with `acsrej`

        Returns
        -------

        """
        if 'wfc' in self.instr.lower():
            crrejtab = os.path.join(self._crrejtab_dir,
                                    '29p1548cj_crr_WFC.fits')
        else:
            crrejtab = os.path.join(self._crrejtab_dir,
                                    'n4e12510j_crr_HRC.fits')

        output = 'tmp_crj_{}.fits'.format(self._i)

        # if the file exist increment _i by one before processing.
        while os.path.isfile(output):
            self._i+= random.randint(0, 500)
            output = 'tmp_crj_{}.fits'.format(self._i)
        try:
             acsrej.acsrej(input,
                          output=output,
                          verbose=True,
                          crrejtab=crrejtab,
                          crmask=True,
                          initgues='med',
                          skysub='mode')
             if not os.path.isfile(output):
                 raise (FileNotFoundError)
        except Exception as e:
            msg = ('{}\n Failed to '
                   'process input list {}\n {}'.format(e,
                                                       '\n'.join(input),
                                                       self._msg_div))
            LOG.error(msg)
            self.output['failed'].append(input)
        else:
            self.output['passed'].append(input)

    def WFC3(self, input):
        """ Run WFC3 cosmic ray rejection

        Parameters
        ----------
        input : list
            list of files to process with `wf3rej`

        Returns
        -------

        """
        output = 'tmp_crj_{}.fits'.format(self._i)
        # if the file exist increment _i by one before processing.

        while os.path.isfile(output):
            self._i += 1
            output = 'tmp_crj_{}.fits'.format(self._i)

        crrejtab = os.path.join(self._crrejtab_dir,
                                'n9i1435li_crr_UVIS.fits')
        for f in input:
            with fits.open(f,mode='update') as hdu:
                hdu[0].header['CCDTAB'] = os.path.join(self._crrejtab_dir,
                                                       't291659mi_ccd.fits')
        try:
            wf3rej(input,
                   output='tmp_crj_{}.fits'.format(self._i),
                   verbose=True,
                   crrejtab=crrejtab,
                   crmask=True,
                   initgues='med',
                   skysub='mode')

            if not os.path.isfile(output):
                raise(FileNotFoundError)

        except Exception as e:
            msg = ('{}\n Failed to '
                   'process input list {}\n {}'.format(e,
                                                       '\n'.join(input),
                                                       self._msg_div))

            LOG.error(msg)
            self.output['failed'].append(input)
        else:
            self.output['passed'].append(input)

    def STIS(self, input):
        """ Run STIS cosmic ray rejection

        Parameters
        ----------
        input : flist
            List of files to process with `ocrreject`

        Returns
        -------

        """
        if len(input) < 2:
            self.output['failed'].append(input)
        else:
            output = 'tmp_crj_{}.fits'.format(self._i)

            # if the file exist increment _i by one before processing.
            while os.path.isfile(output):
                self._i += 1
                output = 'tmp_crj_{}.fits'.format(self._i)

            crrejtab = os.path.join(self._crrejtab_dir,
                                    'j3m1403io_crr.fits')
            try:
                ocrreject.ocrreject(' '.join(input),
                                    output=output,
                                    crrejtab=crrejtab,
                                    verbose=True,
                                    crmask='yes',
                                    initgues='med',
                                    skysub='mode'
                                    )
                if not os.path.isfile(output):
                    raise(FileNotFoundError)
            except Exception as e:
                msg = ('{}\n Failed to '
                       'process input list {}\n {}'.format(e,
                                                           '\n'.join(input),
                                                           self._msg_div))
                LOG.error(msg)
                self.output['failed'].append(input)
            else:
                self.output['passed'].append(input)

    def sort(self):
        """ Sort the input files by exposure time and aperture

        The original input list could potentially contain a mix of subarray
        formats and exposure times. Attempting to combine images of with
        different exposure times or subarray formats will raise an error in
        the calibration pipelines.

        To combat this, all of the files are sorted
        by their exposure time and `CCDAMP` header value. The `CCDAMP` keyword
        describes the CCD amplifier readout configuration which can be used to
        match images taken with similar formats for CR rejection.

        Returns
        -------

        """

        found_exptimes = []
        found_formats = []
        files_to_process = self.flist

        # Check for compression artifacts and remove them from the flist
        for f in files_to_process:
            has_artifact = self.check_for_artifact(f)
            if has_artifact:
                self.ouput['failed'].append(f)
                LOG.info('Removing {} from analysis'.format(f))
                self.flist.remove(f)

        # Make a list of all the EXPTIME and CCDAMP values
        for f in self.flist:
            with fits.open(f) as hdu:
                prhdr = hdu[0].header
                scihdr = hdu[1].header

            if 'exptime' in prhdr:
                found_exptimes.append(prhdr['exptime'])

            elif 'exptime' in scihdr:
                found_exptimes.append(scihdr['exptime'])

            found_formats.append(str(prhdr['CCDAMP']))

        # Find the unique values
        unique_sizes = set(found_formats)
        unique_exp = set(found_exptimes)

        # Loop through possible combinations of CCDAMP and EXPTIME
        for ap in unique_sizes:
            for t in unique_exp:
                idx = where((array(found_exptimes) == t) &
                                 (array(found_formats) == ap))[0]
                if not idx.any():
                   pass
                else:
                    msg = ('Found {} images with '
                           'size={} and t={} \n {}'.format(len(idx),
                                                           ap,
                                                           t,
                                                           self._msg_div))
                    LOG.info(msg)

                    # Generate a list of common inputs for CR rejection.
                    self.input['{}_{}'.format(ap, t)] = array(self.flist)[idx]

        # Now we check to make sure each list of files is less than the limit
        for key, val in self.input.items():
            if len(val) > 40:
                LOG.info('{} exceeds input limit, ' \
                      'splitting to smaller groups'.format(key))
                split = array_split(val, 4)
                # Convert tuple to list for checking later
                self.input[key] = tuple(split)

    def format_inputs(self):
        """ Convenience method for preparing input lists for different pipelines

        Returns
        -------
        data : list
            A list of lists where each sub-list is a set of files to CR reject
        """
        data = []
        for key in self.input.keys():
            if isinstance(self.input[key], tuple):
                for val in self.input[key]:
                    data.append(list(val))
            else:
                data.append(list(self.input[key]))
        return data

    def cr_reject(self):
        """ Run cosmic ray rejection in a parallelized manner.

        Returns
        -------

        """
        start_time = time.time()
        msg = ('\n Rejecting cosmic rays... \n{}'.format(self._msg_div,
                                                           self._msg_div))
        LOG.info(msg)
        if 'acs' in self.instr.lower():
            # Parallelized CR rejection
            data = self.format_inputs()
            a = [dask.delayed(self.ACS)(d) for d in data]
            dask.compute(*a, scheduler='processes')

        elif 'wfc3' in self.instr.lower():
            data = self.format_inputs()
            a = [dask.delayed(self.WFC3)(d) for d in data]
            dask.compute(*a, scheduler='processes')

        elif 'stis' in self.instr.lower():
            data = self.format_inputs()
            a = [dask.delayed(self.STIS)(d) for d in data]
            dask.compute(*a, scheduler='processes')
        if 'failed' in self.output.keys():
            self.output['failed'] = list(itertools.chain.from_iterable(
                self.output['failed']))
        LOG.info('Done!')
        end_time = time.time()
        LOG.info('Duration: {}'.format((end_time - start_time)/60))


class ProcessIR(object):
    """Class for processing IR data

    Parameters
    ----------
    flist : list
        list of filenames to process

    """
    def __init__(self, flist):

        self._flist = flist

    @property
    def flist(self):
        return self._flist

    @flist.getter
    def flist(self):
        """list of filenames to process"""
        return self._flist

    def make_exts(self, fname):
        """ Generate a list of tuples corresponding to each read.

        Parameters
        ----------
        fname : str
            Filename of image to process

        Returns
        -------

        """
        with fits.open(fname) as hdu:
           num_samples = hdu[0].header['nsamp']
        LOG.info('The number of non-destructive reads '
                 'is: {}'.format(num_samples))

        sci_exts = [('sci', n) for n in range(1, num_samples + 1)]
        err_exts = [('err', n) for n in range(1, num_samples + 1)]
        dq_exts = [('dq', n) for n in range(1, num_samples + 1)]
        samp_exts = [('samp', n) for n in range(1, num_samples + 1)]
        time_exts = [('time', n) for n in range(1, num_samples + 1)]

        return list(zip(sci_exts, err_exts, dq_exts, samp_exts, time_exts))

    def edit_hdr(self, hdr):
        """Convenience function for updating the header of each read

        When we create the new FITS files for each individual read, we must
        update the EXTVER keyword to be 1. This is because there is only one
        version of each extension type per read.

        Parameters
        ----------
        hdr : astropy.io.fits.Header
            FITS Header

        Returns
        -------
        astropy.io.fits.Header
            Updated version of the input header

        """
        hdr['extver'] = 1
        return hdr

    def write_out(self, fname):
        """Write out each read in the IMA file into an individual FITS file

        Each read will be saved to a file named read_N.fits, where N corresponds
        to the read number in they order they occured. This means the 4th read
        will be saved to the file read_4.fits. Note, this is **different**
        than the order they are stored in the IMA file.

        Parameters
        ----------
        fname : str
            filename of IMA

        Returns
        -------

        """
        exts = self.make_exts(fname)
        dirname = os.path.dirname(fname)
        hdu = fits.open(fname)
        for i, (sci, err, dq, samp, time) in enumerate(exts):
            f_out = dirname + '/read_{}.fits'.format(i)
            sci_ext = hdu[hdu.index_of(sci)]
            err_ext = hdu[hdu.index_of(err)]
            dq_ext = hdu[hdu.index_of(dq)]
            samp_ext = hdu[hdu.index_of(samp)]
            time_ext = hdu[hdu.index_of(time)]
            hdu_list = fits.HDUList()
            hdu_list.append(fits.PrimaryHDU(header=hdu[0].header))

            hdu_list.append(
                fits.ImageHDU(data=sci_ext.data,
                              header=self.edit_hdr(sci_ext.header))
            )

            hdu_list.append(
                fits.ImageHDU(data=err_ext.data,
                              header=self.edit_hdr(err_ext.header))
            )

            hdu_list.append(
                fits.ImageHDU(data=dq_ext.data,
                              header=self.edit_hdr(dq_ext.header))
            )

            hdu_list.append(
                fits.ImageHDU(data=samp_ext.data,
                              header=self.edit_hdr(samp_ext.header))
            )

            hdu_list.append(
                fits.ImageHDU(data=time_ext.data,
                              header=self.edit_hdr(time_ext.header))
            )

            hdu_list.writeto(f_out, overwrite=True)
        hdu.close()

    def decompose(self):
        """ Run the decomposition of the IMA files into there individual reads
        """
        results = [dask.delayed(self.write_out)(fname) for fname in self.flist]
        dask.compute(*results)

#
#
#
# if __name__ == "__main__":
#     # For debugging purposes
#     flist = glob.glob('./../crrejtab/STIS/mastDownload/HST/*/*flt.fits')
#     p = ProcessCCD('stis_ccd',flist)
#     p.sort()
#     p.cr_reject()
