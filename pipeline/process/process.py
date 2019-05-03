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
import logging
import os
import urllib.request as request
import urllib.error as error
import time

# # non native imports
from astropy.io import fits
import dask
from numpy import array
from numpy import array_split
from numpy import concatenate
from numpy import where
import numpy.random as random
import yaml

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
    def __init__(self, instr, flist, instr_cfg=None):

        # Set up base path
        self._mod_dir = os.path.dirname(os.path.abspath(__file__))
        self._base = os.path.join('/',
                                  *self._mod_dir.split('/')[:-2])

        self._instr = instr
        if instr_cfg is None:
            cfg_file = os.path.join(self._base,
                                    'CONFIG',
                                    'pipeline_config.yaml')
            with open(cfg_file, 'r') as fobj:
                cfg = yaml.load(fobj)

            self._instr_cfg = cfg[instr]
        else:
            self._instr_cfg = instr_cfg
        self._flist = flist
        self._num = len(flist)




        self._crrejtab = os.path.join(self._base,
                                      *self._instr_cfg['crrejtab'].split('/'))

        self._data_dir = os.path.join(self._base,
                                          'data',
                                      self.instr.split('_')[0],
                                          '')
        self._input = {}
        self._output = defaultdict(list)
        self._i = 0
        self._msg_div = '-' * 79

    @property
    def base(self):
        return self._base

    @base.getter
    def base(self):
        return self._base
    
    @property
    def crrejtab(self):
        return self._crrejtab

    @crrejtab.getter
    def crrejtab(self):
        """Cosmic ray rejection parameter table"""
        return self._crrejtab

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

    def _download_reffile(self, ref_file):
        """Convenience method for download CCDTABs"""

        crds_url = ('https://hst-crds.stsci.edu/'
                    'unchecked_get/references/hst/{}'.format(ref_file))
        fout = os.path.join(self._base,
                            'data',
                            self.instr.split('_')[0],
                            ref_file)

        if os.path.isfile(fout):
            LOG.info('CCDTAB exists, skipping download..')
            return fout
        LOG.info(fout)
        try:
            fout, httpmsg = request.urlretrieve(url=crds_url,filename=fout)

        except error.URLError as e:
            LOG.error(e)
            fout = 'N/A'
        else:
            msg = (
                'Successfully downloaded {} from https://hst-crds.stsci.edu/\n'
                'Path: {}'.format(ref_file, fout)
            )
            LOG.info(msg)
        finally:
            return fout

    def check_for_artifact(self, f, extname='dq', extnums=[1,2]):
        """ Scan the DQ extension for compression artifacts

        In early ACS images when the option for compressing data was available,
        there is a possibility of Reed-Solomon decoding errors being incorrectly
        classified as cosmic rays during the cosmic ray rejection step.
        """

        ext_tuples = [(extname, num) for num in extnums]
        ext_data = []
        with fits.open(f, mode='update') as hdu:
            for val in ext_tuples:
                try:
                    ext = hdu.index_of(val)
                except KeyError:
                    LOG.warning('{} is missing for {}'.format(val, f))
                else:
                    ext_data.append(hdu[ext].data)

                try:
                    exptime = hdu[0].header['EXPTIME']
                except KeyError as e:
                    LOG.warning('{}\n Searching SCI header'.format(e))
                    exptime = hdu[1].header['EXPTIME']


        # If second DQ ext is missing, only work with the first
        # Otherwise combine each DQ ext to make full-frame

        dq = concatenate(ext_data, axis=0)
        artifacts = where(dq == 2)[0]

        if artifacts.size > 0 or exptime < 0.1:
            return True
        else:
            return False


    def ACS(self, input, i):
        """ Run ACS cosmic ray rejection

        Parameters
        ----------
        input : list
            List of files to process with `acsrej`

        Returns
        -------
        tuple : list, bool
            The tuple contains the input list and a boolean flag. The flag will
            be True if the processing was successful and False if not.
        """


        output = 'tmp_crj_{}.fits'.format(i)
        failed = True
        try:
             acsrej.acsrej(input,
                          output=output,
                          verbose=True,
                          crrejtab=self.crrejtab,
                          crmask=True,
                          crsigmas='8,6,4',
                          initgues='med',
                          skysub='mode')
             if not os.path.isfile(output):
                 raise (FileNotFoundError)
        except Exception as e:
            msg = ('{}\n Failed to '
                   'process input list\n {}\n {}'.format(e,
                                                       '\n'.join(input),
                                                       self._msg_div))
            LOG.error(msg)
            return input, failed
        else:
            failed = False
            return input, failed
            # self.output['passed'].append(input)

    def WFC3(self, input, i):
        """ Run WFC3 cosmic ray rejection

        Parameters
        ----------
        input : list
            list of files to process with `wf3rej`

        Returns
        -------
        tuple : list, bool
            The tuple contains the input list and a boolean flag. The flag will
            be True if the processing was successful and False if not.
        """
        output = 'tmp_crj_{}.fits'.format(i)
        # if the file exist increment _i by one before processing.

        failed = True

        try:
            wf3rej(input,
                   output=output,
                   verbose=True,
                   crrejtab=self.crrejtab,
                   crsigmas='6,5,4',
                   crmask=True,
                   initgues='med',
                   skysub='mode')

            if not os.path.isfile(output):
                raise(FileNotFoundError)

        except Exception as e:
            msg = ('{}\n Failed to '
                   'process input list\n {}\n {}'.format(e,
                                                       '\n'.join(input),
                                                       self._msg_div))

            LOG.error(msg)
            return input, failed
        else:
            failed = False
            return input, failed
            # self.output['passed'].append(input)

    def STIS(self, input, i):
        """ Run STIS cosmic ray rejection

        Parameters
        ----------
        input : list
            List of files to process with `ocrreject`

        Returns
        -------
        tuple
            The tuple contains the input list and a boolean flag. The flag will
            be True if the processing was successful and False if not.
        """
        failed = True
        if len(input) < 2:
            return input, failed
        else:
            output = 'tmp_crj_{}.fits'.format(i)

            try:
                ocrreject.ocrreject(' '.join(input),
                                    output=output,
                                    crrejtab=self.crrejtab,
                                    verbose=True,
                                    crsigmas='6,5,4',
                                    crmask='yes',
                                    initgues='med',
                                    skysub='mode'
                                    )
                if not os.path.isfile(output):
                    raise(FileNotFoundError)
            except Exception as e:
                msg = ('{}\n Failed to '
                       'process input list\n {}\n {}'.format(e,
                                                           '\n'.join(input),
                                                           self._msg_div))
                LOG.error(msg)
                return input, failed
            else:
                failed = False
                return input, failed
                # self.output['passed'].append(input)

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
            has_artifact = self.check_for_artifact(
                f,
                extname='dq',
                extnums=self.instr_cfg['instr_params']['extnums']
            )
            if has_artifact:
                self.output['failed'].append(f)
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
            if len(val) > 30:
                LOG.info('{} exceeds input limit, ' \
                      'splitting to smaller groups'.format(key))
                split = array_split(val, 3)
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
        results = []
        data = self.format_inputs()
        randints = [random.randint(0, 2500) for i in range(len(data))]
        pairs = zip(data, randints)
        pipeline_dir = os.getcwd()

        os.chdir(self._data_dir)
        # TODO: add a cleaner implementation for downloading CCDTAB
        # TODO: pass data to download_refile(), parse all CCDTAB filenames and
        # TODO: only download the unique files in the list.
        if 'acs' in self.instr.lower():
            # The full path to the CCDTAB is too long for a FITS header keyword
            # Instead, we change to the data directory and run the analysis there
            # Once we are finished, we change back.

            # For the ACS images we need to download the correct CCDTAB
            for dataset in data:
                for f in dataset:
                    with fits.open(f, mode='update') as hdu:
                        jref_ccdtab = hdu[0].header['CCDTAB']
                        jref_ccdtab = jref_ccdtab.split('$')[-1]
                        local_ccdtab = self._download_reffile(jref_ccdtab)
                        hdu[0].header['CCDTAB'] = jref_ccdtab


            results = [dask.delayed(self.ACS)(d, i) for d, i in pairs]
            results = list(dask.compute(*results,
                                        scheduler='processes',
                                        num_workers=os.cpu_count()))

        elif 'wfc3' in self.instr.lower():
            # For the ACS images we need to download the correct CCDTAB
            for dataset in data:
                for f in dataset:
                    with fits.open(f, mode='update') as hdu:
                        jref_ccdtab = hdu[0].header['CCDTAB']
                        jref_ccdtab = jref_ccdtab.split('$')[-1]
                        local_ccdtab = self._download_reffile(jref_ccdtab)
                        hdu[0].header['CCDTAB'] = jref_ccdtab

            # WFC3 only has one CCDTAB, so we've downlodaed it locally already
            results = [dask.delayed(self.WFC3)(d, i) for d, i in pairs]
            results = dask.compute(*results,
                                   scheduler='processes',
                                   num_workers=os.cpu_count())

        elif 'stis' in self.instr.lower():
            results = [dask.delayed(self.STIS)(d, i) for d, i  in pairs]

            results = dask.compute(*results,
                         scheduler='processes',
                         num_workers=os.cpu_count())
        os.chdir(pipeline_dir)
        # Each computation returns a tuple (input, failed). Use this to sort
        # which files were processed successful and which were not

        passed = [result[0] for result in results if not result[1]]
        failed = [result[0] for result in results if result[1]]

        # success = [result for result in results if result is not None]
        # failed = [result for result in results if result is None]

        for flist in passed:
            self.output['passed'] += flist

        for flist in failed:
            self.output['failed'] += flist

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
if __name__ == "__main__":
    # For debugging purposes
    flist = glob.glob('/Users/nmiles/hst_cosmic_rays/data/ACS/WFC/mastDownload/HST/*/*flt.fits')
    p = ProcessCCD('ACS_WFC',flist)
    p.sort()
    p.cr_reject()
