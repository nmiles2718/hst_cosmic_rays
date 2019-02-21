#!/usr/bin/env python
from collections import defaultdict
import glob
import itertools
import os
import time

# non native imports
from astropy.io import fits
import dask
from numpy import array
from numpy import array_split
from numpy import concatenate
from numpy import where
from numpy.random import randint

from acstools import acsrej
from stistools import ocrreject
from wfc3tools import wf3rej

class ProcessData(object):
    def __init__(self, instr, flist):
        self.instr = instr
        self.flist = flist
        self.num = len(flist)
        self.input = {}
        self.output = defaultdict(list)
        self.dq = None
        self.i = 0

    def check_for_artifact(self, f):
        """ Grab the DQ extensions from fits file
        """
        dq1 = ('dq', 1)  # Chip 2
        dq2 = ('dq', 2)  # Chip 1
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
            except KeyError:
                print('{1} is missing for {0}'.format(f, dq1))
                ext1 = None
            try:
                ext2 = hdu.index_of(dq2)
                ext2_data = hdu[ext2].data
            except KeyError:
                print('{1} is missing for {0}'.format(f, dq2))
                ext2 = None
        # If second DQ ext is missing, only work with the first
        # Otherwise combine each DQ ext to make full-frame
        if not ext2:
            self.dq = ext1_data
        else:
            self.dq = concatenate([ext1_data, ext2_data], axis=0)
        artifacts = where(self.dq == 2)[0]

        if artifacts.size > 0 or expt < 0.1:
            return True
        else:
            return False



    def ACS(self, input):
        if 'wfc' in self.instr.lower():
            crrejtab = './../crrejtab/ACS/n4e12511j_crr_WFC.fits'
        else:
            crrejtab = './../crrejtab/ACS/n4e12510j_crr_HRC.fits'
        output = 'tmp_crj_{}.fits'.format(self.i)

        # if the file exist increment i by one before processing.
        while os.path.isfile(output):
            self.i+= randint(0, 500)
            output = 'tmp_crj_{}.fits'.format(self.i)
        try:
             acsrej.acsrej(input=input,
                          output=output,
                          verbose=True,
                          crsigmas='5,4,3',
                          crrejtab=crrejtab,
                          crmask=True,
                          initgues='med',
                          skysub='mode')
             if not os.path.isfile(output):
                 raise (FileNotFoundError)
        except Exception:
            self.output['failed'].append(input)
        else:
            self.output['passed'].append(input)

    def WFC3(self, input):
        output = 'tmp_crj_{}.fits'.format(self.i)
        # if the file exist increment i by one before processing.
        while os.path.isfile(output):
            self.i += 1
            output = 'tmp_crj_{}.fits'.format(self.i)

        crrejtab = './../crrejtab/WFC3/n9i1435li_crr_UVIS.fits'
        for f in input:
            with fits.open(f,mode='update') as hdu:
                hdu[0].header['CCDTAB']='./../crrejtab/WFC3/t291659mi_ccd.fits'
        try:
            wf3rej(input=input,
                   output='tmp_crj_{}.fits'.format(self.i),
                   verbose=True,
                   crrejtab=crrejtab,
                   crmask=True,
                   initgues='med',
                   skysub='mode')
            if not os.path.isfile(output):
                raise(FileNotFoundError)
        except Exception as e:
            print('-'*60)
            print('Failed to process input list', input)
            self.output['failed'].append(input)
        else:
            self.output['passed'].append(input)

    def STIS(self, input):
        if len(input) < 2:
            self.output['failed'].append(input)
        else:
            output = 'tmp_crj_{}.fits'.format(self.i)

            # if the file exist increment i by one before processing.
            while os.path.isfile(output):
                self.i += 1
                output = 'tmp_crj_{}.fits'.format(self.i)

            crrejtab = './../crrejtab/STIS/j3m1403io_crr.fits'
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
                self.output['failed'].append(input)
            else:
                self.output['passed'].append(input)

    def sort(self):
        """
        We must sort the files by ex posure times and then check to see
        that each list of files with a given exposure time is less than
        an arbitrary upper limit of say, 60 files. If we start combining larger
        chunks we will potentially run into a memory issue becuase of how
        the image combination in handled.

        """

        found_exptimes = []
        found_sizes = []
        removed = []
        files_to_process = self.flist
        # we have to make a list of all exptimes, then sort by unique ones
        for f in files_to_process:
            has_artifact = self.check_for_artifact(f)
            if has_artifact:
                removed.append(f)
                print('Removing {} from analysis'.format(f))
                self.flist.remove(f)

        for f in self.flist:
            with fits.open(f) as hdu:
                prhdr = hdu[0].header
                scihdr = hdu[1].header
            if 'exptime' in prhdr:
                found_exptimes.append(prhdr['exptime'])
            elif 'exptime' in scihdr:
                found_exptimes.append(scihdr['exptime'])
            found_sizes.append('{}'.format(prhdr['CCDAMP']))

        self.output['failed'].append(removed)
        # Find the unique values
        unique_sizes = set(found_sizes)
        unique_exp = set(found_exptimes)
        self.flist = array(self.flist)
        for ap in unique_sizes:
            for t in unique_exp:
                idx = where((array(found_exptimes) == t) &
                                 (array(found_sizes) == ap))[0]
                if not idx.any():
                   pass
                else:
                    print('Found {} images with size={} and t={}'.format(len(idx),
                                                            ap,
                                                            t))
                    print('-'*60)
                    #for f in self.flist[idx]:
                    #    print(fits.getval(f, 'exptime', ext=('sci',1)))
                    self.input['{}_{}'.format(ap, t)] = self.flist[idx]

        # Now we check to make sure each list of files is less than the limit
        for key, val in self.input.items():
            if len(val) > 50:
                print('{} exceeds input limit, ' \
                      'splitting to smaller groups'.format(key))
                split = array_split(val, 4)
            # Convert tuple to list for checking later
                self.input[key] = tuple(split)

    def format_inputs(self):
        data = []
        for key in self.input.keys():
            if isinstance(self.input[key], tuple):
                for val in self.input[key]:
                    data.append(list(val))
            else:
                data.append(list(self.input[key]))
        return data

    def cr_reject(self):
        start_time = time.time()
        print('-'*60)
        print('Rejecting cosmic rays...')
        print('-'*60)
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
        print('Done!')
        end_time = time.time()
        print((end_time - start_time)/60)


if __name__ == "__main__":
    # For debugging purposes
    flist = glob.glob('./../crrejtab/STIS/mastDownload/HST/*/*flt.fits')
    p = ProcessData('stis_ccd',flist)
    p.sort()
    p.cr_reject()
