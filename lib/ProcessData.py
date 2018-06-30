#!/usr/bin/env python
from collections import defaultdict
import itertools
import os

# non native imports
from astropy.io import fits
from acstools import acsrej
from numpy import where
from numpy import array
from numpy import array_split
from stistools import ocrreject
from wfc3tools import wf3rej

class ProcessData(object):
    def __init__(self, instr, flist):
        self.instr = instr
        self.flist = array(flist)
        self.num = len(flist)
        self.input = {}
        self.output = defaultdict(list)

    def ACS(self, input, i):
        if 'wfc' in self.instr.lower():
            crrejtab = './../crrejtab/ACS/n4e12511j_crr_WFC.fits'
        else:
            crrejtab = './../crrejtab/ACS/n4e12510j_crr_HRC.fits'
        output = 'tmp_crj_{}.fits'.format(i)

        # if the file exist increment i by one before processing.
        while os.path.isfile(output):
            i+=1
            output = 'tmp_crj_{}.fits'.format(i)
        try:
            acsrej.acsrej(input=input,
                          output='tmp_crj_{}.fits'.format(i),
                          verbose=True,
                          crrejtab=crrejtab,
                          crmask=True,
                          initgues='med',
                          skysub='mode')
        except Exception as e:
            self.output['failed'].append(input)
        else:
            self.output['passed'].append(input)

    def WFC3(self, input, i):
        print(input)

        output = 'tmp_crj_{}.fits'.format(i)
        # if the file exist increment i by one before processing.
        while os.path.isfile(output):
            i += 1
            output = 'tmp_crj_{}.fits'.format(i)

        crrejtab = './../crrejtab/WFC3/n9i1435li_crr_UVIS.fits'
        try:
            wf3rej(input=input,
                   output='tmp_crj_{}.fits'.format(i),
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

    def STIS(self, input, i):
        output = 'tmp_crj_{}.fits'.format(i)

        # if the file exist increment i by one before processing.
        while os.path.isfile(output):
            i += 1
            output = 'tmp_crj_{}.fits'.format(i)

        crrejtab = './../crrejtab/STIS/j3m1403io_crr.fits'
        try:
            ocrreject.ocrreject(' '.join(input),
                                output='tmp_crj_{}.fits'.format(i),
                                crrejtab=crrejtab,
                                verbose=True,
                                crmask='yes',
                                initgues='med',
                                skysub='mode'
                                )
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
        # we have to make a list of all exptimes, then sort by unique ones
        for f in self.flist:
            with fits.open(f) as hdu:
                prhdr = hdu[0].header
                scihdr = hdu[1].header
            if 'exptime' in prhdr:
                found_exptimes.append(prhdr['exptime'])
            elif 'exptime' in scihdr:
                found_exptimes.append(scihdr['exptime'])

        # Find the unique values
        unique_exp = set(found_exptimes)
        for t in unique_exp:
            print(self.flist[where(array(found_exptimes) == t)[0]])
            self.input[t] = self.flist[where(array(found_exptimes) == t)[0]]

        # Now we check to make sure each list of files is less than the limit
        for key, val in self.input.items():
            if len(val) > 60:
                print('Exceeding input limit, splitting to smaller groups')
                split = array_split(val, 2)
            # Convert tuple to list for checking later
                self.input[key] = tuple(split)


    def cr_reject(self):
        print('-'*60)
        print('Rejecting cosmic rays...')
        print('-'*60)
        i=0
        if 'acs' in self.instr.lower():
            for key in self.input.keys():
                if isinstance(self.input[key], tuple):
                    self.ACS(list(self.input[key][0]), i)
                    self.ACS(list(self.input[key][1]), i)
                else:
                    self.ACS(list(self.input[key]), i)
        elif 'wfc3' in self.instr.lower():
            for key in self.input.keys():
                if isinstance(self.input[key],tuple):

                    self.WFC3(list(self.input[key][0]), 0)
                    self.WFC3(list(self.input[key][1]), 1)
                else:
                    self.WFC3(list(self.input[key]), 2)
        elif 'stis' in self.instr.lower():
            for key in self.input.keys():
                if isinstance(self.input[key], tuple):
                    self.STIS(list(self.input[key][0]), 0)
                    self.STIS(list(self.input[key][1]), 1)
                else:
                    self.STIS(list(self.input[key]), 2)
        if 'failed' in self.output.keys():
            self.output['failed'] = list(itertools.chain.from_iterable(
                self.output['failed']))
        print('Done!')

