#!/usr/bin/env python

# native packages
import argparse
from collections import defaultdict
import glob
import logging
import os
import shutil
import sys
import time

# external packages
import dask
import numpy as np
import pandas as pd
import yaml

# local packages
import download.download as download
import label.labeler as labeler
import process.process as process
import stat_utils.statshandler as statshandler
import utils.datahandler as datahandler
import utils.initialize as initialize
import utils.metadata as metadata
import utils.sendit as sendit


__taskname__ = "pipeline"
__author__ = "Nathan Miles"
__version__ = "1.0"
__vdate__ = "22-Jan-2019"

# __all__ = 'CosmicRayPipeline'


parser = argparse.ArgumentParser()

parser.add_argument('-aws',
                    action='store_true',
                    help='Flag for using AWS for downloads. Only to be used '
                         'when the pipeline is run on EC2',
                    default=False)

parser.add_argument('-download',
                    help='Download the data',
                    action='store_true',
                    default=False)

parser.add_argument('-process',
                    help='Process the raw data',
                    action='store_true',
                    default=True)

parser.add_argument('-ccd',
                    help='Switch for process CCD data',
                    action='store_true',
                    default=True)

parser.add_argument('-ir',
                    help='Switch for process IR data',
                    action='store_true',
                    default=False)


parser.add_argument('-analyze',
                    help='Run the analysis and extract cosmic ray statistics',
                    action='store_true',
                    default=True)

parser.add_argument('-instr',
                    default='stis_ccd',
                    help='HST instrument to process (acs_wfc, '
                         'wfc3_uvis, stis_ccd, acs_hrc)')

parser.add_argument('-initialize',
                    action='store_true',
                    default=False,
                    help='Initialize the HDF5 files for the instrument. \n'
                         '**Warning**: Should only be included the first time'
                         ' the pipeline is run becuase it will overwrite any '
                         'pre-existing HDF5 files.')

parser.add_argument('-store_downloads',
                    action='store_true',
                    default=False,
                    help='Switch for toggling on the storage of downloaded data'
                    )

logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s',
                    level=logging.DEBUG)
LOG = logging.getLogger('CosmicRayPipeline')
LOG.setLevel(logging.INFO)


class CosmicRayPipeline(object):
    def __init__(self, aws=None, analyze=None, download=None, ccd=None,
                 ir=None, instr=None, initialize=None,
                 process=None, store_downloads=None):

        # Initialize Args
        self._aws = aws
        self._analyze = analyze
        self._download = download
        self._ccd = ccd
        self._ir = ir
        self._instr = instr.upper()
        self._initialize = initialize
        self._process = process
        self._store_downloads = store_downloads

        # Necessary evil to dynamically build absolute paths
        self._mod_dir = os.path.dirname(os.path.abspath(__file__))
        self._base = os.path.join('/', *self._mod_dir.split('/')[:-1])
        self._flist = None
        self._processing_times = {
            'download': 0,
            'cr_rejection': 0,
            'analysis': 0
        }
        self._cfg_file = os.path.join(self._base,
                                     'CONFIG',
                                     'pipeline_config.yaml')

        # Load the CONFIG file
        with open(self._cfg_file, 'r') as fobj:
            self._cfg = yaml.load(fobj)
        self._instr_cfg = self.cfg[self._instr]

        self._search_pattern = os.path.join(
            self.base,
            *self.instr_cfg['search_pattern'].split('/')
        )

    @property
    def aws(self):
        return self._aws

    @aws.getter
    def aws(self):
        """Switch for toggling on AWS functionality for downloads"""
        return self._aws

    @property
    def analyze(self):
        return self._analyze

    @analyze.getter
    def analyze(self):
        """Switch for toggling on the analysis step of the pipeline """
        return self._analyze

    @property
    def base(self):
        return self._base

    @base.getter
    def base(self):
        """Base path of the pipleine repository `~/hst_cosmic_rays/`"""
        return self._base

    @property
    def cfg(self):
        return self._cfg

    @cfg.getter
    def cfg(self):
        """Configuration object returned by parsing the
        :py:attr:`~pipeline_updated.CosmicRayPipeline.cfg_file`"""
        return self._cfg

    @property
    def ccd(self):
        return self._ccd

    @ccd.getter
    def ccd(self):
        """Switch for toggling on the CCD analysis"""
        return self._ccd

    @property
    def cfg_file(self):
        return self._cfg_file

    @cfg_file.getter
    def cfg_file(self):
        """Path to the pipeline configuration file

        The config file is stored in `~/hst_cosmic_rays/CONFIG/`"""
        return self._cfg_file

    @property
    def download(self):
        return self._download

    @download.getter
    def download(self):
        """Switch for toggling on the download step of the pipeline"""
        return self._download

    @property
    def instr(self):
        return self._instr

    @instr.getter
    def instr(self):
        """Name of the instrument that is going to be analyzed"""
        return self._instr

    @property
    def instr_cfg(self):
        return self._instr_cfg

    @instr_cfg.getter
    def instr_cfg(self):
        """Instrument specific configuration"""
        return self._instr_cfg

    @property
    def initialize(self):
        return self._initialize

    @initialize.getter
    def initialize(self):
        """Switch for toggling on the initialization of HDF5 data files"""
        return self._initialize

    @property
    def ir(self):
        return self._ir

    @ir.getter
    def ir(self):
        """Switch for toggling on the IR analysis"""
        return self._ir

    @property
    def process(self):
        return self._process

    @process.getter
    def process(self):
        """Switch for toggling on the processing step of the pipeline"""
        return self._process

    @property
    def processing_times(self):
        return self._processing_times

    @processing_times.getter
    def processing_times(self):
        """Container for holding the processing time required by each step"""
        return self._processing_times

    @processing_times.setter
    def processing_times(self, value):
        self._processing_times = value

    @property
    def flist(self):
        return self._flist

    @flist.getter
    def flist(self):
        """List of files to process"""
        return self._flist

    @flist.setter
    def flist(self, value):
        self._flist = value

    @property
    def search_pattern(self):
        return self._search_pattern

    @search_pattern.getter
    def search_pattern(self):
        """Search pattern used to find files to process"""
        return self._search_pattern

    @search_pattern.setter
    def search_pattern(self, value):
        self._search_pattern = value

    @property
    def store_downloads(self):
        return self._store_downloads

    @store_downloads.getter
    def store_downloads(self):
        """Switch for saving the downloaded files"""
        return self._store_downloads

    def run_downloader(self, range, downloader):
        """Download the data"""
        start_time = time.time()
        downloader.query(range=range, aws=False)
        downloader.download(range[0].datetime.date().isoformat())
        end_time = time.time()
        return (end_time - start_time)/60

    def run_labeling_single(self, fname):
        """Convenience method to facilitate parallelization"""

        file_metadata = metadata.GenerateMetadata(fname,
                                                  instr_cfg=self.instr_cfg)

        # Get image metadata
        file_metadata.get_image_data()

        # Get pointing info
        file_metadata.get_wcs_info()

        # Get HST location info
        file_metadata.get_observatory_info()

        cr_label = labeler.CosmicRayLabel(fname)

        label_params = {
            'deblend': False,
            'use_dq': True,
            'extnums': self.instr_cfg['instr_params']['extnums'],
            'threshold_l': 2,
            'threshold_u': 1500,
            'plot': False
        }

        if self.ccd:
            cr_label.run_ccd_label(**label_params)

        # Compute the integration time
        integration_time = cr_label.exptime + \
                           self.instr_cfg['instr_params']['readout_time']

        cr_stats = statshandler.Stats(cr_label, integration_time)
        cr_stats.compute_cr_statistics()
        cr_stats_dict = {
            'cr_affected_pixels': cr_stats.cr_affected_pixels,
            'incident_cr_rate': cr_stats.incident_cr_rate,
            # Note that we save BOTH versions of CR sizes measurements
            'sizes': np.asarray([cr_stats.size_in_sigmas,
                                cr_stats.size_in_pixels]),
            'shapes': cr_stats.shapes,
            'energy_deposited': cr_stats.energy_deposited
        }

        return cr_stats_dict, file_metadata

    def run_labeling_all(self, chunk_num):
        """Run the labeling analysis and compute the statistics"""
        start_time = time.time()

        delayed_objects = [
            dask.delayed(self.run_labeling_single)(f) for f in self.flist
        ]

        results = list(dask.compute(*delayed_objects,
                                    scheduler='single-threaded',
                                    num_workers=1))
        # results = self.run_labeling_single(self.flist[0])

        cr_stats, file_metdata = zip(*results)

        datawriter = datahandler.DataWriter(cfg=self.cfg,
                                            chunk_num=chunk_num,
                                            cr_stats=cr_stats,
                                            file_metadata=file_metdata,
                                            instr=self.instr)
        datawriter.write_results()
        end_time = time.time()

        return (end_time - start_time)/60., results

    def run_processing(self):
        """Process the data"""
        start_time = time.time()

        self.flist = glob.glob(self.search_pattern)

        if self.ccd:
            processor = process.ProcessCCD(self.instr,
                                           self.instr_cfg,
                                           flist=self.flist)
            processor.sort()
            processor.cr_reject()
            if 'failed' in processor.output.keys():
                print(processor.output['failed'])
                failed = set(list(processor.output['failed']))
                msg = ('{} files failed, '
                       'removing from processing list..'.format(len(failed)))
                LOG.warning(msg)
                # remove the failed files for the list of files to process
                self.flist = list(set(self.flist).difference(failed))

        elif self.ir:
            processor = process.ProcessIR(flist=self.flist)
            processor.decompose()

        end_time = time.time()
        return (end_time - start_time) / 60

    def send_email(self, start, stop, results):
        """

        Returns
        -------

        """

        # Compute some averages for each statistics (if applicable)
        msg_data = defaultdict(list)
        for cr_stat, file_info in results:
            msg_data['filename'].append(
                os.path.basename(file_info.fname)
            )
            msg_data['integration_time'].append(file_info.metadata['integration_time'])
            msg_data['date'].append(file_info.metadata['date'])

            msg_data['avg_shape'].append(np.nanmean(cr_stat['shapes']))

            msg_data['avg_size [sigma]'].append(
                np.nanmean(cr_stat['sizes'][0]))

            msg_data['avg_size [pix]'].append(np.nanmean(cr_stat['sizes'][1]))
            msg_data['avg_energy_deposited [e]'].append(
                np.nanmean(cr_stat['energy_deposited'])
            )
            msg_data['CR count'].append(len(cr_stat['energy_deposited']))

        df = pd.DataFrame(msg_data)
        df = df.set_index(keys=['date'], drop=True)
        df.sort_index(inplace=True)


        e = sendit.Emailer(df=df,
                           processing_times=self.processing_times)
        subj = ('Finished analyzing '
               '{} darks from {} to {}'.format(self.instr,
                                               start.datetime.date(),
                                               stop.datetime.date()))
        e.subject = subj
        e.sender = ['nmiles','stsci.edu']
        e.recipient = ['nmiles', 'stsci.edu']

        e.SendEmail(gif=False)

    def _pipeline_cleanup(self, start, stop):
        """Handle necessary cleanup steps required at the end of the pipeline

        Returns
        -------

        """
        LOG.info('Initiating pipeline cleanup...')
        # Write out the dates of the range that was just processed
        processed_fname = os.path.join(
            self.base, 'CONFIG', 'processed_dates_{}.txt'.format(self.instr)
        )
        with open(processed_fname,'a+') as fobj:
            fobj.write('{} {}\n'.format(start.iso, stop.iso))


        # Remove any files that were generated as a result of CR processing
        # for the CCD imagers
        crjs = glob.glob('./tmp*')

        if crjs is not None:
            for a in crjs:
                os.remove(a)

        # Generate the path to the download directory
        download_dir = os.path.join(
            self.base, *self.instr_cfg['astroquery']['download_dir'].split('/')
        )
        LOG.info(
            'Removing all files downloaded into:\n{}'.format(download_dir)
        )
        # Delete all the files stored in mastDownload
        shutil.rmtree('{}/mastDownload'.format(download_dir),
                      ignore_errors=True)

    def run(self):
        """ Run the pipeline according to the passed command line args

        Returns
        -------

        """
        # Get some initialization info required for the pipeline to run
        initializer_obj = initialize.Initializer(self.instr, self.cfg)
        initializer_obj.initialize_dates()
        initializer_obj.get_date_ranges()
        initializer_obj.get_processed_ranges()

        if self.initialize:
            initializer_obj.initialize_HDF5()

        # Initialize the downloader
        downloader = download.Downloader(self.instr, self.instr_cfg)

        # Divide up the dates into 4 chunks
        date_chunks = np.array_split(initializer_obj.dates, 4)
        for i, chunk in enumerate(date_chunks):
            for (start, stop) in chunk:
                if '{} {}'.format(start.iso, stop.iso) in \
                        initializer_obj.previously_analyzed:
                    LOG.info('Already analyzed {} to {}\n'.format(start.iso,
                                                                  stop.iso))
                    continue

                # Start the analysis
                LOG.info('Analyzing data from {} to {}'.format(start.iso,
                                                               stop.iso))
                if self.download:
                    download_time = self.run_downloader(range=(start, stop),
                                                        downloader=downloader)
                    self.processing_times['download'] = download_time

                if self.process:
                    process_time = self.run_processing()
                    self.processing_times['cr_rejection'] = process_time


                if self.analyze:
                    analysis_time, results = self.run_labeling_all(
                        chunk_num=i + 1
                    )
                    self.processing_times['analysis'] = analysis_time

                self.processing_times['total'] = sum(
                    list(self.processing_times.values())
                )

                for key, value in self.processing_times.items():
                    print(key, value)

                # Clean up the files and write out the range just processed
                self._pipeline_cleanup(start, stop)

                # Send the final email
                self.send_email(start, stop, results)


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    p = CosmicRayPipeline(**args)
    # p = CosmicRayPipeline(instr='stis_ccd', ccd=True, download=False,
    #                       process=True, analyze=True, initialize=True)
    p.run()
    # cmd = 'python pipeline_updated.py -instr stis_ccd -download -process -analyze -initialize'
    # os.system(cmd)