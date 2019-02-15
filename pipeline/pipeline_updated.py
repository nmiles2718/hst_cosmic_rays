#!/usr/bin/env python

# native packages
import argparse
import glob
import logging
import os
import time

# external packages
import numpy as np
import yaml

# local packages
import download.download as download
import initialize.initialize as initialize
import process.process as process


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
                    default=False)

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
        self._processing_times = {
            'download': 0,
            'processing': 0,
            'analysis_time': 0
        }
        self._cfg_file = os.path.join(self._base,
                                     'CONFIG',
                                     'pipeline_config.yaml')

        # Load the CONFIG file
        with open(self._cfg_file, 'r') as fobj:
            self._cfg = yaml.load(fobj)
        self._instr_cfg = self.cfg[self._instr]


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

    @property
    def ccd(self):
        return self._ccd

    @ccd.getter
    def ccd(self):
        """Switch for toggling on the CCD analysis"""
        return self._ccd

    @cfg.getter
    def cfg(self):
        """Configuration object returned by parsing the
        :py:attr:`~pipeline_updated.CosmicRayPipeline.cfg_file`"""
        return self._cfg

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

    def run_processing(self):
        """Process the data"""
        search_pattern = os.path.join(
            self.base,
            *self.instr_cfg['search_pattern'].split('/')
        )
        flist = glob.glob(search_pattern)

        if self.ccd:
            processor = process.ProcessCCD(self.instr, flist=flist)
            processor.sort()
            processor.cr_reject()

        elif self.ir:
            processor = process.ProcessIR(flist=flist)
            processor.decompose()

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
                LOG.info('Analyzing data from {} to {}'.format(start.iso,
                                                               stop.iso))
                if self.download:
                    download_time = self.run_downloader(range=(start, stop),
                                                        downloader=downloader)
                    self.processing_times['download'] = download_time

                if self.process:
                    process_time = self.run_processing()
                    self.processing_times['processing'] = process_time


                break
            break








if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    p = CosmicRayPipeline(**args)
    p.run()