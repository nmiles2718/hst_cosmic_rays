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
                    default=False)

parser.add_argument('-ccd',
                    help='Switch for processing CCD data',
                    action='store_true',
                    default=False)

parser.add_argument('-ir',
                    help='Switch for processing IR data',
                    action='store_true',
                    default=False)

parser.add_argument('-chunks',
                    help='Number of chunks to break the generated results into. '
                         '\nFor example, if `-chunks 2` is passed, then two HDF5 '
                         'files for each statistic will be generated. The first '
                         'half of the dataset will be written to file 1 and the '
                         'second half will be written to file 2. This is to '
                         'offset the degradation in write time as the number of '
                         'datasets stored in the HDF5 increases.',
                    type=int,
                    default=4)


parser.add_argument('-analyze',
                    help='Switch for analyzing and extract cosmic ray statistics',
                    action='store_true',
                    default=False)

parser.add_argument('-use_dq',
                    help='Switch for using the DQ arrays to perform labeling',
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
                         '\n**Warning**: Should only be included the first time'
                         ' the pipeline is run becuase it will overwrite any '
                         'pre-existing HDF5 files.')


logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s',
                    )
LOG = logging.getLogger('CosmicRayPipeline')
LOG.setLevel(logging.INFO)


class CosmicRayPipeline(object):
    def __init__(self, aws=None, analyze=None, download=None, ccd=None,
                 chunks=None, ir=None, instr=None, initialize=None,
                 process=None, store_downloads=None, use_dq=None, test=None):
        """ Class for combining the individual tasks into a single pipeline.
        """
        # Initialize Args
        self._aws = aws
        self._analyze = analyze
        self._download = download
        self._ccd = ccd
        self._chunks = chunks
        self._ir = ir
        self._instr = instr.upper()
        self._initialize = initialize
        self._process = process
        self._store_downloads = store_downloads
        self._use_dq = use_dq

        # Necessary evil to dynamically build absolute paths
        self._mod_dir = os.path.dirname(os.path.abspath(__file__))
        self._base = os.path.join('/', *self._mod_dir.split('/')[:-1])
        self._flist = None
        self._processing_times = {
            'download': 0,
            'cr_rejection': 0,
            'analysis': 0
        }

        if test:
            self._cfg_file = os.path.join(self._base,
                                        'CONFIG',
                                        'testing_pipeline_config.yaml')
        else:
            self._cfg_file = os.path.join(self._base,
                                        'CONFIG',
                                        'pipeline_config.yaml')

        # Load the CONFIG file
        with open(self._cfg_file, 'r') as fobj:
            self._cfg = yaml.load(fobj)
        self._instr_cfg = self.cfg[self._instr]
        self._failed_observations = os.path.join(
            self.base,
            *self._instr_cfg['failed'].split('/')
        )
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
    def chunks(self):

        return self._chunks

    @chunks.getter
    def chunks(self):
        """Number of chunks to break the entire dataset into"""
        return self._chunks

    @chunks.setter
    def chunks(self, value):
        self._chunks = value

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
    def failed_observation(self):
        """List of any observations that failed to be processed for given month"""
        return self._failed_observations

    @failed_observation.setter
    def failed_observation(self, value):
        pass

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

    @property
    def use_dq(self):
        return self._use_dq

    @use_dq.getter
    def use_dq(self):
        """Switch for specifying what to use in the labeling analysis"""
        return self._use_dq

    def run_downloader(self, range, downloader):
        """Download the data

        Parameters
        ----------
        range : Tuple
         Tuple of `astropy.time.Time` objects defining the one month interval

        downloader : :py:class:`~download.download.Downloader`

        Returns
        -------
        runtime : float
            Time required to process in minutes
        """
        start_time = time.time()
        downloader.query(range=range, aws=self.aws)
        downloader.download(range[0].datetime.date().isoformat())
        end_time = time.time()
        return (end_time - start_time)/60

    def run_labeling_single(self, fname):
        """Run the labeling analysis on a single image

        Convenience method designed to facilitate the parallelization of the
        labeling analysis

        Parameters
        ----------
        fname : str
            Full path to file to be analyzed

        Returns
        -------
        file_metadata : :py:class:`~utils.metadata.GenerateMetadata`
            Object containing relevant metadata for input file

        cr_stats_dict : `dict`
            Dictionary containing the computed statistics
        """

        file_metadata = metadata.GenerateMetadata(fname,
                                                  instr=self.instr,
                                                  instr_cfg=self.instr_cfg)

        # Get image metadata
        file_metadata.get_image_data()

        # Get pointing info
        file_metadata.get_wcs_info()

        # Get HST location info
        file_metadata.get_observatory_info()

        cr_label = labeler.CosmicRayLabel(
            fname,
            gain_keyword=self.instr_cfg['instr_params']['gain_keyword']
        )

        label_params = {
            'deblend': False,
            'use_dq': self.use_dq,
            'extnums': self.instr_cfg['instr_params']['extnums'],
            'threshold_l': 2,
            'threshold_u': 1e5,
            'plot': False
        }

        if self.ccd:
            cr_label.run_ccd_label(**label_params)

        # Compute the integration time
        integration_time = cr_label.exptime + \
                           self.instr_cfg['instr_params']['readout_time']
        detector_size = self.instr_cfg['instr_params']['detector_size']

        cr_stats = statshandler.Stats(
            cr_label,
            integration_time=integration_time,
            detector_size=detector_size
        )
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
        """Run the labeling analysis and compute the statistics

        Run the labeling process to extract data for every CR in each image and
        save the results.

        Parameters
        ----------
        chunk_num : int
            Current chunk number we are analyzing. Used to write the results to
            the proper file

        Returns
        -------
        runtime : float
            Time required to process in minutes

        results : tuple
            Results from analyzing all files in `flist`.
        """
        start_time = time.time()

        delayed_objects = [
            dask.delayed(self.run_labeling_single)(f) for f in self.flist
        ]

        # dask.visualize(*delayed_objects, filename='labeling_graph.png')

        results = list(dask.compute(*delayed_objects,
                                    scheduler='processes',
                                    num_workers=os.cpu_count()))

        cr_stats, file_metdata = zip(*results)

        datawriter = datahandler.DataWriter(cfg=self.cfg,
                                            chunk_num=chunk_num,
                                            cr_stats=cr_stats,
                                            file_metadata=file_metdata,
                                            instr=self.instr)
        datawriter.write_results()
        end_time = time.time()

        return (end_time - start_time)/60., results

    def run_processing(self, start, stop):
        """ Process the data in the given time interval


        Parameters
        ----------
        start : `astropy.time.Time`
            Start date of the one month interval

        stop : `astropy.time.Time`
            Stop date of the one month interval

        Returns
        -------
        runtime : float
            Time required to process in minutes

        """
        start_time = time.time()

        # Process only if there are files to process
        if self.ccd and self.flist:
            processor = process.ProcessCCD(instr=self.instr,
                                           instr_cfg=self.instr_cfg,
                                           flist=self.flist)
            processor.sort()
            processor.cr_reject()
            if 'failed' in processor.output.keys():
                failed = set(list(processor.output['failed']))
                # Write out the failed files
                fout = '{}_{}_{}.txt'.format(
                    self.failed_observation.split('.')[0],
                    start.to_datetime().date(),
                    stop.to_datetime().date(),
                )
                with open(fout, 'a+') as fobj:
                    for f in processor.output['failed']:
                        fobj.write('{}\n'.format(f))

                msg = ('{} files failed, '
                       'removing from processing list..'.format(len(failed)))
                LOG.warning(msg)
                # remove the failed files for the list of files to process
                self.flist = list(set(self.flist).difference(failed))

        elif self.ir and self.flist:
            processor = process.ProcessIR(flist=self.flist)
            processor.decompose()

        end_time = time.time()
        return (end_time - start_time) / 60

    def send_email(self, start, stop, results):
        """Send email notifying user that a one-month chunk has completed

        Parameters
        ----------
        start : `astropy.time.Time`
            Start date of the one month interval

        stop : `astropy.time.Time`
            Stop date of the one month interval

        results : `list`
            The results from the CR analysis

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
            # Energy deposition follows Landau Distribution, median is a closer
            # estimate of the peak value
            msg_data['avg_energy_deposited [e]'].append(
                np.nanmedian(cr_stat['energy_deposited'])
            )
            msg_data['CR count'].append(len(cr_stat['energy_deposited']))
            msg_data['CR rate [CR/s/cm^2]'].append(cr_stat['incident_cr_rate'])

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
        e.sender = [
            self.config['email']['username'],
            self.config['email']['domain']
            ]
        e.recipient = e.sender

        if self.aws:
            e.SendEmailAWS()
        else:
            e.SendEmail(gif=False)


    def _pipeline_cleanup(self, start, stop, failed):
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

        if failed:
            failed_fname = os.path.join(
                self.base, 'CONFIG','failed_dates_{}.txt'.format(self.instr)
            )
            with open(failed_fname, 'a+') as fobj:
                fobj.write('{} {}\n'.format(start.iso, stop.iso))

        # Remove any files that were generated as a result of CR processing
        # for the CCD imagers
        generated_data = os.path.join(self.base,'data',self.instr.split('_')[0],'tmp*')
        crjs = glob.glob(generated_data)

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
            initializer_obj.initialize_HDF5(chunks=self.chunks)

        # Initialize the downloader
        downloader = download.Downloader(instr=self.instr,
                                         instr_cfg=self.instr_cfg)

        # Divide up the dates into chunks
        date_chunks = np.array_split(initializer_obj.dates, self.chunks)
        for i, chunk in enumerate(date_chunks):
            for (start, stop) in chunk:
                failed = False
                results = None
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

                self.flist = glob.glob(self.search_pattern)

                if self.process:
                    process_time = self.run_processing(start, stop)
                    self.processing_times['cr_rejection'] = process_time

                # Analyze the images and extract the results iff files
                # were successfully processed through CR rejection AND
                # the analyze flag is True.
                if self.analyze and self.flist:
                    analysis_time, results = self.run_labeling_all(
                        chunk_num=i + 1
                    )
                    self.processing_times['analysis'] = analysis_time
                else:
                    failed=True


                self.processing_times['total'] = sum(
                    list(self.processing_times.values())
                )

                # Clean up the files and write out the range just processed
                self._pipeline_cleanup(start, stop, failed)

                # Send the final email iff there were results computed
                if results:
                    self.send_email(start, stop, results)


if __name__ == '__main__':
    args = parser.parse_args()
    args = vars(args)
    p = CosmicRayPipeline(**args)
    p.run()
