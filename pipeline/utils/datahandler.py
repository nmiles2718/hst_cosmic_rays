#!/usr/bin/env python
"""
This module contains two classes for reading and writing the data generated
by the pipeline.
"""


import logging
import os
from astropy.time import Time
import h5py
from numpy import float32, ndarray

logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s',
                    level=logging.DEBUG)

LOG = logging.getLogger('datahandler')

LOG.setLevel(logging.INFO)


class DataWriter(object):
    """
    Class to handle writing out results for each datasets.
    """
    def __init__(self, cfg=None, chunk_num=None, cr_stats=None,
                 file_metadata=None, instr=None):
        """

        Parameters
        ----------
        cfg : dict
            Configuration object

        cr_stats : list
            List of `dict` containing of cosmic ray statistics to write out

        chunk_num : int
            Chunk number that the processed dataset belongs to (1 - 4)

        file_metadata : list
            List of file metadata objects

        """

        self._cfg = cfg
        self._cr_stats = cr_stats
        self._file_metadata = file_metadata
        self._chunk_num = chunk_num
        self._instr = instr

        self._mod_dir = os.path.dirname(os.path.abspath(__file__))
        self._base = os.path.join('/', *self._mod_dir.split('/')[:-2])
        self._msg_div = '-'*79

    @property
    def base(self):
        return self._base

    @base.getter
    def base(self):
        """Base path of the pipleine repository `~/hst_cosmic_rays/`"""
        return self._base

    @property
    def chunk_num(self):
        return self._chunk_num

    @chunk_num.getter
    def chunk_num(self):
        """Chunk number we are analyzing"""
        return self._chunk_num

    @property
    def cfg(self):
        return self._cfg

    @cfg.getter
    def cfg(self):
        """Configuration object returned by parsing the
        :py:attr:`~pipeline_updated.CosmicRayPipeline.cfg_file`"""
        return self._cfg

    @property
    def cr_stats(self):
        return self._cr_stats

    @cr_stats.getter
    def cr_stats(self):
        """:py:class:`~stat_utils.statshandler.ComputeStats`
            object containing all the statistics to write out"""
        return self._cr_stats

    @property
    def file_metadata(self):
        return self._file_metadata

    @file_metadata.getter
    def file_metadata(self):
        """A list of :py:class:`~utils.metadata.GenerateMetadata` objects"""
        return self._file_metadata

    @property
    def instr(self):
        return self._instr

    @instr.getter
    def instr(self):
        """One of the valid instrument names"""
        return self._instr

    def write_statistic(self, statistic):
        """Convenience method for writing out the statistics

        Parameters
        ----------
        statistic : str
            One of the valid statistics to write out.

        Returns
        -------

        """
        msg = ('Writiting out results\n '
               'statistic: {}\n '
               'number of datasets: {}\n {}\n'.format(statistic,
                                                 len(self.cr_stats),
                                                 self._msg_div))
        LOG.info(msg)

        rel_path = self.cfg[self.instr]['hdf5_files'][statistic]
        full_path = os.path.join(self.base, *rel_path.split('/'))

        fout = full_path.replace('.hdf5', '_{}.hdf5'.format(self.chunk_num))

        with h5py.File(fout, 'a', libver='latest') as f:
            grp = f[statistic]
            for file_info, stats in zip(self.file_metadata, self.cr_stats):
                dset_name = os.path.basename(file_info.fname)
                dset = grp.create_dataset(name=dset_name,
                                          data=stats[statistic],
                                          dtype=float32)

                for (key, val) in file_info.metadata.items():
                    # Check the datatype and save it accordingly
                    if isinstance(val, ndarray):
                        dset.attrs.create(name=key,
                                          data=val,
                                          shape=val.shape,
                                          dtype=float32)

                    elif isinstance(val, Time):
                        dset.attrs[key] = val.iso
                    else:
                        dset.attrs[key] = val

    def write_results(self):
        """Write out all the results for the analyzed dataset

        Returns
        -------

        """
        for key in self.cr_stats[0].keys():
            self.write_statistic(key)

class DataReader(object):

    def __init__(self, instr):
        """

        Parameters
        ----------
        instr
        """

        self._instr = instr
        self._mod_dir = os.path.dirname(os.path.abspath(__file__))
        self._base = os.path.join('/', *self._mod_dir.split('/')[:-2])
        self._msg_div = '-' * 79


if __name__ == '__main__':
    pass
