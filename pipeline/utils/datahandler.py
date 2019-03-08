#!/usr/bin/env python
"""
This module contains two classes for reading and writing the data generated
by the pipeline.
"""

from collections import defaultdict, Iterable
import glob
import logging
import os

from astropy.time import Time
import dask.array as da
import h5py
import numpy as np
import pandas as pd
import sunpy
import sunpy.timeseries
import sunpy.data.sample
import yaml

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
                                          dtype=np.float32)

                for (key, val) in file_info.metadata.items():
                    # Check the datatype and save it accordingly
                    if isinstance(val, np.ndarray):
                        dset.attrs.create(name=key,
                                          data=val,
                                          shape=val.shape,
                                          dtype=np.float32)

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

#TODO: finish data reader. Need to figure out an efficient way to do this
class DataReader(object):

    def __init__(self, instr, statistic, cfg=None):
        """

        Parameters
        ----------
        instr
        """

        self._instr = instr.upper()
        self._statistic = statistic
        self._mod_dir = os.path.dirname(os.path.abspath(__file__))
        self._base = os.path.join('/', *self._mod_dir.split('/')[:-2])

        self._cfg_file = os.path.join(self._base,
                                      'CONFIG',
                                      'pipeline_config.yaml')
        self._hdf5_files = None
        self._energy_deposited = None
        self._incident_cr_rate = None
        self._size_sigmas = None
        self._size_pixels = None
        self._shape = None
        self._pixels_affected = None
        self._metadata = None

        if cfg is None:
            # Load the CONFIG file
            with open(self._cfg_file, 'r') as fobj:
                self._cfg = yaml.load(fobj)

        self._instr_cfg = self.cfg[self._instr]

        self._msg_div = '-' * 79

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
    def data(self):
        return

    @property
    def cfg(self):
        return self._cfg

    @cfg.getter
    def cfg(self):
        """Configuration object returned by parsing the
        :py:attr:`~pipeline_updated.CosmicRayPipeline.cfg_file`"""
        return self._cfg

    @property
    def instr(self):
        return self._instr

    @instr.getter
    def instr(self):
        """One of the valid instrument names"""
        return self._instr

    @property
    def instr_cfg(self):
        return self._instr_cfg

    @instr_cfg.getter
    def instr_cfg(self):
        """Instrument specific configuration"""
        return self._instr_cfg

    @property
    def hdf5_files(self):
        return self._hdf5_files

    @hdf5_files.setter
    def hdf5_files(self, value):
        self._hdf5_files = value

    @property
    def statistic(self):
        """Statistic to be read in"""
        return self._statistic

    def find_hdf5(self):
        """ Find the HDF5 files for the given py:attr:`statistic`

        Returns
        -------

        """
        rel_path = self.instr_cfg['hdf5_files'][self.statistic]
        full_path = os.path.join(self.base, *rel_path.split('/'))
        hdf5_files = glob.glob(full_path.replace('.hdf5', '*'))
        msg = (
            'Found the following data files\n {} \n{}'.format(
                '\n'.join(hdf5_files), self._msg_div)
        )
        LOG.info(msg)
        self.hdf5_files = hdf5_files

    def read_cr_stat(self, fill_value=-999, units=None):
        tmp = []
        for f in self.hdf5_files:
            fobj = h5py.File(f, mode='r')
            grp = fobj[self.statistic]
            for name in grp.keys():
                dset = grp[name]
                if not units:
                    tmp.append(da.from_array(dset, chunks=(15000)))
                elif units == 'sigmas':
                    tmp.append(da.from_array(dset[:][0], chunks=(15000)))
                else:
                    tmp.append(da.from_array(dset[:][1], chunks=(15000)))
        x = da.concatenate(tmp, axis=0)
        # Remove an NaN's and replace them with the fill value
        data = da.ma.fix_invalid(x, fill_value=fill_value)
        LOG.info('Final array {}'.format(x.shape))

        # Set the proper attribute based on the statistic
        if self.statistic == 'energy_deposited':
            self._energy_deposited = data
        elif self.statistic == 'sizes':
            if units == 'pixels':
                self._size_pixels = data
            elif units == 'sigmas':
                self._size_sigmas = data
            else:
                LOG.info(
                    'Please provide the '
                    'units for cosmic ray {}'.format(self.statistic)
                )
        elif self.statistic == 'shapes':
            self._shape = data




    def read_cr_rate(self):
        data = defaultdict(list)
        metadata = defaultdict(list)
        for f in self.hdf5_files:
            fobj = h5py.File(f, mode='r')
            grp = fobj[self.statistic]
            for name in grp.keys():
                data['obsname'].append(name)

                # Get the data for the current dataset
                dset = grp[name]

                # Record the data for the given statistic
                if isinstance(dset.value, Iterable):
                    data[self.statistic] += list(dset.value)
                else:
                    data[self.statistic].append(dset.value)
                # Get the attributes stored with the data
                attrs = dset.attrs
                for key in attrs.keys():
                    val = attrs[key]
                    if key == 'date':
                        val = Time(val, format='iso')
                        data[key].append(val)
                    elif key in ['altitude','latitude','longitude']:
                        try:
                            data['{}_start'.format(key)].append(val[0])
                            data['{}_end'.format(key)].append(val[-1])
                        except IndexError:
                            data[key].append(val)
                    else:
                        data[key].append(val)
        data['mjd'] = [val.mjd for val in data['date']]
        date_index = pd.DatetimeIndex([val.iso for val in data['date']])
        self.data_df = pd.DataFrame(data, index = date_index)
        self.data_df.sort_index(inplace=True)

    def plot_solar_cycle(self, variable=None, ax = None, smoothed=False):
        """ Retrieve solar cycle information

        Parameters
        ----------
        variable
        ax
        smoothed

        Returns
        -------

        """
        noaa = sunpy.timeseries.TimeSeries(sunpy.data.sample.NOAAINDICES_TIMESERIES,
                                           source='NOAAIndices')


        if variable is None and ax is not None:
            noaa.peek(type='sunspot RI', ax=ax)
        elif ax is not None:
            noaa.peek(type=variable, ax=ax)
        return noaa


def main():
    d = DataReader(instr='STIS_CCD', statistic='incident_cr_rate')
    d.find_hdf5()


if __name__ == '__main__':
    main()
