import argparse
from collections import defaultdict
import glob
import json
import logging
import os
_MOD_DIR = os.path.dirname(os.path.abspath(__file__))
_BASE = os.path.join('/', *_MOD_DIR.split('/')[:-1])

import shutil
import sys
sys.path.append(os.path.join(_BASE, 'pipeline'))

import warnings
warnings.simplefilter("ignore")


from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage
from utils import initialize
from utils import datahandler as dh

_PLOT_DIR = os.path.join(_BASE, 'analyzing_cr_rejection', 'plots')
_RESULTS_DIR = os.path.join(_BASE, 
					'analyzing_cr_rejection', 
					'results',
					 'STIS'
					 )

logging.basicConfig(format='%(levelname)-4s '
                           '[%(module)s.%(funcName)s:%(lineno)d]'
                           ' %(message)s',
                    )
LOG = logging.getLogger('compare_results')
LOG.setLevel(logging.INFO)


def create_data_objects(flist):
	"""
	Parameters
	----------
	flist : TYPE
	    Description
	
	Returns
	-------
	TYPE
	    Description
	"""
	obj = dh.DataReader(instr='stis_ccd', statistic='incident_cr_rate')
	obj.hdf5_files = flist
	obj.read_cr_rate()
	return obj


def make_MEF(fname, hdf5file1=None, hdf5file2=None):
	"""
	Parameters
	----------
	fname : TYPE
	    Description
	hdf5file1 : None, optional
	    Description
	hdf5file2 : None, optional
	    Description
	"""
	LOG.info(f'Creating a MEF for {fname}')
	dset = os.path.basename(fname)
	hdu_list = fits.HDUList()
	with fits.open(fname) as hdu:
		prhdr = hdu[0].header
		scihdr = hdu[1].header
		sci = hdu[1].data
		dqhdr = hdu[3].header
		dq = hdu[3].data
	prhdu = fits.PrimaryHDU(header=prhdr)
	hdu_list.append(fits.ImageHDU(header=scihdr, data=sci))
	hdu_list.append(fits.ImageHDU(header=dqhdr, data=dq))
	if hdf5file1 is not None:
		label1, metadata1 = label_from_file(
			hdf5file=hdf5file1, 
			dset_name=dset, 
			shape=sci.shape
		)
		hdr1 = fits.Header(cards=[], copy=False)
		params = '_'.join(os.path.basename(hdf5file1).split('.hdf5')[0].split('_')[-3:])
		hdr1.fromkeys(metadata1)
		hdr1['EXTNAME'] = 'CRLABEL'
		hdu_list.append(fits.ImageHDU(header=hdr1, data=label1))
	if hdf5file2 is not None:
		label2, metadata2 = label_from_file(
			hdf5file=hdf5file2,
			dset_name=dset,
			shape=sci.shape
		)
	
		hdr2 = fits.Header(cards=[], copy=False)
		params = '_'.join(os.path.basename(hdf5file2).split('.hdf5')[0].split('_')[-3:])
		hdr2.fromkeys(metadata2)
		hdr2['EXTNAME'] = 'CRLABEL'
		hdu_list.append(fits.ImageHDU(header=hdr2, data=label2))
	LOG.info(f"{hdu_list.info()}")
	hdu_list.writeto(f"{fname.replace('_flt.fits', '_all.fits')}", overwrite=True)
	

def label_from_file(hdf5file, dset_name, shape=None):
	"""
	Parameters
	----------
	hdf5file : TYPE
	    Description
	dset_name : TYPE
	    Description
	shape : None, optional
	    Description
	
	Returns
	-------
	TYPE
	    Description
	"""
	dh1 = dh.DataReader(instr='stis_ccd', statistic='cr_affected_pixels')
	cr_affected_pixels, metadata = dh1.read_single_dst(hdf5file, dset_name)
	template = np.zeros(shape)
	for (y,x) in cr_affected_pixels:
		template[int(y)][int(x)] +=1
	label, num_feat = ndimage.label(template,
                                        structure=np.ones((3,3)))
	return label, metadata


def examine_label(dirname=_RESULTS_DIR, exptime=60.0):
	"""
	Parameters
	----------
	dirname : TYPE, optional
	    Description
	exptime : float, optional
	    Description
	"""
	flist = glob.glob(f"{dirname}/stis*cr_affected_pixels*hdf5")
	file1 = display_menu(flist)
	params1 = '_'.join(os.path.basename(file1).split('.hdf5')[0].split('_')[-2:])
	dir1 = os.path.join(
		_BASE,'analyzing_cr_rejection',
			f"{exptime}_{params1}"
	)
	print(dir1)
	dataset1 = glob.glob(dir1+'/*flt.fits')
	file2 = display_menu(flist)
	params2 = '_'.join(os.path.basename(file2).split('.hdf5')[0].split('_')[-2:])
	dir2 = os.path.join(
		_BASE, 'analyzing_cr_rejection', f"{exptime}_{params2}")
	print(dir2)
	dataset2 = glob.glob(dir2+'/*flt.fits')
	print(len(dataset1), len(dataset2))

	for f1, f2 in zip(dataset1, dataset2):
		make_MEF(fname=f1, hdf5file1=file1)
		make_MEF(fname=f2, hdf5file2=file2)
		

def exptime_summary(dh, title=''):
	"""
	Parameters
	----------
	dh : TYPE
	    Description
	title : str, optional
	    Description
	"""
	counts = dh.data_df.integration_time.value_counts()
	fig, ax = plt.subplots(nrows=1, ncols=1)
	counts.plot.barh(ax=ax)
	ax.set_title(title)
	ax.set_ylabel('Integration Time [seconds]')
	plt.show()


def compare_by_exptime(dh, title=''):
	"""
	Parameters
	----------
	dh : TYPE
	    Description
	title : str, optional
	    Description
	"""
	longexp = dh.data_df.integration_time.gt(1000)
	shortexp = dh.data_df.integration_time.lt(100)

	fig, ax = plt.subplots(nrows=1, ncols=1)
	fig.autofmt_xdate()
	ax.scatter(dh.data_df[longexp].index,
	 dh.data_df[longexp].incident_cr_rate, label='longexp')
	ax.scatter(dh.data_df[shortexp].index, 
		dh.data_df[shortexp].incident_cr_rate, label='shortexp')
	ax.legend(loc='best')
	ax.set_title(title)
	plt.show()


def compare_by_rej_params(
	dh1, 
	dh2,
	label1='',
	label2='',
	title='',
	fout=None,
	figsize=(6,5)
):
	"""
	Parameters
	----------
	dh1 : TYPE
	    Description
	dh2 : TYPE
	    Description
	label1 : str, optional
	    Description
	label2 : str, optional
	    Description
	title : str, optional
	    Description
	fout : None, optional
	    Description
	figsize : tuple, optional
	    Description
	"""

	years = mdates.YearLocator()   # every year
	months = mdates.MonthLocator()  # every month
	days_major = mdates.DayLocator(interval=5)
	days_minor = mdates.DayLocator(interval=1)
	years_fmt = mdates.DateFormatter('%Y-%m-%d')


	expcut = dh1.data_df.integration_time.gt(1000)
	shortexp = dh1.data_df.integration_time.lt(100)

	fig, (ax1, ax2) = plt.subplots(
		nrows=1,
		ncols=2, 
		figsize=figsize, 
		sharex=True
		)

	diff = dh2.data_df[expcut].incident_cr_rate - \
			dh1.data_df[expcut].incident_cr_rate
	dates = dh1.data_df[expcut].index.values
	rate1 = dh1.data_df[expcut].incident_cr_rate
	rate2 = dh2.data_df[expcut].incident_cr_rate
	fig.autofmt_xdate()
	# ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
	# ax2.xaxis.set_major_locator(plt.MaxNLocator(10))
	ax1.scatter(dh1.data_df[expcut].index.values, 
		rate1, label=label1)
	ax1.scatter(dh2.data_df[expcut].index.values,
		rate2, label=label2)
	ax1.set_ylabel('CR Rate [CR/cm$^2$/second]')
	ax1.legend(loc='best')
	ax2.scatter(diff.index.values, diff, c='k')
	ax2.set_title(f'{label2} - {label1}')
	ax1.set_xlim(
		(Time('2019-05-25', format='iso').to_datetime(),
		Time('2019-07-01', format='iso').to_datetime())
		)
	# ax1.fmt_xdata = mdates.DateFormatter('%m-%d')
	# ax2.fmt_xdata = mdates.DateFormatter('%m-%d')
	ax1.xaxis.set_major_locator(days_major)
	ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
	ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
	ax1.xaxis.set_minor_locator(days_minor)

	ax2.xaxis.set_major_locator(days_major)
	ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
	ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
	ax2.xaxis.set_minor_locator(days_minor)

	fig.suptitle(title)
	if fout is not None:
		fout = os.path.join(_PLOT_DIR, fout)
		fig.savefig(fout, format='png', dpi=300, bbox_inches='tight')
	plt.show()


def get_default_parameters(dh):
	"""
	Parameters
	----------
	dh : TYPE
	    Description
	"""
	expsum = dh.data_df.integration_time.sum()
	n_images = len(dh.data_df)
	tb = Table.read('/Users/nmiles/hst_cosmic_rays/j3m1403io_crr.fits')


def display_menu(flist):
	# Generate a list of options for the user to choose from
	out_str = 'Choose a dataset to analyze:\n'
	for i, f in enumerate(flist):
		out_str += f"{i}) {os.path.basename(f)}\n"
	LOG.info(f"{out_str}\n{'-'*79}")
	idx = int(input('Enter selection: '))
	file = flist[idx]
	LOG.info(f"Selected option: {file}")
	return file


def run_comparison(fout=None):
	"""
	Parameters
	----------
	fout : None, optional
	    Description
	"""
	results_dir = os.path.join(_BASE, 
					'analyzing_cr_rejection', 
					'results',
					 'STIS'
					 )
	# Get a list of all the files generated with median combination
	medflist = glob.glob(
		os.path.join(_RESULTS_DIR,'stis*cr_rate*med*hdf5')
		)
	# Get a list of all the files generated with minimum combination
	minflist = glob.glob(
		os.path.join(_RESULTS_DIR, 'stis*cr_rate*min*hdf5')
		)
	flist = medflist + minflist

	file1 = display_menu(flist)
	file2 = display_menu(flist)
	
	# Extract the CR parameters from the filename
	file1_rej_param = file1.replace('.hdf5','')
	file1_rej_param = ' '.join(file1_rej_param.split('_')[-2:])

	# Extract the CR parameters from the filename
	file2_rej_param = file2.replace('.hdf5','')
	file2_rej_param = ' '.join(file2_rej_param.split('_')[-2:])
	

	dh_1 = create_data_objects([file1])
	dh_2 = create_data_objects([file2])

	# exptime_summary(dh_min, title='Exposure Times')
	# compare_by_exptime(dh_min, title=min_rej_param)
	# compare_by_exptime(dh_med, title=med_rej_param)

	compare_by_rej_params(dh_1, dh_2,
	 label1=file1_rej_param, label2=file2_rej_param,
	 title=f'{file1_rej_param} vs. {file2_rej_param}',
	 fout=f'{file1_rej_param}_{file2_rej_param}.png', figsize=(6.5,4.5))

	examine_label()






