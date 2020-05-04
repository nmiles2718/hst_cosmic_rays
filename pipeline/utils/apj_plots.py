#!/usr/bin/env python

from collections import defaultdict
from collections import Iterable
import datetime as dt
import logging
from itertools import chain
from datetime import timedelta
import glob
import os
from astropy.stats import sigma_clip, sigma_clipped_stats, gaussian_sigma_to_fwhm
from astropy.timeseries import LombScargle
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.visualization import LinearStretch, ZScaleInterval,\
    LogStretch, SqrtStretch, ImageNormalize
import costools
import datahandler as dh
import sys
sys.path.append('/Users/nmiles/hst_cosmic_rays/pipeline/')
import dask.array as da
from process import process
from label import labeler

import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.6
import dask.array as da
from matplotlib import rc
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib import ticker
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
# mpl.use('qt5agg')
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec
plt.style.use('ggplot')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import  inset_axes, zoomed_inset_axes
import matplotlib.patches as patches

import numpy as np
import pandas as pd

import sunpy.net
from sunpy.timeseries import TimeSeries
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import find_peaks

# Local packages
import datahandler as dh
import visualize
import metadata



COSMICRAY_ML = '/Users/nmiles/hst_cosmic_rays/cosmicrayml/'
APJ_PLOT_DIR = '/Users/nmiles/hst_cosmic_rays/APJ_plots/'


def plot_cosmicrayml_pointins(data_dict, vmins=[0,0], vmaxes=[1000, 1000]):
    norms = {}
    wcs_info = {}
    for key, vmin, vmax in zip(data_dict.keys(), vmins, vmaxes):
        norm = ImageNormalize(
            data_dict[key].data,
            stretch=LogStretch(),
            vmin=vmin,
            vmax=vmax
        )
        norms[key] = norm
        img_wcs = WCS(data_dict[key].header, data_dict[key])
        wcs_info[key] = img_wcs

    fig = plt.Figure()
    keys = list(wcs_info.keys())
    ax = fig.add_subplot(1,2,1, projection=wcs_info[keys[0]])
    ax1 = fig.add_subplot(1,2,2, projection=wcs_info[keys[1]])

    for key, ax in zip(keys,[ax,ax1]):
        ax.imhow(data_dict[key].data, norm=norms[key], origin='lower', cmap='gray')




def get_cosmicrayml_pointings(
        dirname='/Users/nmiles/hst_cosmic_rays/cosmicrayml/stellar_data/N44-CENTER'
):
    flist = glob.glob(f"{dirname}/pointing?/combined*drc_sci.fits")
    data_dict = {}

    for f in flist:
        data_dict[f"{os.path.basename(f)}"] = fits.open(f)


    return data_dict


def read_data(stat='energy_deposited',min_exptime=50, units=None):
    reader_wfc = dh.DataReader(instr='ACS_WFC', statistic=stat)
    reader_hrc = dh.DataReader(instr='ACS_HRC', statistic=stat)
    reader_wfc3 = dh.DataReader(instr='WFC3_UVIS', statistic=stat)
    reader_wfpc2 = dh.DataReader(instr='WFPC2', statistic=stat)
    reader_stis = dh.DataReader(instr='STIS_CCD', statistic=stat)

    for r in [reader_wfpc2, reader_wfc, reader_hrc, reader_wfc3]:
        r.find_hdf5()

    if 'rate' in stat:
        # flist = glob.glob(
        #     '../../results/STIS_crrejtab_CRSIGMAS/*cr_rate*hdf5'
        # )
        # reader_stis.hdf5_files = flist
        for r in [reader_wfpc2, reader_stis, reader_wfc, reader_hrc, reader_wfc3]:
            r.read_cr_rate()
    else:
        for r in [reader_wfpc2,reader_stis, reader_wfc, reader_hrc, reader_wfc3]:
            r.read_cr_stat(units=units, min_exptime=min_exptime)
    return reader_hrc, reader_stis, reader_wfc, reader_wfpc2, reader_wfc3

def get_solar_min_and_max(noaa_data):
    solar_cycle = {'Cycle 23': None, 'Cycle 24':None}
    min_1996 = noaa_data['1993-01-01':'1997-01-01'].idxmin()
    max_2000 = noaa_data['1998-01-01':'2004-01-01'].idxmax()
    min_2009 = noaa_data['2008-01-01':'2011-01-01'].idxmin()
    max_2014 = noaa_data['2011-01-01':'2017-01-01'].idxmax()

    solar_cycle['Cycle 23'] = (min_1996['sunspot RI smooth'],
                                    max_2000['sunspot RI smooth'])

    solar_cycle['Cycle 24'] = (min_2009['sunspot RI smooth'],
                                    max_2014['sunspot RI smooth'])
    return solar_cycle


def draw_map(scale=0.25, ax=None):
    mapbase = Basemap(projection='cyl', ax=ax)
    # Draw the meridia
    mapbase.shadedrelief(scale=scale)
    # lats and longs are returned as a dictionary
    lats = mapbase.drawparallels(np.linspace(-90, 90, 13),
                                 labels=[True, False, False, False],
                                 fontsize=8)

    lons = mapbase.drawmeridians(np.linspace(-180, 180, 13),
                                 labels=[False, False, False, True],
                                 fontsize=8)

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')

    return mapbase


def hst_loc_per_observation(
        fname = os.path.join(APJ_PLOT_DIR, 'odbrf7ggq_flt.fits'),
        figsize=(6,5),
        instr='STIS_CCD',
        exp1=None,
        exp2=None
):
    observation_exp1 = metadata.GenerateMetadata(
        fname=fname,
        instr=instr
    )
    observation_exp1.get_image_data()
    if exp1 is None:
        exp1 = observation_exp1.metadata['integration_time']
        observation_exp1.get_observatory_info(time_delta=exp1)
    else:
        observation_exp1.get_observatory_info(time_delta=exp1)

    observation_exp2 = metadata.GenerateMetadata(
        fname=fname,
        instr=instr
    )
    observation_exp2.get_image_data()
    if exp2 is None:
        exp2 = observation_exp2.metadata['integration_time']
        observation_exp2.get_observatory_info(time_delta=exp2)
    else:
        observation_exp2.get_observatory_info(time_delta=exp2)

    return observation_exp1, observation_exp2


def hst_loc_plot(hrc, stis, wfc, wfpc2, uvis,cartopy=False, orbital_path1=None, orbital_path2=None):
    v = visualize.Visualizer()
    combined_data = {'latitude_start':[],'longitude_start':[], 'incident_cr_rate':[],
                     'integration_time':[],'instr':[],'date':[],'mjd':[]}
    
    instrument_name=['ACS/HRC','STIS/CCD','ACS/WFC','WFPC2','UVIS']
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                          '#f781bf', '#a65628', '#984ea3',
                          '#999999', '#e41a1c', '#dede00']
    for i,df in enumerate([hrc, stis, wfc, wfpc2, uvis]):
        lat = list(df['latitude_start'])
        lon = list(df['longitude_start'])
        instr = [instrument_name[i]]*len(lat)
    #    print(len(instr))
        cr_rate = list(df['incident_cr_rate'])
        combined_data['instr'] += instr
        combined_data['latitude_start'] += lat
        combined_data['longitude_start'] += lon
        combined_data['incident_cr_rate'] += cr_rate
        combined_data['integration_time'] += list(df['integration_time'])
        combined_data['date'] += list(df.index)
        combined_data['mjd'] += list(df.mjd)
    for key in combined_data.keys():
        print(key, len(combined_data[key]))
    df_combined = pd.DataFrame(combined_data)
    df_combined_cut = df_combined[df_combined.incident_cr_rate.gt(0.4)] 
    df_combined.to_csv('combined_hst_cr_vs_loc_data.txt', header=True, index=False)
    if cartopy:
        fig = v.plot_hst_loc_cartopy(df=df_combined, key='start', orbital_path1=orbital_path1,
                             orbital_path2=orbital_path2)
    else:
        fig = v.plot_hst_loc(df=df_combined, key='start', orbital_path1=orbital_path1,
                            orbital_path2=orbital_path2)
    fout = os.path.join(APJ_PLOT_DIR, 'cr_rate_vs_location_allinstr_test.png')
    fig.savefig(fout, format='png', dpi=350, bbox_inches='tight')
    return df_combined_cut

def combine_integration_time_info(hrc, stis, wfc, wfpc2, uvis):
    instrument_name=['STIS/CCD','ACS/HRC','ACS/WFC','WFPC2','WFC3/UVIS']
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                          '#f781bf', '#a65628', '#984ea3',
                          '#999999', '#e41a1c', '#dede00']
    data_dict = {}
    all_data_dict = {'instr':[], 'integration_time':[],'date':[]}
    for i,df in enumerate([stis, hrc, wfc, wfpc2, uvis]):
        counts = df['integration_time'].value_counts()
        data_dict[instrument_name[i]] = counts
        all_data_dict['instr'] += [instrument_name[i]]*len(df)
        all_data_dict['integration_time'] += list(df['integration_time'])
        all_data_dict['date'] += list(df.index.values)
    date_time_index = pd.DatetimeIndex(all_data_dict['date'])
    df = pd.DataFrame(data_dict)
    all_df = pd.DataFrame(all_data_dict, index=date_time_index)
    return df, all_df


def plot_morph(
        data,
        bins,
        drange,
        figsize=(6,5),
        xlabel=None,
        ylabel=None,
        title=None,
        ax=None,
        color=None,
        label=None,
        normalize=True,
        logx=False,
        logy=True,
        lw=1.75,
        ls='-',
        use_fwhm=False
):
    v = visualize.Visualizer()
    if use_fwhm:
        data = data * gaussian_sigma_to_fwhm

    fig, ax, hist, bins = v.plot_hist(
        data,
        bins=bins,
        c=color,
        range=drange,
        ax=ax,
        logy=logy,
        label=label,
        normalize=normalize
    )

    return fig, ax, hist, bins



def plot_exptime_counts(integration_df, combined_df, N=20, logy=True, logx=True, add_title=True):
    expcut = combined_df.integration_time.gt(800)
    counts = combined_df['integration_time'][expcut].value_counts()
    total = integration_df.sum().sum()
    print(integration_df.sum())
    #counts.sort_index(inplace=True)
    counts_norm = integration_df/total
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                          '#f781bf', '#a65628', '#984ea3',
                          '#999999', '#e41a1c', '#dede00']

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5))
    # Only show plots for flashdurs that have more than 30 observations
    cut = counts.iloc[:N]
    cut_norm =  counts_norm.iloc[:N]
    match_cut = integration_df.loc[cut.index,:]
    instr_sum = match_cut.sum()    
    total = match_cut.sum().sum()
    percentages=100*instr_sum/total
    new_labels = [f'{instr} ({val:.3f}%)' for instr, val in zip(percentages.index.values, percentages)]
    print(new_labels)
    match_cut.sort_index(inplace=True)
    barh = match_cut.plot.barh(ax=ax, logy=logy, logx=logx,color=CB_color_cycle,rot=0,stacked=True)
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())
    barh_legend = plt.legend(bbox_to_anchor=(1.0001, 1),
                            loc=2,
                            ncol=1, edgecolor='k')
    handles, labels = ax.get_legend_handles_labels()
    for label in labels:
        print(match_cut[label].sum())
    barh_legend = plt.legend(handles, new_labels,
                            bbox_to_anchor=(1.0001,1),
                            loc=2,
                            ncol=1, edgecolor='k')
    ax.set_ylabel('Integration Time [s]')
    if add_title:
        ax.set_title(f'Top {N:.0f} Integration Times')
    fout = os.path.join(APJ_PLOT_DIR, 'exptime_comp_plot.png')
    fig.savefig(fout, format='png', dpi=350, bbox_inches='tight')
    return counts, cut


def compute_basic_stats(hrc, stis, wfc, wfpc2, uvis, use_dask=False):
    labels=['ACS/HRC', 'STIS/CCD', 'ACS/WFC', 'WFPC2', 'WFC3/UVIS']
    for dset, label in zip([hrc, stis, wfc, wfpc2, uvis],labels):
      #  avg = da.nanmean(dset, axis=0).compute()
       # std = da.nanstd(dset, axis=0).compute()
        #print(f'{label}: {avg}+\-{std}\n')
        if use_dask:
            quantiles = da.percentile(dset, q=[25, 50, 75], interpolation='linear').compute()
        else:
            quantiles = np.percentile(dset, q=[25, 50, 75], interpolation='linear')
        print('Percentiles for {}'.format(label))
        for val, q in zip(quantiles, [25, 50, 75]):
            print('{:.0f} {:.4f}'.format(q, val))


def plot_morphology(
        hrc,
        stis,
        wfc,
        wfpc2,
        uvis,
        bins,
        drange,
        figsize=(6,5),
        xlabel=None,
        ylabel=None,
        title=None,
        ax=None,
        color=None,
        label=None,
        normalize=False,
        logx=False,
        logy=True,
        lw=1.75,
        ls='-',
        xlim=None,
        ylim=None,
        fout=None
):

    instrument_name=['STIS/CCD','ACS/HRC','ACS/WFC','WFPC2','WFC3/UVIS']
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                            '#f781bf', '#a65628', '#984ea3',
                            '#999999', '#e41a1c', '#dede00']
    v = visualize.Visualizer()
    for i, data in enumerate([stis, hrc, wfc, wfpc2, uvis]):
        print('Analyzing distribution for {}'.format(instrument_name[i]))
        if i == 0: 
            fig, ax, hist, bins = v.plot_hist(
                data,
                bins=bins,
                c=CB_color_cycle[i],
                range=drange,
                logy=logy,
                logx=logx,
                label=instrument_name[i],
                normalize=normalize
            )
        else:
            fig, ax, hist, bins = v.plot_hist(
                  data,
                  bins=bins,
                  c=CB_color_cycle[i],
                  range=drange,
                  ax=ax,
                  logx=logx,
                  logy=logy,
                  label=instrument_name[i],
                  normalize=normalize
             )
    ax.tick_params(axis='both', which='major', width=1.5, length=5)
    ax.legend(loc='best',edgecolor='k')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylabel('Normalized Bin Count')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if fout is not None:
        fig.savefig(os.path.join(APJ_PLOT_DIR, fout), format='png', dpi=300)
    plt.show()
    return fig, ax, hist, bins


def orbital_altitude(hrc, stis, wfc, wfpc2, uvis):
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                           '#f781bf', '#a65628', '#984ea3',
                           '#999999', '#e41a1c', '#dede00']
    labels = ['STIS/CCD','ACS/HRC', 'ACS/WFC', 'WFPC2', 'WFC3/UVIS']
    datasets = [stis, hrc, wfc, wfpc2, uvis]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7, 3.5))
    for dset, label, color in zip(datasets,labels, CB_color_cycle):
        altitude_df = dset.loc[:, ['mjd', 'altitude_start']]
        averaged = altitude_df.resample(rule='5D').mean()
        ax.scatter(averaged.index, averaged['altitude_start'], 
                    s=2.25,
                    alpha=0.6,
                    color=CB_color_cycle[0])
    sm2 = Time('1995-01-13', format='iso').to_datetime()
    sm2_line = Time('1997-02-11',format='iso').to_datetime()
    sm3b_line = Time('2002-03-01', format='iso').to_datetime()
    sm3b = Time('2002-04-01', format='iso').to_datetime()
    sm4_line = Time('2009-05-11', format='iso').to_datetime()
    sm4 = Time('2009-06-01',format='iso').to_datetime()
    ax.text(x=sm2, y=605, s='SM2',fontsize=11)
    ax.text(x=sm3b, y=582, s='SM3B', fontsize=11)
    ax.text(x=sm4, y=570, s='SM4', fontsize=11)
    ax.axvline(sm2_line, ls='--', c='k',lw=1.)
    ax.axvline(sm3b_line, ls='--', c='k',lw=1.)
    ax.axvline(sm4_line, ls='--', c='k',lw=1.)
    ax.set_xlabel('Date',fontsize=11, color='k')
    ax.set_ylabel('Orbital Altitude [km]',fontsize=11, color='k')
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    ax.tick_params(axis='both',labelsize=11, which='both', color='k', labelcolor='k')
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)
        tick.set_ha('right')
    #ax.set_title('Orbital Decay of HST')
    fig.savefig(os.path.join(APJ_PLOT_DIR,'orbital_decay.png'), format='png', dpi=200, bbox_inches='tight')
    plt.show()


def rate_hist(hrc, stis, wfc, wfpc2, uvis):
    #fig = plt.figure(figsize=(5,3.5))
    #ax = fig.add_subplot(1,1,1)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(6,3))
    ax = axes[0]
    ax1 = axes[1]
    #gs0 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    #gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=6,wspace=0.5, subplot_spec=gs0[0])
    #gs10 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=6,wspace=0.5, subplot_spec=gs0[1])
    #ax1 = fig.add_subplot(gs00[0,1:3]) 
    #ax2 = fig.add_subplot(gs00[0,3:5])  
    #ax3 = fig.add_subplot(gs10[0,:2])
    #ax4 = fig.add_subplot(gs10[0,2:4])
    #ax5 = fig.add_subplot(gs10[0,4:6])      
    #axes = [ax1, ax2, ax3, ax4, ax5]
    #master_ax = fig.add_subplot(111,frameon=False)
    #master_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #master_ax.grid(False)
    #fig, axes = plt.subplots(nrows=, ncols=1, figsize=(6,6))
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                          '#f781bf', '#a65628', '#984ea3',
                          '#999999', '#e41a1c', '#dede00']
    labels = ['STIS/CCD','ACS/HRC', 'ACS/WFC', 'WFPC2', 'WFC3/UVIS']
    datasets = [stis, hrc, wfc, wfpc2, uvis]
    for i, (label, dset) in enumerate(zip(labels, datasets)):
        flags = dset.incident_cr_rate.gt(0.2) 
        cr_rate = dset['incident_cr_rate'][flags]
        
        results =  sigma_clip(cr_rate, masked=True, return_bounds=True, sigma=3, maxiters=5)
        masked_array = results[0]
        bounds = results[1]
        mean = masked_array.mean()
        mean_std = masked_array.std()/masked_array.count()
        median = cr_rate.quantile(0.5)
        lower_20 = cr_rate.quantile(0.25)
        upper_80 = cr_rate.quantile(0.75)
        print(f'Number of excluded images: {len(cr_rate) - masked_array.count()}')
        msg = (
            f"mean: {mean:.6f}\nmedian: {median:6f}\nstd: {mean_std:e}\n25%: {lower_20:.6f}\n75%: {upper_80:.6f}"
        )
        print(label+f'\n{masked_array.count()}'+'\n'+msg+'\n')
        print('-'*60)
        hist, edges = np.histogram(cr_rate, bins=60, range=(0, 3))
    #    if i == 0:
    #        edges_avg = 0.5 * (edges[:-1] + edges[1:])
    #        master_origin = edges_avg[np.argmax(hist)]
    #        ax1.step(0.5*(edges[:-1] + edges[1:]), hist/np.max(hist), label=label, color=CB_color_cycle[i])
     #   else:
        edges_avg = 0.5*(edges[:-1] + edges[1:])
        shift = edges_avg[np.argmax(hist)]
        print(shift)
        ax1.step(0.5*(edges[:-1] + edges[1:]) - shift, hist/np.max(hist), label=label, color=CB_color_cycle[i]) 
        ax.step(0.5*(edges[:-1] + edges[1:]), hist/np.max(hist), label=label, color=CB_color_cycle[i])
        #ax.set_xticks([0,0.5, 1.0, 1.5, 2, 2.5, 3])
        #ax.axvline(median, c='k', ls='-', lw=0.75, )
        #ax.axvspan(lower_20, upper_80, color='gray', alpha=0.3)
        #ax1.axis('off')
        #ax1.tick_params(labelleft=False,labelbottom=False)  
        
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax.xaxis.set_major_locator(MultipleLocator(0.5))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.yaxis.set_minor_locator(MultipleLocator(0.05))
        ax1.xaxis.set_minor_locator(MultipleLocator(0.1))
        ax1.xaxis.set_major_locator(MultipleLocator(0.5))
        ax1.yaxis.set_major_locator(MultipleLocator(0.2))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
        

        #ax.axvline(lower_20, c='k', ls='--', lw=0.75)
        #ax.axvline(upper_80, c='k', ls='--', lw=0.75)
        leg = ax.legend(loc='best', edgecolor='k', fontsize=8)
        ax.set_xlim((0,3.0))
        ax1.set_xlim((0-shift, 3-shift))
        #ax1.set_xlim((-6, 6)) 
        ax.set_ylim((-0.03,1.02))
        ax.set_ylim((-0.03, 1.02))
         #ax.text(1.8, 0.62,'N='+'{:,}'.format(hist.sum()))
        t = ax.text(0,0,'', fontsize=14)
        #ax.tick_params(axis='x', which='major', width=1.5, length=5)
        ax.tick_params(axis='both', which='minor', width=1,length=2.5)
        ax.tick_params(axis='both', which='major', width=1.5, labelsize=8,length=5)
        ax1.tick_params(axis='both', which='minor', width=1, length=2.5)
        ax1.tick_params(axis='both', which='major', width=1.5,labelsize=8, length=5)
    
    fig.canvas.draw()
    labels = [item.get_text() for item in ax1.get_xticklabels()]
    print(labels)
    new_labels = []
    for label in labels:
        if label == '0.0':
            new_labels.append('Peak Bin')
        else:
            new_labels.append('')
    ax1.set_xticklabels(new_labels, ha='center')
        


    ax.set_xlabel('CR Flux [$CR/s/cm^2$]', fontsize=9)
    ax.set_ylabel('Normalized Bin Count', fontsize=9)
    #fig.text(0.38, 0.02, 'Cosmic Ray Rate [$CR/s/cm^2$]', fontproperties=t.get_font_properties())
    #fig.text(0.05, 0.63, 'Normalized Bin Count', rotation='vertical', fontproperties=t.get_font_properties())
    fout = os.path.join(APJ_PLOT_DIR, 'cr_rate_hist.png')
    fig.savefig(fout, format='png', dpi=350, bbox_inches='tight')
    plt.show()


def cr_rejection_algorithm(
        flist=None,
        x=2045,
        y=1061,
        box_w=10,
        box_h=10,
        figsize=(7,4.5),
        fout='example_of_cr_rejection_transparent_poster.png'
):
    if flist is None:
        flist = glob.glob(
            '/Users/nmiles/hst_cosmic_rays/APJ_plots/test_data/*flt.fits'
        )

    img_data = []
    pix_data = []
    cr_affected = []
    for f in flist[:12]:
        print(f)
        with fits.open(f) as hdu:
            data = hdu[1].data
            dq = hdu[3].data
            img_data.append(data)
            pix_data.append(data[y][x])
            if dq[y][x] > 8100:
                cr_affected.append(True)
            else:
                cr_affected.append(False)

    print(sum(cr_affected))
    # p = process.ProcessCCD(instr='ACS_WFC', flist=flist[:12])
    # p.sort()
    # p.cr_reject()

    fig = plt.figure(figsize=figsize)
    gs0 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, wspace=0.4)
    gs00 = gridspec.GridSpecFromSubplotSpec(
        nrows=4,
        ncols=3,
        wspace=0.15,
        hspace=0.45,
        subplot_spec=gs0[0]
    )
    # fig.suptitle(
    #     'Comparing Pixel Values Across 12 Dark Frames',
    #     fontweight='bold', color='white'
    # )
    scatter_ax = fig.add_subplot(gs0[0, 1])
    # scatter_ax.set_title(rf'Pixel Value at ({x},{y})')
    scatter_ax.scatter([i+1 for i in range(12)], pix_data, label='Pixel Value')
    scatter_ax.set_xticks([i+1 for i in range(12)])
    scatter_ax.set_xticklabels([str(i+1) for i in range(12)])
    yticks = ticker.MaxNLocator(11)
    scatter_ax.set_ylim((-50,1000))
    scatter_ax.yaxis.set_major_locator(yticks)
    # scatter_ax.set_yticks([100*i for i in range(9)])
    scatter_ax.axhline(np.median(pix_data),
                       ls='dashed',
                       c='k',
                       label=rf'Med: {np.median(pix_data):.2f} $e^-$')
    scatter_ax.set_xlabel('Image Number', color='k')
    scatter_ax.set_ylabel(r'Signal [$e^-$]', color='k')

    leg = scatter_ax.legend(loc='upper left', edgecolor='k', fontsize=8,
                            bbox_to_anchor=(1.02, 1.))
    # scatter_ax.grid(ls='-',color='k')
    # plt.setp(scatter_ax.spines.values(), color='k')
    # for text in leg.get_texts():
    #     plt.setp(text, color='w')
    # ax.set_ylim(0, noaa_df['sunspot RI'].max() + 30)
    scatter_ax.tick_params(which='both', axis='both', color='k',
                   labelcolor='k')
    img_subplots = [
        fig.add_subplot(gs00[i,j]) for i in range(4) for j in range(3)
    ]

    norm = ImageNormalize(
        img_data[0],
        stretch=LinearStretch(),
        vmin=0,
        vmax=85
    )
    mk_patch = lambda x, y: patches.Rectangle(
        (x-0.5,y-0.5), width=1, height=1, fill=False, color='r', lw=1.15
    )
    for i, ax in enumerate(img_subplots):
        ax.grid(False)
        ttl=ax.set_title(f"{i+1}", fontsize=11, color='k')
        ttl.set_position([.5, 0.98])
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        im = ax.imshow(img_data[i], norm=norm, cmap='gray', origin='lower')
        ax.set_xlim((x - box_w, x + box_w))
        ax.set_ylim((y - box_h, y + box_h))
        patch = mk_patch(x, y)
        ax.add_patch(patch)


    fig.savefig(os.path.join(APJ_PLOT_DIR, fout),
                format='png',
                dpi=300,
                bbox_inches='tight',
                transparent=False)

    plt.show()


def periodogram(hrc, stis, wfc, wfpc2, uvis):
    fig, ax = plt.subplots(nrows=1, ncols=1,  figsize=(6.5,5.5))
    #ax = fig.add_subplot(111)
    #gs0 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace=0.25)
    #gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=6,
    #                                        wspace=0.55, subplot_spec=gs0[0])
    #gs10 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=6,
    #                                        wspace=0.55, subplot_spec=gs0[1])
    #ax = fig.add_subplot(111)
    axins = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(1.05, .58, .5, .4),
                   bbox_transform=ax.transAxes, loc=2, borderpad=0)
    axins.tick_params(which='both',left=False, right=True, labelleft=False, labelright=True)
    axins1 = inset_axes(ax, width="100%", height="100%",
                   bbox_to_anchor=(1.05, 0.02, .5, .4),
                   bbox_transform=ax.transAxes, loc=2, borderpad=0)
    axins1.tick_params(which='both',left=False, right=True, labelleft=False, labelright=True)
    
    #axins1 = inset_axes(ax, width="50%", height="75%",
    #               bbox_to_anchor=(.1, .5, .6, .5),
    #               bbox_transform=ax.transAxes, loc=3)
    #axins2 = inset_axes(ax, width="50%", height="75%",
    #                 bbox_to_anchor=(.6, .2, .6, .5),
    #                 bbox_transform=ax.transAxes, loc=3)
    #ax2 = fig.add_subplot(gs00[0,3:5])
    #ax3 = fig.add_subplot(gs10[0,:2])
    #ax4 = fig.add_subplot(gs10[0,2:4])
    #ax5 = fig.add_subplot(gs10[0,4:6])
    #axes = [ax1, ax2, ax3, ax4, ax5]
    #master_ax = fig.add_subplot(111,frameon=False)
    #master_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    #master_ax.grid(False)
    #plt.setp(ax.spines.values(), color='k')
    #fig, axes = plt.subplots(nrows=, ncols=1, figsize=(6,6))
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                          '#f781bf', '#a65628', '#984ea3',
                          '#999999', '#e41a1c', '#dede00']
    labels = ['STIS/CCD','ACS/HRC', 'ACS/WFC', 'WFPC2', 'WFC3/UVIS']
    datasets = [stis, hrc, wfc, wfpc2, uvis]
    data_out = {}
    peak1 = 0
    peak2 = 0
    offset = 0.000

    factor = 0
    
    #ax.annotate(
    #        s='Peak 1',
    #        xy=(0.002,0.025),
    #        xytext=(0.004,0.04),
    #        color='k',
    #        fontsize=12,
    #        arrowprops={'arrowstyle':'simple'})
    ax.annotate(s='Peak 1', xy=(0.0002,0.027), xytext=(0.0002, 0.04),
                ha='center', va='bottom',color='k',fontsize=12,
                arrowprops=dict(arrowstyle='-[, widthB=1., lengthB=0.5', lw=2.0,color='k'))
    ax.annotate(s='Peak 2', xy=(0.0205,0.074), xytext=(0.0205, 0.085),
                ha='center', va='bottom',color='k',fontsize=12,
                arrowprops=dict(arrowstyle='-[, widthB=1., lengthB=0.5', lw=2.0,color='k'))
    for i, (label, dset) in enumerate(zip(labels, datasets)):
        #if i == 1:
        #    continue
        #if i == 4:
        #    continue
        flags = dset.integration_time.gt(300)
        df = dset[['mjd','incident_cr_rate']][flags]
        #smoothed = df.rolling(window='30D', min_periods=5).median()
        smoothed = df
        smoothed_no_nan = smoothed.dropna()
        ls = LombScargle(
             smoothed_no_nan['mjd'], smoothed_no_nan['incident_cr_rate'], normalization='standard'
        )
        freq = ls.autofrequency(samples_per_peak=10,maximum_frequency=1, minimum_frequency=1/(365.25 * 15))
        #freq, power = ls.autopower(samples_per_peak=10, minimum_frequency=0)
        power = ls.power(freq)
        if label=='ACS/HRC':
            peak_range_1 = np.where((freq < 0.01) & (freq > 0.0005))[0]
            peak_range_1a = np.where((freq < 0.01) & (freq > 0))[0]
            max_power_range_idx1a = np.argmax(power[peak_range_1a])
            max_power_idx1a_full = peak_range_1a[max_power_range_idx1a]
            max_power_freq_1a = freq[max_power_idx1a_full]
            data_out[f"{label}_Pa"] = 1/max_power_freq_1a
            data_out[f"{label}_fa"] = max_power_freq_1a
        else:
            peak_range_1 = np.where((freq < 0.01) & (freq > 0.0))[0]
        max_power_idx1_range = np.argmax(power[peak_range_1])
        max_power_idx1_full = peak_range_1[max_power_idx1_range]
        max_power_freq1 = freq[max_power_idx1_full]
        if label=='ACS/HRC':
            #axins.axvline(max_power_freq_1a + factor *offset, color=CB_color_cycle[i], ls='--', lw=1.1)
            pass
        #axins.axvline(max_power_freq1+factor*offset, color=CB_color_cycle[i], ls='--', lw=1.1)
        max_power_period1 = 1/max_power_freq1
        peak1 += ls.false_alarm_probability(power[max_power_idx1_full])
        data_out[f"{label}_P"] = max_power_period1
        data_out[f"{label}_f"] = max_power_freq1
        #data_out[f"{label}_df"] = smoothed_no_nan
        #data_out[f"{label}_t0"] = np.min(df.mjd.values)
        fap1 = ls.false_alarm_probability(power[max_power_idx1_full])
        print(max_power_period1/365,max_power_freq1, label, fap1, ls.false_alarm_level(fap1))
        peak_range_2 = np.where((freq < 0.03) & (freq>0.01))[0]
        max_power_idx2_range = np.argmax(power[peak_range_2])
        max_power_idx2_full = peak_range_2[max_power_idx2_range]
        max_power_freq2 = freq[max_power_idx2_full]
        max_power_period2 = 1/max_power_freq2
      
          
       # axins1.axvline(max_power_freq2+factor*offset, color=CB_color_cycle[i], ls='--', lw=1.1)
        
        peak2 += ls.false_alarm_probability(power[max_power_idx2_full])
        print(max_power_period2, label, max_power_freq2, ls.false_alarm_probability(power[max_power_idx2_full]))
        #data_out[label] = (freq, power)
        ax.plot(freq+factor*offset, power, color=CB_color_cycle[i], label=f'{label}', lw=1.5)
        axins.plot(freq+factor*offset, power, color=CB_color_cycle[i], label=f"{label}", lw=1.5)
        axins.set_xlim(0, 0.003)
        axins.set_ylim(0, 0.03)
        axins1.plot(freq+factor*offset, power, color=CB_color_cycle[i], label=f"{label}", lw=1.5)
        axins1.set_xlim(0.019, 0.023)
        axins1.set_ylim(0, 0.08)
        ax.set_xlim(-0.0025, 0.04)
        ax.set_ylim(0, 0.14)
        #xticks =[0, 0.01, 0.02, 0.03,  0.04]
        ax.xaxis.set_minor_locator(MultipleLocator(0.002))
        ax.yaxis.set_minor_locator(MultipleLocator(0.005))
        ax.yaxis.set_major_locator(MultipleLocator(0.02))
        ax.xaxis.set_major_locator(MultipleLocator(0.01))
        #ax.set_xticks(xticks)
        ax.tick_params(axis='both',labelsize=10, which='both', color='k', labelcolor='k')
        ax.tick_params(axis='both', which='minor', width=1, length=2.5)
        ax.tick_params(axis='both', which='major', width=1.5, length=5)
        #axins.xaxis.set_minor_locator(MultipleLocator(0.0002))
        #axins.yaxis.set_minor_locator(AutoMinorLocator(5))
        axins.set_title('Peak 1', fontsize=10)
        axins.set_xlabel('Frequency [cycles/day]', fontsize=8, color='k')
        axins.set_ylabel('Lomb-Scargle Power', fontsize=8, color='k')
        axins.yaxis.set_label_position('right')
        for a in [axins, axins1]:
            a.tick_params(axis='both',labelsize=8, which='both', color='k', labelcolor='k')
            a.tick_params(axis='both', which='minor', width=1, length=2.5)
            a.tick_params(axis='both', which='major', width=1.5, length=5)
            
        axins.xaxis.set_major_locator(MultipleLocator(0.001))
        axins.xaxis.set_minor_locator(MultipleLocator(0.0002))
        axins.yaxis.set_major_locator(MultipleLocator(0.01))
        axins.yaxis.set_minor_locator(MultipleLocator(0.002))
        axins1.set_title('Peak 2', fontsize=10)
        axins1.set_xlabel('Frequency [cycles/day]', fontsize=8, color='k')
        axins1.set_ylabel('Lomb-Scargle Power', fontsize=8, color='k')
        axins1.yaxis.set_label_position('right')
        axins1.xaxis.set_major_locator(MultipleLocator(0.001))
        axins1.xaxis.set_minor_locator(MultipleLocator(0.0002))
        axins1.yaxis.set_major_locator(MultipleLocator(0.02))
        axins1.yaxis.set_minor_locator(MultipleLocator(0.005))
        #axins.set_xticks(xticks)
        #ax.tick_params(axis='both',labelsize=10, which='both', color='k', labelcolor='k')
        #ax.tick_params(axis='both', which='minor', width=1, length=2.5)
        #ax.tick_params(axis='both', which='major', width=1.5, length=5)
        
        # ax.tick_params(axis='x', which='major', width=1.5, length=5)
        #ax.set_xticklabels(xticks, rotation=45, ha='right')
        #for tick in ax.get_xticklabels():
        #    tick.set_rotation(45)
        #for tick in ax.get_xticklabels():
        #    tick.set_rotation(30)
        #    tick.set_ha('right')
        peaks, _ = find_peaks(power, height=0.08, distance = 10)
        #for val in peaks:
        #    print(val, freq[val])
        #    try:
        #        ax.axvline(freq[val], linestyle='--', color='k', linewidth=1, alpha=0.45)
        #    except Exception:
        #        pass
        leg = ax.legend(loc='best', fontsize=10, edgecolor='k')
        #leg1 = axins.legend(loc='best', fontsize=8, edgecolor='k')
        #$leg1 = axins1.legend(loc='best', fontsize=8, edgecolor='k')
        #leg.get_frame().set_edgecolor('k')
        #t = ax.text(0,0,'',fontsize=10)
        factor +=1
    #ax.grid(linestyle='-', linewidth='0.5', color='k')
    print(f'Average FAP for peak 1 {peak1/5:}')
    print(f'Average FAP for peak 2 {peak2/5:}')
    ax.set_ylabel('Lomb-Scargle Power', fontsize=12, labelpad=10, color='k')
    ax.set_xlabel('Frequency [cycles/day]', fontsize=12, labelpad=10, color='k')
    #fig.text(0.32, 0.02, 'Frequency [cycles/day]', fontproperties=t.get_font_properties(), color='k')
    #fig.text(0.03, 0.63, 'Lomb-Scargle Power', fontproperties=t.get_font_properties(), rotation='vertical', color='k')
    fout = os.path.join(APJ_PLOT_DIR, 'cr_rate_periodogram.png')
    fig.savefig(fout, format='png', dpi=350, bbox_inches='tight',transparent=False)
    plt.show()
    return data_out


def rate_vs_time(hrc, stis, wfc, wfpc2, uvis,ms=4, min_exptime=40, delta=0.15):
    v = visualize.Visualizer()
    fig, ax1 = plt.subplots(nrows=1, ncols=1,
                       figsize=(7,5),
                       sharex=True)
  
    smooth_type = 'rolling'
    window='30D'
    min_periods=25
    #for ax in [ax1, ax2]:
    #    ax.grid(linestyle='-', linewidth='0.5', color='k')
    #    plt.setp(ax.spines.values(), color='k')
    stis_plot_params = {
        'df': stis,
        'legend_label': 'STIS/CCD',
        'ax': ax1,
        'i':0,
    #    'thick':13.5,
        'smooth_type': smooth_type,
        'window': window,
        'min_periods': min_periods
    }
    hrc_plot_params = {
        'df': hrc,
        'legend_label': 'ACS/HRC',
        'ax': ax1,
     #   'thick':14,
        'i':1,
        'smooth_type': smooth_type,
        'window': window,
        'min_periods': min_periods
    }
    wfc_plot_params = {
        'df': wfc,
        'legend_label': 'ACS/WFC',
        'ax': ax1,
      #  'thick':14.5,
        'i':2,
        'smooth_type': smooth_type,
        'window': window,
        'min_periods': min_periods
    }
    wfpc2_plot_params = {
        'df': wfpc2,
        'legend_label': 'WFPC2',
        'ax': ax1,
        'i':3,
       # 'thick':10,
        'smooth_type': smooth_type,
        'window': window,
        'min_periods': min_periods
    }
    uvis_plot_params = {
        'df': uvis,
        'legend_label': 'WFC3/UVIS',
        'ax': ax1,
        'i':4,
        #'thick':16,
        'smooth_type': smooth_type,
        'window': window,
        'min_periods': min_periods
    }
    ax1.tick_params(which='both', axis='both', color='k', labelcolor='k',labelsize=11)
    #ax2.tick_params(which='both', axis='both', color='k', labelcolor='k')
    #fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    #fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    # datasets = [stis_plot_params, wfpc2_plot_params, wfc_plot_params]
    datasets = [stis_plot_params, wfpc2_plot_params, hrc_plot_params,
                wfc_plot_params, uvis_plot_params]
    for i,dset in enumerate(datasets):
        yoffset = i*delta
        dset['legend_label'] +=f" + {i}*$\Delta$"
        fig, ax1 = v.plot_cr_rate_vs_time(**dset,ms=ms,normalize=True,min_exptime=min_exptime, yoffset=yoffset)

    ax1.yaxis.set_minor_locator(MultipleLocator(0.05))
    ax1.yaxis.set_major_locator(MultipleLocator(0.2))
    ax1.xaxis.set_minor_locator(mdates.YearLocator())
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    #ax2.yaxis.set_minor_locator(MultipleLocator(5))
    #ax2.yaxis.set_major_locator(MultipleLocator(25))
    #ax2.xaxis.set_minor_locator(mdates.YearLocator())
    solar_df = read_solar_data()
    solar_cycle = get_solar_min_and_max(solar_df)
   
   #solar_cycle ={'maximum_cycle23':Time('2000-04-01', format='iso'),
   #              'maximum_cycle24':Time('2014-02-01', format='iso'),
   #              'minimum_cycle23':Time('1996-05-01', format='iso'),
   #              'minimum_cycle24': Time('2008-12-01', format='iso')}
    #ax2.plot(solar_df.index.values, 
    #        solar_df['sunspot RI'], 
    #        label='Monthly Mean',
    #        c='#1E88E5')

    #ax2.plot(solar_df.index.values,
    #         solar_df['sunspot RI smooth'],
    #         label='Smoothed',
    #         c='#D81B60')
    ax1.tick_params(labelbottom=True)
    for i,cycle in enumerate(solar_cycle.keys()):
            # Min
            if i == 0:
                ax1.axvline(solar_cycle[cycle][0],label='Solar Min', ls='--', color='k')
             # Max
                ax1.axvline(solar_cycle[cycle][1], label='Solar Max',ls='-', color='k')
            

            ax1.axvline(solar_cycle[cycle][0],ls='--', color='k')
            # Max
            ax1.axvline(solar_cycle[cycle][1],ls='-', color='k')
            #ax2.axvline(solar_cycle[cycle][1], ls='-', color='k')

    ax1_legend = ax1.legend(loc='upper left',
                            bbox_to_anchor = (1.02, 1.),
                            ncol=1,
                            labelspacing=0.2,fontsize=11,
                            columnspacing=0.5)
    ax1_legend.get_frame().set_edgecolor('k')

    for i in range(len(ax1_legend.legendHandles)): 
        ax1_legend.legendHandles[i]._sizes = [15]
    date_min = Time('1992-01-01', format='iso')
    date_max = Time('2020-01-01', format='iso')
    ax1.set_xlim((date_min.to_datetime(), date_max.to_datetime()))
  #  ax2.set_xlim((date_min.to_datetime(), date_max.to_datetime()))
  #  ax2.set_ylabel('$R_I$', fontsize=10, color='k')
  #  ax2.set_xlabel('Date', fontsize=10,color='k')
    ax1.set_ylim(0.75, 2)
    ax1.set_ylabel('Median Normalized CR Flux',fontsize=11, color='k')
    ax1.set_xlabel('Observation Date', fontsize=11, color='k')
    #for ax in [ax1, ax2]:
    ax1.tick_params(axis='both', which='minor', width=1, length=2.5)
    ax1.tick_params(axis='both', which='major', width=1.5, length=5)
    fout = os.path.join(APJ_PLOT_DIR,'cr_rate_vs_time.png')
    fig.savefig(fout, format='png', dpi=350, bbox_inches='tight',transparent=False)
    plt.show()


def ccd_substrate_model():
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 6))
    ax.set_facecolor('white')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.text(-1, 17,
            'ACS/WFC CCD Substrate Layers',
            fontsize='x-large')



    # top layer
    text_x = 8
    ax.text(6, 12, 'CCD Housing\n Environment', fontsize='large')

    arrow_w=0.05

    ax.text(0, 15.2,'Incident Light')
    ax.arrow(2, 15, 0.55, -4.75,
             color='k',shape='full', width=arrow_w)
    # reflected
    ax.arrow(2+0.7, 15-4.95, 0.55, 3,
             color='k', linestyle='dashed', width=arrow_w)
    ax.plot([2.65,2.65],[10,15], ls='--', color='k')
    ax.axhline(10, xmin=0.1)

    # Si
    ax.arrow(2.65, 10., 0.53, -3.75,
             color='k', shape='full', width=arrow_w)
    # reflected
    ax.arrow(2.65 + 0.6, 10 - 3.95, 0.45, 1.75,
             color='k', linestyle='dashed',
             width=arrow_w)

    ax.plot([3.25, 3.25], [6, 9], ls='--', color='k')
    ax.text(text_x, 7.7, '$Si$', fontsize='large')
    ax.axhline(6, xmin=0.1)

    # SiO2
    ax.arrow(3.25, 6., 0.35, -1.75, color='k', shape='full', width=arrow_w)

    #reflected
    ax.arrow(3.24 + 0.45, 6 - 1.95, 0.3, 1.,
             color='k', linestyle='dashed',
             width=arrow_w)
    ax.plot([3.68, 3.68], [4, 5.5], ls='--', color='k')
    ax.text(text_x, 4.8,r'$SiO_2$', fontsize='large')
    ax.axhline(4, xmin=0.1)

    # Si3N4
    ax.text(text_x, 2.8, r'$Si_{3}N_4$', fontsize='large')
    ax.arrow(3.7, 4., 0.45, -1.8, color='k', shape='full', width=arrow_w)
    # reflected
    ax.arrow(3.75 + 0.5, 4 - 1.95, 0.25, 1.,
             color='k', linestyle='dashed',
             width=arrow_w)
    ax.plot([4.22, 4.22], [2, 3.5], ls='--', color='k')
    ax.axhline(2, xmin=0.1)

    # Si
    ax.text(text_x, 0.85, r'$Si$', fontsize='large')
    ax.arrow(4.25, 2, 0.55, -1.8, color='k', shape='full', width=arrow_w)
    # reflected
    ax.arrow(4.25 + 0.65, 2 - 1.9, 0.3, 1.,
             color='k', linestyle='dashed',
             width=arrow_w)

    ax.plot([4.87, 4.87], [0, 1.5], ls='--', color='k')

    # Substrate
    ax.axhline(0, xmin=0.1)
    ax.text(text_x, -1, r'$Substrate$', fontsize='large')

    ax.axhline(-1.75, xmin=0.1)

    ax.grid(False)
    ax.set_xlim(-1,12)
    ax.set_ylim(-2,15.5)
    fout = os.path.join(APJ_PLOT_DIR,'ccd_substrate_example.png')
    fig.savefig(fout,
                format='png',
                dpi=300,
                bbox_inches='tight',
                transparent=True,
                frameon=False)


def thickness_histograms():
    data_dict_th = {
        'acs_hrc': {
            'fname': '/Users/nmiles/hst_cosmic_rays/results/hrc_th_Si.fits',
            'data': None,
            'interval': [12.49, 16.03],
            'norm': None,
            'im': None,
            'cbar_ticks': [13, 14, 15, 16],
        },
        'acs_wfc': {
            'fname': '/Users/nmiles/hst_cosmic_rays/results/wfc_th1.fits',
            'data': None,
            'interval': [12.60, 17.10],
            'norm': None,
            'im': None,
            'cbar_ticks': [13, 14, 15, 16, 17],
        },
        'wfc3_uvis': {
            'fname': '/Users/nmiles/hst_cosmic_rays/results/wfc3_uvis_thickness.fits',
            'data': None,
            'interval': [13.50, 18.00],
            'cbar_ticks': [14, 15, 16, 17, 18],
            'norm': None,
            'im': None
        }

    }

    data_dict_cr = {
        'acs_hrc': {
            'fname': '/Users/nmiles/hst_cosmic_rays/results/smoothed_acs_hrc.fits',
            'data': None,
            'interval': [130, 205],
            'norm': None,
            'im': None,
            'cbar_ticks': [130, 145, 160, 175, 190, 205],
        },
        'acs_wfc': {
            'fname': '/Users/nmiles/hst_cosmic_rays/results/acs_wfc_smoothed.fits',
            'data': None,
            'interval': [60, 110],
            'norm': None,
            'im': None,
            'cbar_ticks': [60, 70, 80, 90, 100, 110],
        },
        'wfc3_uvis': {
            'fname': '/Users/nmiles/hst_cosmic_rays/results/wfc3_uvis_smoothed.fits',
            'data': None,
            'interval': [140, 240],
            'cbar_ticks': [140, 160, 180, 200, 220, 240],
            'norm': None,
            'im': None
        }

    }

    for key in data_dict_th.keys():
        data_dict_th[key]['data'] = fits.getdata(data_dict_th[key]['fname'])
        data_dict_th[key]['norm'] = ImageNormalize(
            data_dict_th[key]['data'],
            stretch=LinearStretch(),
            vmin=data_dict_th[key]['interval'][0],
            vmax=data_dict_th[key]['interval'][1]
        )
        # Get the CR data
        cr_data = fits.getdata(data_dict_cr[key]['fname'])
        smoothed = gaussian_filter(cr_data, sigma=2)
        data_dict_cr[key]['data'] = smoothed
        data_dict_cr[key]['norm'] = ImageNormalize(
            data_dict_cr[key]['data'],
            stretch=LinearStretch(),
            vmin=data_dict_cr[key]['interval'][0],
            vmax=data_dict_cr[key]['interval'][1]
        )

    v = visualize.Visualizer()
    fig, axes = v.mk_fig(nrows=1, ncols=3, figsize=(5, 3),
                         sharex=True,sharey=True)



def thickness_plot(fname=None, fname_comp=None, fout=None, instr=None):
    """

    Parameters
    ----------
    fname
    fname_comp

    Returns
    -------

    """
    data_dict_th = {
        'acs_hrc':{
            'fname':'/Users/nmiles/hst_cosmic_rays/results/hrc_th_Si.fits',
            'data': None,
            'interval': [12.49, 16.03],
            'norm': None,
            'im' : None,
            'cbar_ticks': [13, 14, 15, 16],
        },
        'acs_wfc':{
            'fname':'/Users/nmiles/hst_cosmic_rays/results/wfc_th1.fits',
            'data': None,
            'interval':[12.60, 17.10],
            'norm':None,
            'im': None,
            'cbar_ticks': [13, 14, 15, 16, 17],
        },
        'wfc3_uvis':{
            'fname':'/Users/nmiles/hst_cosmic_rays/results/wfc3_uvis_thickness.fits',
            'data':None,
            'interval': [13.50, 18.00],
            'cbar_ticks': [14, 15, 16, 17, 18],
            'norm':None,
            'im': None
        }

    }

    data_dict_cr = {
        'acs_hrc':{
            'fname':'/Users/nmiles/hst_cosmic_rays/results/smoothed_acs_hrc.fits',
            'data': None,
            'interval': [130, 205],
            'norm': None,
            'im' : None,
            'cbar_ticks': [130, 145, 160, 175, 190, 205],
        },
        'acs_wfc':{
            'fname':'/Users/nmiles/hst_cosmic_rays/results/acs_wfc_smoothed.fits',
            'data': None,
            'interval': [60, 110],
            'norm':None,
            'im': None,
            'cbar_ticks': [60, 70, 80, 90, 100, 110],
        },
        'wfc3_uvis':{
            'fname':'/Users/nmiles/hst_cosmic_rays/results/wfc3_uvis_smoothed.fits',
            'data':None,
            'interval': [140, 240],
            'cbar_ticks': [140, 160, 180, 200, 220, 240],
            'norm':None,
            'im': None
        }

    }

    # rc('text', usetex=True)

    data = []

    for key in data_dict_th.keys():
        data_dict_th[key]['data'] = fits.getdata(data_dict_th[key]['fname'])
        data_dict_th[key]['norm'] = ImageNormalize(
            data_dict_th[key]['data'],
            stretch=LinearStretch(),
            vmin=data_dict_th[key]['interval'][0],
            vmax=data_dict_th[key]['interval'][1]
        )
        # Get the CR data
        cr_data = fits.getdata(data_dict_cr[key]['fname'])
        smoothed = gaussian_filter(cr_data, sigma=2)
        data_dict_cr[key]['data'] = smoothed
        data_dict_cr[key]['norm'] = ImageNormalize(
            data_dict_cr[key]['data'],
            stretch=LinearStretch(),
            vmin=data_dict_cr[key]['interval'][0],
            vmax=data_dict_cr[key]['interval'][1]
        )

    # v = visualize.Visualizer()
    # fig, axes = v.mk_fig(nrows=2, ncols=3, figsize=(9, 6))
    fig = plt.figure(figsize=(8, 4))
    gs0 = gridspec.GridSpec(ncols=3, nrows=2, figure=fig, hspace=0.05, wspace=0.3)
    # fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(9,6))
    axes = [
        fig.add_subplot(gs0[i, j]) for i in range(2) for j in range(3)
    ]

    # Plot the thickness data
    for ax, key in zip(axes[:3], data_dict_th.keys()):
        ax.grid(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if key =='acs_wfc':
            data_dict_th[key]['im'] = ax.imshow(data_dict_th[key]['data'],
                                         norm=data_dict_th[key]['norm'],
                                         cmap='plasma')
        else:
            data_dict_th[key]['im'] = ax.imshow(data_dict_th[key]['data'],
                                             norm=data_dict_th[key]['norm'],
                                             cmap='plasma', origin='lower')
        ax.set_title('{}'.format(key.replace('_','/').upper()))
        divider = make_axes_locatable(ax)
        # cax = divider.append_axes('bottom', size='5%', pad=0.05)
        # cbar = fig.colorbar(data_dict_th[key]['im'], cax=cax,
        #                     ticks=data_dict_th[key]['cbar_ticks'],
        #                     orientation='horizontal')
        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(data_dict_th[key]['im'], cax=cax,
                            ticks=data_dict_th[key]['cbar_ticks'],
                            orientation='vertical')
        # cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1f'))
        # cbar.ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        cbar_labels = [str(x) for x in data_dict_th[key]['cbar_ticks']]
        # cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
        cbar.ax.set_yticklabels(cbar_labels, ha='left', rotation=0, fontsize=8)

        cbar.set_label(r'Thickness $[\mu m]$', fontsize=10)
    # fout = os.path.join(APJ_PLOT_DIR, 'thickness_all_instr.png')
    # fig.savefig(fout,
    #             transparent=True,
    #             format='png',
    #             dpi=350,
    #             bbox_inches='tight')
    # plt.show()

    # fig, axes = v.mk_fig(nrows=1, ncols=3, figsize=(10, 5))
    # plot the CR data
    for ax, key in zip(axes[3:], data_dict_th.keys()):
        ax.grid(False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        # ax.set_title('{}'.format(key.replace('_', '/').upper()))
        data_dict_cr[key]['im'] = ax.imshow(data_dict_cr[key]['data'],
                                            norm=data_dict_cr[key]['norm'],
                                            cmap='plasma', origin='lower')

        divider = make_axes_locatable(ax)
        # cax = divider.append_axes('bottom', size='5%', pad=0.05)
        # cbar = fig.colorbar(data_dict_cr[key]['im'], cax=cax,
        #                     ticks=data_dict_cr[key]['cbar_ticks'],
        #                     orientation='horizontal')

        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(data_dict_cr[key]['im'], cax=cax,
                            ticks=data_dict_cr[key]['cbar_ticks'],
                            orientation='vertical')
        cbar_labels = [str(x) for x in data_dict_cr[key]['cbar_ticks']]
        cbar.ax.set_yticklabels(cbar_labels,ha='left', rotation=0, fontsize=8)
        cbar.set_label(r'Num. of CR Strikes', fontsize=10)

    # Add a colorbar to show the image scaling
    # divider1 = make_axes_locatable(ax1)
    # cax1 = divider1.append_axes('bottom', size='5%', pad=0.1)
    # cbar1 = fig1.colorbar(im1, cax=cax1, orientation='horizontal')
    # cbar1.ax.set_xticklabels(cbar1.ax.get_xticklabels(), rotation=45)
    # cbar1.set_label('Cosmic Ray Strikes')
    # if not astrofits:
    #     norm1 = ImageNormalize(comp_data,
    #                            stretch=LinearStretch(),
    #                            interval=ZScaleInterval())
    #     im2 = ax2.imshow(comp_data, norm=norm1, cmap='plasma')
    # else:
    #     norm1 = ImageNormalize(comp_data,
    #                            stretch=LinearStretch(),
    #                            vmin=12.5, vmax=16)
    #     im2 = ax2.imshow(comp_data, cmap='plasma', norm=norm1)#, origin='lower')
    # # Add a colorbar to show the image scaling
    # divider2 = make_axes_locatable(ax2)
    # cax2 = divider2.append_axes('bottom', size='5%', pad=0.1)
    # cbar2 = fig2.colorbar(im2, cax=cax2, orientation='horizontal')
    # cbar2.ax.set_xticklabels(cbar2.ax.get_xticklabels(), rotation=45)
    # cbar2.set_label(r'Thickness $[\mu m]$')
    # ax1.grid(False)
    # ax2.grid(False)
    # ax1.set_title('WFC Cosmic Ray Incidence Heat Map')
    # ax2.set_title('WFC Fringing Thickness Map')
    #
    # # fig.suptitle(instr,
    # #              x=0.5, y=0.9,
    # #              horizontalalignment='center',
    # #              size=16, weight='bold')
    # fig1.savefig('cr_heat_map_WFC.png',
    #              transparent=True, format='png', dpi=350, bbox_inches='tight')
    fout = os.path.join(APJ_PLOT_DIR, 'cr_th_all_instr_proceedings.eps')
    fig.savefig(fout,
                 format='eps',
                 dpi=150, bbox_inches='tight')
    plt.show()

def plot_example_darks(hrc=None, wfc=None, wfpc2=None, stis=None, uvis=None):
    hrc = '/Users/nmiles/hst_cosmic_rays/data/ACS/HRC/mastDownload/HST/j8ba0hrpq/j8ba0hrpq_flt.fits'
    wfc = '/Users/nmiles/hst_cosmic_rays/data/ACS/WFC/j8jbrcgrq_flt.fits'
    stis = '/Users/nmiles/hst_cosmic_rays/data/STIS/STIS_grazing_CR/o3sl01pcq_flt.fits'
    wfpc2 = '/Users/nmiles/hst_cosmic_rays/data/WFPC2/mastDownload/HST/u21y2801t/u21y2801t_c0m.fits'
    uvis = '/Users/nmiles/hst_cosmic_rays/data/WFC3/UVIS/icfcafaaq_blv_tmp.fits'

    fig = plt.figure(figsize=(7,5))
    gs0 = gridspec.GridSpec(ncols=5, nrows=1, figure=fig, hspace=0, wspace=0)
    # gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=6, hspace=0,
    #                                         wspace=0., subplot_spec=gs0[0])
    # gs10 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=6, hspace=0,
    #                                         wspace=0., subplot_spec=gs0[1])
    # ax1 = fig.add_subplot(gs00[0, 1:3])
    # ax2 = fig.add_subplot(gs00[0, 3:5])
    # ax3 = fig.add_subplot(gs10[0, :2])
    # ax4 = fig.add_subplot(gs10[0, 2:4])
    # ax5 = fig.add_subplot(gs10[0, 4:6])
    ax1 = fig.add_subplot(gs0[0])
    ax2 = fig.add_subplot(gs0[1])
    ax3 = fig.add_subplot(gs0[2])
    ax4 = fig.add_subplot(gs0[3])
    ax5 = fig.add_subplot(gs0[4])
    axes = [ax1, ax2, ax3, ax4, ax5]
    labels = ['ACS/HRC', 'ACS/WFC', 'STIS/CCD', 'WFPC2', 'WFC3/UVIS']
    datasets = [hrc, wfc, stis, wfpc2, uvis]

    for dset, label, ax in zip(datasets, labels, axes):
        with fits.open(dset) as hdu:
            data = hdu[1].data
            norm = ImageNormalize(data, stretch=LinearStretch(), interval=ZScaleInterval())
            ax.imshow(data, norm=norm, origin='lower', cmap='gray')
            ax.yaxis.set_major_locator(plt.NullLocator())
            ax.xaxis.set_major_locator(plt.NullLocator())
            ax.grid(False)
            ax.set_xlim((100,400))
            ax.set_ylim((100,400))
            # ax.text(x=120,y=356, s=label, color='#F4CC70',fontsize=8, fontweight='medium', backgroundcolor='white')
            # ax.set_title(label)

    fig.savefig(
        os.path.join(APJ_PLOT_DIR, 'example_darks_transparent.png'),
        format='png',
        dpi=350,
        bbox_inches='tight',
        transparent=True)

    plt.show()


def get_solar_min_and_max(noaa_data):
    """ Get the dates of solar max and solar min for cycle 23 & 24

    Parameters
    ----------
    noaa_data

    Returns
    -------

    """
    solar_cycle = {'Cycle 23': None, 'Cycle 24':None}
    min_1996 = noaa_data['1993-01-01':'1997-01-01'].idxmin()
    max_2000 = noaa_data['1998-01-01':'2004-01-01'].idxmax()
    min_2009 = noaa_data['2008-01-01':'2011-01-01'].idxmin()
    max_2014 = noaa_data['2011-01-01':'2017-01-01'].idxmax()

    solar_cycle['Cycle 23'] = (min_1996['sunspot RI smooth'],
                                    max_2000['sunspot RI smooth'])

    solar_cycle['Cycle 24'] = (min_2009['sunspot RI smooth'],
                                    max_2014['sunspot RI smooth'])
    return solar_cycle


def download_solar_indices():
    """Downloads the monthly mean and smoothed monthly mean solar indices from NOAA

    Returns
    -------

    """
    noaa_client = sunpy.net.dataretriever.sources.NOAAIndicesClient()

    # Get the NOAA indices for the defined time period
    query = noaa_client.search(sunpy.net.attrs.Time('1990/1/1', '2019/3/1'))
    noaa_client.fetch(query, './../../data/')


def read_solar_data():
    noaa = TimeSeries('./../../data/RecentIndices.txt', source='NOAAIndices')
    df = noaa.to_dataframe()
    return df

def plot_solar_cycle(all_integration_df, smoothed=True, figsize=(5,4), save=True):
    v = visualize.Visualizer()
    noaa_df = read_solar_data()
    solar_cycles = get_solar_min_and_max(noaa_data=noaa_df)
    fig, ax = v.mk_fig(nrows=1, ncols=1, figsize=figsize)
    ax1 = ax.twinx()
    ax.set_axisbelow(True)
    
    sampled = all_integration_df.resample(rule='6M').sum()
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                          '#f781bf', '#a65628', '#984ea3',
                          '#999999', '#e41a1c', '#dede00']
    ax1.step(sampled.index.values,
                    sampled.integration_time/(60*60),where='pre',
                    label='Total Integration Time [hr]',c='k',alpha=0.65, ls='--'
                    )
    ax1.yaxis.set_major_locator(MultipleLocator(200))
    ax1.yaxis.set_minor_locator(MultipleLocator(50))
    ax1.grid(False)
    ax1.set_ylim((0, 1000))
    ax.yaxis.set_major_locator(MultipleLocator(50))
    ax.yaxis.set_minor_locator(MultipleLocator(10))
    ax.set_ylim((0,250))
    ax.plot(noaa_df.index.values,
            noaa_df['sunspot RI smooth'],
            label='Smoothed Sunspot Number',
            c='#1E88E5')
    date_min = Time('1990-12-01', format='iso')
    date_max = Time('2020-01-01', format='iso')


    # Add vertical lines for cycle 23 minima and shade between
    cycle23_start = Time(solar_cycles['Cycle 23'][0]).to_datetime()
    cycle23_max = Time(solar_cycles['Cycle 23'][1]).to_datetime()

    cycle23_end = Time(solar_cycles['Cycle 24'][0]).to_datetime()
    cycle24_max = Time(solar_cycles['Cycle 24'][1]).to_datetime()

   # mpl.rcParams['hatch.linewidth'] = 0.2
    ax.axvspan(cycle23_start, cycle23_end,
               facecolor='k',
               alpha=0.2,
               )

    ax.text(cycle23_max + timedelta(days=2*365), 200, s='Cycle 23', fontsize=12)

    predicted_cycle24_end = Time('2019-06-01', format='iso').to_datetime()
    ax.axvspan(cycle23_end, predicted_cycle24_end,
               facecolor='r',
               alpha=0.2,
               )
    ax.text(cycle24_max - timedelta(days=365), 200, s='Cycle 24',
            fontsize=12)

    operational_coverage_start = Time('1994-01-01', format='iso').to_datetime()
    #ax.axvspan(operational_coverage_start, predicted_cycle24_end,
    #           facecolor='r',
    #           alpha=0.0,hatch='/'
    #           )
    ax.tick_params(axis='both', which='major', width=1.5, length=5)
    ax.tick_params(axis='both', which='minor', width=1, length=2.5)
    ax1.tick_params(axis='both', which='major', width=1.5, length=5)
    ax.tick_params(axis='both', which='minor', width=1, length=2.5)
    xstart = Time('1992-01-01', format='iso').to_datetime()
    xstop = Time('2020-01-01', format='iso').to_datetime()
    ax.set_xlim((xstart, xstop))
    ax.set_xlabel('Date', fontsize=10)
    ax1.set_ylabel('Total Integration Time [hr]',fontsize=10)
    ax.set_ylabel('Smoothed Sunspot Number')
    leg = ax.legend(loc='upper left', fontsize=10, edgecolor='k')
    leg1 = ax1.legend(loc='upper right', fontsize=10, edgecolor='k')
    #ax.set_ylim(0, noaa_df['sunspot RI'].max() + 30)
    ax.xaxis.set_minor_locator(mdates.YearLocator())
    # ax.set_title('International Sunspot Number')
    if save:
        fout = os.path.join(APJ_PLOT_DIR, 'solar_cycle.png')
        fig.savefig(fout,format='png', dpi=300,bbox_inches='tight', transparent=False)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='left')


def read_SAA_data():
    stis = dh.DataReader(instr='STIS_CCD', statistic='incident_cr_rate')
    stis.hdf5_files = \
        glob.glob(
            '/Users/nmiles/hst_cosmic_rays/results/STIS/stis_ccd_saa_cr_rate*hdf5'
        )
    stis.read_cr_rate()
    wfc3 = dh.DataReader(instr='WFC3_UVIS', statistic='incident_cr_rate')
    wfc3.hdf5_files = \
        glob.glob(
            '/Users/nmiles/hst_cosmic_rays/results/WFC3/wfc3_uvis_saa_cr_rate*hdf5'
        )
    wfc3.read_cr_rate()
    return stis, wfc3

def plot_stis_saa_images():
    data_dir = '/Users/nmiles/hst_cosmic_rays/data/STIS/SAA_data/HST/'
    df = pd.read_csv(
        'stis_ccd_catalog.txt',
        header=0,
        index_col='date_start',
        parse_dates=True
    )
    df = df.sort_index()
    df = df[df.integration_time.lt(1000)]
    saa_northern = (312.0, 1.0)
    saa_southern = (300.0, -60.0)

    mask = df['latitude_start'] < saa_northern[1]
    df_saa_cut = df[mask]
    norm_data = fits.getdata(
        '/Users/nmiles/hst_cosmic_rays/data/'
        'STIS/STIS_grazing_CR/o3sl01pcq_flt.fits'
    )
    norm_hdr = fits.getheader(
        '/Users/nmiles/hst_cosmic_rays/data/'
        'STIS/STIS_grazing_CR/o3sl01pcq_flt.fits'
    )
    norm = ImageNormalize(norm_data*norm_hdr['ATODGAIN'], stretch=SqrtStretch(), vmin=0, vmax=50)


    img_data = {}
    img_info = defaultdict(list)
    for f in df_saa_cut['obs_id']:
        hdr = fits.getheader(os.path.join(data_dir,f.split('_flt')[0], f))
        img_info['OBSID'].append(f)
        img_info['EXPTIME [s]'].append(hdr['TEXPTIME'])
        img_info['EXPSTART [MJD]'].append(hdr['TEXPSTRT'])
        img_data[f] = fits.getdata(
            os.path.join(data_dir,f.split('_flt')[0], f)
        )
        img_data[f] *= hdr['ATODGAIN']
    fig = plt.figure(figsize=(7,6))
    gs = gridspec.GridSpec(4, 5, hspace=0.25, wspace=0)
    axes = []
    for i in range(19):
        ax = fig.add_subplot(gs[i])
        ax.grid(False)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        axes.append(ax)

    xlim = (425, 575)
    ylim = (425, 575)
    for i, key in enumerate(img_data.keys()):
        im = axes[i].imshow(img_data[key], norm=norm, cmap='gray', origin='lower')
        axes[i].set_title(f"{i+1}", fontsize=10)
        axes[i].set_xlim(xlim)
        axes[i].set_ylim(ylim)
    cax = fig.add_axes([0.9, 0.1, 0.033, 0.77])
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('ELECTRONS', weight='bold')
    df = pd.DataFrame(img_info)
    tb = df.to_latex(index=False)
    with open('stis_saa_images.txt', 'a') as fobj:
        fobj.write(tb)

    fig.savefig(
        os.path.join(APJ_PLOT_DIR, 'stis_saa_observations.png'),
        format='png',
        dpi=250,
        bbox_inches='tight'
    )
    plt.show()
    return img_data

def compute_saa_rates():
    df = pd.read_csv(
        'stis_ccd_catalog.txt',
        header=0,
        index_col='date_start',
        parse_dates=True
    )
    df = df.sort_index()
    df = df[df.integration_time.lt(1000)]
    saa_northern = (312.0, 1.0)
    saa_southern = (300.0, -60.0)

    mask = df['latitude_start'] < saa_northern[1]
    df_saa_cut = df[mask]
    stis_avg_energy_per_cr = 2621.1422
    area = 4.624

    rates = []
    num_crs =[]
    for i, row in df_saa_cut.iterrows():
        num_cr = row['cumulative_energy']/stis_avg_energy_per_cr
        rate = num_cr/(row['integration_time']*area)
        num_crs.append(num_cr)
        rates.append(rate)
    df_saa_cut['estimated_rates'] = rates
    df_saa_cut['num_crs'] = num_crs
    df.to_csv('stis_ccd_catalog_with_estimated_rates.txt', header=True, index=True)
    return df_saa_cut


def stis_saa_plot(data_df=None, i=5):

    # stis_reader = dh.DataReader(statistic='incident_cr_rate', instr='STIS_CCD')
    # flist = glob.glob(
    #     '/Users/nmiles/hst_cosmic_rays/results/'
    #     'STIS/stis_saa_results/stis*cr_rate*hdf5'
    # )
    # stis_reader.hdf5_files = flist
    # print(stis_reader.hdf5_files)
    # stis_reader.read_cr_rate()
    stis = pd.read_csv(
        'stis_ccd_catalog.txt',
        header=0,
        index_col='date_start',
        parse_dates=True
    )
    # print(stis_reader.data_df)
    # stis = stis_reader.data_df['1997-02-01':'1997-03-01']

    saa_eastern = (39.0, -30.0)  # lon/lat
    saa_western = (267.0, -20.0)
    saa_northern = (312.0, 1.0)
    saa_southern = (300.0,-60.0)

    mask = (stis['latitude_start'] < saa_northern[1]) #& (stis['incident_cr_rate'] < 20)

    stis_saa_cut = stis[mask]
    # fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7,6))
    fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(7, 5))
    # Create the lat/lon map
    m = Basemap(projection='cyl',llcrnrlon=-120,
                llcrnrlat= -60,
                urcrnrlon= 60,
                urcrnrlat= 10,
                ax=ax1)

    m.shadedrelief(scale=0.2)

    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13),
                                  labels=[True, True, False, False])

    lons = m.drawmeridians(np.linspace(-180, 180, 13),
                                  labels=[False, False, False, True])

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')

    saa = [list(t) for t in zip(*costools.saamodel.saaModel(i))]
    # Ensure the polygon representing the SAA is a closed curve by adding
    # the starting points to the end of the list of lat/lon coords
    saa[0].append(saa[0][0])
    saa[1].append(saa[1][0])
    m.plot(saa[1], saa[0],
                  c='r',
                  latlon=True,
                  label='SAA contour {}'.format(i))

    hst_lon, hst_lat = stis_saa_cut['longitude_start'], stis_saa_cut['latitude_start']
    shifted_lon = []
    for i, lon in enumerate(hst_lon):
        if lon > 180.0:
            shifted_lon.append(lon - 360.0)
        else:
            shifted_lon.append(lon)
    x_coord, y_coord = m(shifted_lon, hst_lat)
    labels = [k for k in range(len(stis_saa_cut))]
    indices = []
    for j, (lon, lat, label) in enumerate(zip(hst_lon, hst_lat, labels)):
        if j >= 4:
            indices.append(j)
            print(j -4 +1, lat, lon)
            m.scatter(lon, lat,
                    marker='o', s=12,c='r',
                    latlon=True)

            ax1.annotate('{}'.format(j-4 + 1),
                         xy=(x_coord[j], y_coord[j]),fontsize=9,
                         xycoords='data')

    ax2.semilogy([k - 4 + 1 for k in indices],
                stis_saa_cut['cumulative_energy_per_area_per_time'][indices],
                 marker='o')
    # STIS CR stats computed from server data
    # 25% 2524.830691
    # 50% 3215.160755
    # 75% 4341.264237
    ax2.fill_between(
        np.arange(0, np.max(indices)+1),
        2524.830691,
        4341.264237,
        alpha=0.25,
        color='k'
    )
    ax2.axhline(3215.160755, color='k', ls='--')
    ax1.legend(loc='best')
    ax2.set_xticks([k - 4 + 1 for k in indices])
    ax2.tick_params(which='both', axis='both', labelsize=10)
    ax1.legend(loc='best')
    ax2.set_xticks([k - 4 + 1 for k in indices])
    ax2.set_xlim(0.25,19.75)
    ax2.set_ylim(1e3, 1e7)
    ax2.set_ylabel('Rate of Energy Deposition [$e^-/s/cm^2$]', fontsize=10)
    ax2.set_xlabel('Observation Number', fontsize=10)
    # ax1.set_title('STIS/CCD Observations of the South Atlantic Anomaly (SAA)')
    fig1.savefig('stis_saa_crossing.png', format='png', dpi=250, bbox_inches='tight')
    fig2.savefig('stis_saa_total_energy.png', format='png', dpi=250, bbox_inches='tight')
    plt.show()
    # with open('stis_saa_darks.txt', 'a') as fobj:
    #     for indx in indices:
    #         fobj.write('{}\n'.format(stis_saa_cut['date'][indx]))

def plot_grazing_cr():
    fname = '/Users/nmiles/hst_cosmic_rays/data/STIS/STIS_grazing_CR/o3sl01pcq_flt.fits'
    metadatum = metadata.GenerateMetadata(fname=fname, instr='STIS_CCD')
    metadatum.get_image_data()
    metadatum.get_observatory_info()

    crlabel = labeler.CosmicRayLabel(
        fname=fname,
        gain_keyword=metadatum.instr_cfg['instr_params']['gain_keyword']
    )
    crlabel.get_data(
        extname='sci',
        extnums=metadatum.instr_cfg['instr_params']['extnums']
    )

    crlabel.ccd_labeling(
        use_dq=False,
        threshold_l=2,
        structure_element=[[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        threshold_u=5000,
    )
    # crlabel.plot()
    mean, med, std = sigma_clipped_stats(crlabel.sci.ravel(), maxiters=5,
                                         sigma_upper=4, sigma_lower=3.5)
    sizes = pd.Series(np.bincount(crlabel.label.ravel()), name='sizes')
    sizes = sizes.sort_values(ascending=False)
    print(sizes.iloc[:5])
    label_ids = sizes.index.values
    print(label_ids[:5])
    cr_indx = label_ids[2]
    coords = np.where(crlabel.label == cr_indx)
    xcr = 814
    ycr = 177

    xbound = 800
    ybound = 112
    yflag = coords[0] > 112
    xflag = coords[1] > 800

    coords = set(
        list(zip(coords[0][xflag & yflag], coords[1][xflag & yflag]))
    )
    print(len(coords))
    coords_to_exclude = set([(188, 811), (189, 811), (190, 811), (191,811),
                             (188, 812), (189, 812), (190, 812), (191,812),
                             (188, 813),
                             (176, 812),(177, 812), (176, 813),(177, 813)])

    coords = list(coords.difference(coords_to_exclude))
    print(len(coords))
    coords = np.array(coords)
    # print(coords)
    delta_y = np.max(coords[:, 1]) - np.min(coords[:, 1])
    delta_x = np.max(coords[:, 0]) - np.min(coords[:, 0])
    pix_values = []
    for coord in coords:
        pix_values.append(crlabel.sci[coord[0]][coord[1]])

    origin = (154, 830)
    print(origin)
    distances = []
    for coord in coords:
        diff = coord - origin
        if diff[1] < 0:
            distances.append(-1*np.hypot(*diff))
        else:
            distances.append(np.hypot(*diff))
    data = list(zip(pix_values, coords, distances))
    data.sort(key=lambda val: val[-1])
    pix_values, coords, distances = zip(*data)
    pix_values = np.array(pix_values)
    coords = np.array(coords)
    distances = np.array(distances)

    norm = ImageNormalize(crlabel.sci, stretch=SqrtStretch(), vmin=0, vmax=100)
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(
        ncols=3, nrows=2, figure=fig, hspace=0.45, wspace=0.15
    )
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1], sharex=ax2, sharey=ax2)
    ax4 = fig.add_subplot(gs[1, 2], sharex=ax2, sharey=ax2)
    # fig, ax = plt.subplots(nrows=1, ncols=2)
    ax1.scatter(distances, pix_values, marker='o')
    ax1.set_yscale('log')
    ax1.axhline(med, ls='--', color='k', label=f'Med: {med:.2f}$e^-$')
    # ax1.axhline(100, ls='--', color='r')
    ax1.legend(loc='upper left', edgecolor = 'k', fontsize = 8,
               bbox_to_anchor = (1.02, 1.)
               )
    for axis in [ax2, ax3, ax4]:
        axis.imshow(crlabel.sci, origin='lower', norm=norm, cmap='gray')
        axis.grid(False)
        axis.get_yaxis().set_visible(False)
        axis.get_xaxis().set_visible(False)
        axis.annotate(
            s='Origin',
            xy=(origin[1], origin[0]),
            xytext=(origin[1]+5, origin[0]+17),
            color='w',
            fontsize=12,
            arrowprops={'arrowstyle':'simple'}
        )


    # origin_patch = patches.Circle((origin[1], origin[0]), radius=1., color='r')
    # ax2.add_patch(origin_patch)
    color = '#e41a1c'
    for i, coord in enumerate(coords):
        if pix_values[i] > 1000:
            patch = patches.Rectangle((coord[1] - 0.5, coord[0] - 0.5),
                                      width=1, height=1., lw=0.85, color=color,
                                      fill=False)
            ax4.add_patch(patch)
        elif pix_values[i] >= 250 and pix_values[i] <= 1000:
            patch = patches.Rectangle((coord[1] - 0.5, coord[0] - 0.5),
                                      width=1, height=1., lw=0.85, color=color,
                                      fill=False)
            ax3.add_patch(patch)
        else:
            patch = patches.Rectangle((coord[1] - 0.5, coord[0] - 0.5),
                                      width=1, height=1., lw=0.85, color=color,
                                      fill=False)
            ax2.add_patch(patch)


    ax1.set_ylabel('$p(x,y)$ [$e^-$]', fontsize=10)
    ax1.set_xlabel('Distance From Origin [pix]', fontsize=10)
    ax2.set_title('$p(x,y) < 250 e^-$', fontsize=10)
    ax3.set_title('$ 250 \leq p(x,y) \leq 1000 e^-$',fontsize=10)
    ax4.set_title('$ p(x,y) > 1000 e^- $', fontsize=10)
    ax1.set_ylim(1, 1e4)
    ax1.set_xlim(-60, 60)
    ax1.xaxis.set_minor_locator(MultipleLocator(5))
    ax1.tick_params(axis='both', which='minor', width=1, length=2.5)
    ax1.tick_params(axis='both', which='major', width=1.5, length=5)
    ax2.set_xlim((795, 860))
    ax2.set_ylim((110, 205))
    fig.savefig(
        os.path.join(APJ_PLOT_DIR,'grazing_cr_plot.png'),
        dpi=250,
        bbox_inches='tight',
        format='png'
    )
    plt.show()

