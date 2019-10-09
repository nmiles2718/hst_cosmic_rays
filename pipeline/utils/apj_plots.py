#!/usr/bin/env python

from collections import defaultdict
from collections import Iterable
import logging
from itertools import chain
from datetime import timedelta
import glob
import os
from astropy.stats import sigma_clipped_stats, LombScargle, gaussian_sigma_to_fwhm
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import LinearStretch, ZScaleInterval,\
    AsinhStretch, SqrtStretch, ImageNormalize
import datahandler as dh
import dask.array as da

from matplotlib import rc
from matplotlib import ticker
import matplotlib as mpl
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
mpl.use('qt5agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec
plt.style.use('ggplot')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches

import numpy as np
import pandas as pd

import sunpy.net
from sunpy.timeseries import TimeSeries
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import find_peaks
import visualize
import metadata



APJ_PLOT_DIR = '../../APJ_plots/'


def read_data(stat='energy_deposited',min_exptime=50, units=None):
    reader_wfc = dh.DataReader(instr='ACS_WFC', statistic=stat)
    reader_hrc = dh.DataReader(instr='ACS_HRC', statistic=stat)
    reader_wfc3 = dh.DataReader(instr='WFC3_UVIS', statistic=stat)
    reader_wfpc2 = dh.DataReader(instr='WFPC2', statistic=stat)

    for r in [reader_wfpc2, reader_wfc, reader_hrc, reader_wfc3]:
        r.find_hdf5()

    reader_stis = dh.DataReader(instr='STIS_CCD', statistic=stat)
    flist = glob.glob('../../results/STIS_crrejtab_CRSIGMAS/*{}*hdf5'.format(stat))
    reader_stis.hdf5_files = flist
    if 'rate' in stat:
        for r in [reader_wfpc2,reader_stis, reader_wfc, reader_hrc, reader_wfc3]:
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


def plot_hst_loc_per_observation(
        fname = os.path.join(APJ_PLOT_DIR, 'odbrf7ggq_flt.fits'),
        figsize=(6,5),
        instr='STIS_CCD',
        exp1=None,
        exp2=None
):
    #fig, ax = plt.subplots(figsize=figsize, nrows=1, ncols=1)
    # Get observation metadata
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

    # # Draw the map
    # mapbase = draw_map(scale=0.5, ax=ax)
    #
    # Plot the path of HST
    # mapbase.plot(
    #    observation_exp1.metadata['longitude'],
    #    observation_exp1.metadata['latitude'],
    #    label=f'Int. Time: {exp1:.1f}s'
    #)
    #mapbase.plot(
    #    observation_exp2.metadata['longitude'],
    #    observation_exp2.metadata['latitude'],
    #    label=f'Int. Time: {exp2:.1f}s',
    #)
    #ax.legend(loc='upper right', edgecolor='k')
    # plt.show()
    return observation_exp1, observation_exp2


def hst_loc_plot(hrc, stis, wfc, wfpc2, uvis, orbital_path1=None, orbital_path2=None):
    #fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
    v = visualize.Visualizer()
    combined_data = {'latitude_start':[],'longitude_start':[], 'incident_cr_rate':[],
                     'integration_time':[]}
    instrument_name=['ACS/HRC','STIS/CCD','ACS/WFC','WFPC2','UVIS']
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                          '#f781bf', '#a65628', '#984ea3',
                          '#999999', '#e41a1c', '#dede00']
    for i,df in enumerate([hrc, stis, wfc, wfpc2, uvis]):
        lat = list(df['latitude_start'])
        lon = list(df['longitude_start'])
        cr_rate = list(df['incident_cr_rate'])
        combined_data['latitude_start'] += lat
        combined_data['longitude_start'] += lon
        combined_data['incident_cr_rate'] += cr_rate
        combined_data['integration_time'] += list(df['integration_time'])
    df_combined = pd.DataFrame(combined_data)
    fig = v.plot_hst_loc(df=df_combined, key='start', orbital_path1=orbital_path1,
                            orbital_path2=orbital_path2)
    fout = os.path.join(APJ_PLOT_DIR, 'cr_rate_vs_location_allinstr.png')
    fig.savefig(fout, format='png', dpi=350, bbox_inches='tight')
    return df_combined

def combine_integration_time_info(hrc, stis, wfc, wfpc2, uvis):
    instrument_name=['STIS/CCD','ACS/HRC','ACS/WFC','WFPC2','WFC3/UVIS']
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                          '#f781bf', '#a65628', '#984ea3',
                          '#999999', '#e41a1c', '#dede00']
    data_dict = {}
    for i,df in enumerate([stis, hrc, wfc, wfpc2, uvis]):
        counts = df['integration_time'].value_counts()
        data_dict[instrument_name[i]] = counts

    df = pd.DataFrame(data_dict)
    return df


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
    #colors = []
    #for int_time in cut.index:
    #    match = df[df['integration_time']==int_time]
    #    color = set(match['color'])
    #    instrument = set(match['instrument'])
    #    print(f'{color} {instrument}')
    #    colors.append(color)
    #cut.sort_index(inplace=True
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
    #ax.legend(loc='upper right',ncol=3, edgecolor='k')
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

def compute_basic_stats(hrc, stis, wfc, wfpc2, uvis):
    labels=['ACS/HRC', 'STIS/CCD', 'ACS/WFC', 'WFPC2', 'WFC3/UVIS']
    for dset, label in zip([hrc, stis, wfc, wfpc2, uvis],labels):
      #  avg = da.nanmean(dset, axis=0).compute()
       # std = da.nanstd(dset, axis=0).compute()
        #print(f'{label}: {avg}+\-{std}\n')
        quantiles = da.percentile(dset, q=[25, 50, 75], interpolation='linear').compute()
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
                label=instrument_name[i],
                normalize=normalize
            )
        else:
            if instrument_name[i] == 'WFC3/UVIS':
               #pass 
                print('removing postlfash')
                data = data - 12*10.0
            fig, ax, hist, bins = v.plot_hist(
                  data,
                  bins=bins,
                  c=CB_color_cycle[i],
                  range=drange,
                  ax=ax,
                  logy=logy,
                  label=instrument_name[i],
                  normalize=normalize
             )
    #ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    #ax.yaxis.set_minor_locator(AutoMinorLocator(5))
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
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3.25, 4.25))
    for dset, label, color in zip(datasets,labels, CB_color_cycle):
        altitude_df = dset.loc[:, ['mjd', 'altitude_start']]
        averaged = altitude_df.resample(rule='5D').mean()
        ax.scatter(averaged.index, averaged['altitude_start'], 
                    label=label,
                    s=3,
                    color=color)
    sm2 = Time('1995-07-01', format='iso').to_datetime()
    sm3b = Time('2002-03-01', format='iso').to_datetime()
    ax.text(x=sm2, y=605, s='SM2',fontsize=10)
    ax.text(x=sm3b, y=582, s='SM3B', fontsize=10)
    #ax.axvline(sm2, ls='--', c='k')
    #ax.axvline(sm3b, ls='--', c='k')
    ax.set_xlabel('Date',fontsize=10, color='k')
    ax.set_ylabel('Orbital Altitude [km]',fontsize=10, color='k')
    ax_leg = ax.legend(loc='upper right', edgecolor='k', fontsize=8)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    for i in range(len(ax_leg.legendHandles)):
         ax_leg.legendHandles[i]._sizes = [15]
    ax.tick_params(axis='both',labelsize=10, which='both', color='k', labelcolor='k')
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha('right')
    #ax.set_title('Orbital Decay of HST')
    fig.savefig(os.path.join(APJ_PLOT_DIR,'orbital_decay.png'), format='png', dpi=300, bbox_inches='tight')
    plt.show()

def rate_hist(hrc, stis, wfc, wfpc2, uvis):
    fig = plt.figure(figsize=(10,8))
    gs0 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig)
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=6,wspace=0.5, subplot_spec=gs0[0])
    gs10 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=6,wspace=0.5, subplot_spec=gs0[1])
    ax1 = fig.add_subplot(gs00[0,1:3]) 
    ax2 = fig.add_subplot(gs00[0,3:5])  
    ax3 = fig.add_subplot(gs10[0,:2])
    ax4 = fig.add_subplot(gs10[0,2:4])
    ax5 = fig.add_subplot(gs10[0,4:6])      
    axes = [ax1, ax2, ax3, ax4, ax5]
    #fig, axes = plt.subplots(nrows=, ncols=1, figsize=(6,6))
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                          '#f781bf', '#a65628', '#984ea3',
                          '#999999', '#e41a1c', '#dede00']
    labels = ['STIS/CCD','ACS/HRC', 'ACS/WFC', 'WFPC2', 'WFC3/UVIS']
    datasets = [stis, hrc, wfc, wfpc2, uvis]
    for i, (ax, label, dset) in enumerate(zip(axes, labels, datasets)):
        flags = dset.incident_cr_rate.gt(0.2) 
        cr_rate = dset['incident_cr_rate'][flags]
        mean, median, std = sigma_clipped_stats(cr_rate, sigma=3, maxiters=5)
        median = cr_rate.quantile(0.5)
        lower_20 = cr_rate.quantile(0.1)
        upper_80 = cr_rate.quantile(0.9)
        hist, edges = np.histogram(cr_rate, bins=60, range=(0, 3))
        ax.step(edges[1:], hist/np.max(hist), label=label, color=CB_color_cycle[i])
        ax.set_xticks([0,0.5, 1.0, 1.5, 2, 2.5, 3])
        ax.axvline(median, c='k', ls='-', lw=0.75, )
        ax.axvspan(lower_20, upper_80, color='gray', alpha=0.3)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.axvline(lower_20, c='k', ls='--', lw=0.75)
        ax.axvline(upper_80, c='k', ls='--', lw=0.75)
        leg = ax.legend(loc='best', edgecolor='k')
        ax.set_xlim((0,3.0))
        ax.text(1.8, 0.62,'$N_{samp}$='+'{:,}'.format(hist.sum()))
        t = ax.text(0,0,'', fontsize=14)
        ax.tick_params(axis='x', which='major', width=1.5, length=5)
        #ax.set_xlabel('Cosmic ray rate [$CR/s/cm^2$]')
    fig.text(0.38, 0.02, 'Cosmic Ray Rate [$CR/s/cm^2$]', fontproperties=t.get_font_properties())
    fig.text(0.05, 0.63, 'Normalized Bin Count', rotation='vertical', fontproperties=t.get_font_properties())
    fout = os.path.join(APJ_PLOT_DIR, 'cr_rate_hist.png')
    fig.savefig(fout, format='png', dpi=350, bbox_inches='tight')
    plt.show()


def cr_rejection_algorithm(
        flist=None,
        x=2045,
        y=1061,
        box_w=10,
        box_h=10,
        figsize=(8,6),
        fout='example_of_cr_rejection.png'
):
    if flist is None:
        flist = glob.glob(
            '/Users/nmiles/hst_cosmic_rays/APJ_plots/test_data/*flt.fits'
        )

    img_data = []
    pix_data = []
    for f in flist[:12]:
        data = fits.getdata(f)
        img_data.append(data)
        pix_data.append(data[y][x])


    fig = plt.figure(figsize=figsize)
    gs0 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, wspace=0.25)
    gs00 = gridspec.GridSpecFromSubplotSpec(
        nrows=4,
        ncols=3,
        wspace=0.05,
        hspace=0.5,
        subplot_spec=gs0[0]
    )
    fig.suptitle(
        'Comparing Pixel Values Across 12 Dark Frames',
        fontweight='bold'
    )
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
                       label=rf'median: {np.median(pix_data):.3f} $e^-$')
    scatter_ax.set_xlabel('Image Number')
    scatter_ax.set_ylabel(r'Signal [$e^-$]')
    scatter_ax.legend(loc='upper right', edgecolor='k')


    img_subplots = [
        fig.add_subplot(gs00[i,j]) for i in range(4) for j in range(3)
    ]



    norm = ImageNormalize(
        img_data[0],
        stretch=LinearStretch(),
        vmin=0,
        vmax=80
    )
    mk_patch = lambda x, y: patches.Rectangle(
        (x-0.5,y-0.5), width=1, height=1, fill=False, color='r', lw=1.25
    )
    for i, ax in enumerate(img_subplots):
        ax.grid(False)
        ax.set_title(f"{i+1}", fontsize=12)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        im = ax.imshow(img_data[i], norm=norm, cmap='gray', origin='lower')
        ax.set_xlim((x - box_w, x + box_w))
        ax.set_ylim((y - box_h, y + box_h))
        patch = mk_patch(x, y)
        ax.add_patch(patch)


    fig.savefig(os.path.join(APJ_PLOT_DIR, fout),
                format='png',
                dpi=300, bbox_inches='tight')



def periodogram(hrc, stis, wfc, wfpc2, uvis):
    fig = plt.figure(figsize=(3,4))
    ax = fig.add_subplot(111)
    gs0 = gridspec.GridSpec(ncols=1, nrows=2, figure=fig, hspace=0.25)
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=6,
                                            wspace=0.55, subplot_spec=gs0[0])
    gs10 = gridspec.GridSpecFromSubplotSpec(nrows=1, ncols=6,
                                            wspace=0.55, subplot_spec=gs0[1])
    #ax1 = fig.add_subplot(gs00[0,1:3])
    #ax2 = fig.add_subplot(gs00[0,3:5])
    #ax3 = fig.add_subplot(gs10[0,:2])
    #ax4 = fig.add_subplot(gs10[0,2:4])
    #ax5 = fig.add_subplot(gs10[0,4:6])
    axes = [ax, ax, ax, ax, ax]
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
    offset = 0.002
    factor = 0
    for i, (ax, label, dset) in enumerate(zip(axes, labels, datasets)):
        if i == 1:
            continue
        if i == 4:
            continue
        flags = dset.integration_time.gt(200)
        df = dset[['mjd','incident_cr_rate']][flags]
        smoothed = df.rolling(window='20D', min_periods=15).mean()
        smoothed_no_nan = smoothed.dropna()
        ls = LombScargle(
             smoothed_no_nan['mjd'], smoothed_no_nan['incident_cr_rate']
        )
        freq, power = ls.autopower()
        peak_range_1 = np.where((freq < 0.01) & (freq > 0.0))[0]
        max_power_idx1_range = np.argmax(power[peak_range_1])
        max_power_idx1_full = peak_range_1[max_power_idx1_range]
        max_power_freq1 = freq[max_power_idx1_full]
        max_power_period1 = 1/max_power_freq1
        peak1 += ls.false_alarm_probability(power[max_power_idx1_full])
        print(max_power_period1/365, label, ls.false_alarm_probability(power[max_power_idx1_full]), max_power_freq1)
        peak_range_2 = np.where((freq < 0.03) & (freq>0.01))[0]
        max_power_idx2_range = np.argmax(power[peak_range_2])
        max_power_idx2_full = peak_range_2[max_power_idx2_range]
        max_power_freq2 = freq[max_power_idx2_full]
        max_power_period2 = 1/max_power_freq2
        peak2 += ls.false_alarm_probability(power[max_power_idx2_full])
        print(max_power_period2, label, max_power_freq2, ls.false_alarm_probability(power[max_power_idx2_full]))
        data_out[label] = (freq, power)
        ax.plot(freq + factor*offset, power, color=CB_color_cycle[i], label=f'{label}+{factor:.0f}*offset', lw=1.5)
        ax.set_xlim(-0.0025, 0.04)
        xticks =[0, 0.01, 0.02, 0.03,  0.04]
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.set_xticks(xticks)
        ax.tick_params(axis='both',labelsize=8, which='both', color='k', labelcolor='k')
       # ax.tick_params(axis='x', which='major', width=1.5, length=5)
        #ax.set_xticklabels(xticks, rotation=45, ha='right')
        #for tick in ax.get_xticklabels():
        #    tick.set_rotation(45)
        ax.set_ylim(0, 0.5)
        for tick in ax.get_xticklabels():
            tick.set_rotation(30)
            tick.set_ha('right')
        peaks, _ = find_peaks(power, height=0.08, distance = 10)
        #for val in peaks:
        #    print(val, freq[val])
        #    try:
        #        ax.axvline(freq[val], linestyle='--', color='k', linewidth=1, alpha=0.45)
        #    except Exception:
        #        pass
        leg = ax.legend(loc='best', fontsize=8)
        leg.get_frame().set_edgecolor('k')
        t = ax.text(0,0,'',fontsize=10)
        factor +=1
    #ax.grid(linestyle='-', linewidth='0.5', color='k')
    print(f'Average FAP for peak 1 {peak1/5:}')
    print(f'Average FAP for peak 2 {peak2/5:}')
    ax.set_ylabel('Lomb-Scargle Power', fontsize=10, color='k')
    ax.set_xlabel('Frequency [cycles/day]', fontsize=10, color='k')
    #fig.text(0.32, 0.02, 'Frequency [cycles/day]', fontproperties=t.get_font_properties(), color='white')
    #fig.text(0.03, 0.63, 'Lomb-Scargle Power', fontproperties=t.get_font_properties(), rotation='vertical', color='white')
    fout = os.path.join(APJ_PLOT_DIR, 'cr_rate_periodogram_poster_stacked.png')
    fig.savefig(fout, format='png', dpi=350, bbox_inches='tight',transparent=False)
    plt.show()
    return data_out


def rate_vs_time(hrc, stis, wfc, wfpc2, uvis):
    v = visualize.Visualizer()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,
                       figsize=(5,4),
                       sharex=True)
    smooth_type = 'rolling'
    window='90D'
    min_periods=70
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
    ax1.tick_params(which='both', axis='both', color='k', labelcolor='k')
    ax2.tick_params(which='both', axis='both', color='k', labelcolor='k')
    #fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    #fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    # datasets = [stis_plot_params, wfpc2_plot_params, wfc_plot_params]
    datasets = [stis_plot_params, wfpc2_plot_params, hrc_plot_params,
                wfc_plot_params, uvis_plot_params]
    for i,dset in enumerate(datasets):
        fig, ax1 = v.plot_cr_rate_vs_time(**dset,normalize=True,min_exptime=300)

    ax1.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax1.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax2.yaxis.set_minor_locator(AutoMinorLocator(5))
    solar_df = read_solar_data()
    solar_cycle = get_solar_min_and_max(solar_df)
   
   #solar_cycle ={'maximum_cycle23':Time('2000-04-01', format='iso'),
   #              'maximum_cycle24':Time('2014-02-01', format='iso'),
   #              'minimum_cycle23':Time('1996-05-01', format='iso'),
   #              'minimum_cycle24': Time('2008-12-01', format='iso')}
    ax2.plot(solar_df.index.values, 
            solar_df['sunspot RI'], 
            label='Monthly Mean',
            c='#1E88E5')

    ax2.plot(solar_df.index.values,
             solar_df['sunspot RI smooth'],
             label='Smoothed',
             c='#D81B60')
    ax1.tick_params(labelbottom=False)
    ax1_legend = ax1.legend(loc='best',
                            ncol=3,
                            labelspacing=0.2,fontsize=8,
                            columnspacing=0.5)
    ax1_legend.get_frame().set_edgecolor('k')

    for i in range(len(ax1_legend.legendHandles)): 
        ax1_legend.legendHandles[i]._sizes = [15]

    for cycle in solar_cycle.keys():
            # Min
            ax1.axvline(solar_cycle[cycle][0], ls='--', color='k')
            ax2.axvline(solar_cycle[cycle][0],ls='--', color='k')

            # Max
            ax1.axvline(solar_cycle[cycle][1], ls='--', color='k')
            ax2.axvline(solar_cycle[cycle][1], ls='--', color='k')

    ax2_legend = ax2.legend(loc='best', fontsize=8)
    ax2_legend.get_frame().set_edgecolor('k')
    
    for tick in ax2.get_xticklabels():
        tick.set_rotation(30)
        tick.set_ha('right')
    
    date_min = Time('1990-12-01', format='iso')
    date_max = Time('2019-10-01', format='iso')
    ax1.set_xlim((date_min.to_datetime(), date_max.to_datetime()))
    ax2.set_xlim((date_min.to_datetime(), date_max.to_datetime()))
    ax2.set_ylabel('$R_I$', fontsize=10, color='k')
    ax2.set_xlabel('Date', fontsize=10,color='k')
    ax1.set_ylim(0.6, 1.4)
    ax1.set_ylabel('Mean Normalized CR Rate',fontsize=10, color='k')
    
    fout = os.path.join(APJ_PLOT_DIR,'cr_rate_vs_time_poster.png')
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

    rc('text', usetex=True)

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

    v = visualize.Visualizer()
    fig, axes = v.mk_fig(nrows=2, ncols=3, figsize=(9, 6))

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
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        cbar = fig.colorbar(data_dict_th[key]['im'], cax=cax,
                            ticks=data_dict_th[key]['cbar_ticks'],
                            orientation='horizontal')
        # cbar.ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%2.1f'))
        # cbar.ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        cbar_labels = [str(x) for x in data_dict_th[key]['cbar_ticks']]
        # cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation=45)
        cbar.ax.set_xticklabels(cbar_labels, rotation=45, fontsize=10)

        cbar.set_label(r'Thickness $[\mu m]$')
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
        cax = divider.append_axes('bottom', size='5%', pad=0.05)
        cbar = fig.colorbar(data_dict_cr[key]['im'], cax=cax,
                            ticks=data_dict_cr[key]['cbar_ticks'],
                            orientation='horizontal')
        cbar_labels = [str(x) for x in data_dict_cr[key]['cbar_ticks']]
        cbar.ax.set_xticklabels(cbar_labels, rotation=45, fontsize=10)
        cbar.set_label(r'Number of CR Strikes')

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
    fout = os.path.join(APJ_PLOT_DIR, 'cr_th_all_instr.png')
    fig.savefig(fout,
                 transparent=True,
                 format='png',
                 dpi=350,
                 bbox_inches='tight')
    plt.show()

def plot_example_dark():
    pass


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


def plot_solar_cycle(smoothed=True, figsize=(5,4), save=True):
    v = visualize.Visualizer()
    noaa_df = read_solar_data()
    solar_cycles = get_solar_min_and_max(noaa_data=noaa_df)
    fig, ax = v.mk_fig(nrows=1, ncols=1, figsize=figsize)
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    ax.plot(noaa_df.index.values,
            noaa_df['sunspot RI'],
            label='Monthly Mean',
            c='#1E88E5')
    ax.plot(noaa_df.index.values,
            noaa_df['sunspot RI smooth'],
            label='Smoothed',
            c='#D81B60')
    date_min = Time('1990-12-01', format='iso')
    date_max = Time('2019-03-01', format='iso')


    # Add vertical lines for cycle 23 minima and shade between
    cycle23_start = Time(solar_cycles['Cycle 23'][0]).to_datetime()
    cycle23_max = Time(solar_cycles['Cycle 23'][1]).to_datetime()

    cycle23_end = Time(solar_cycles['Cycle 24'][0]).to_datetime()
    cycle24_max = Time(solar_cycles['Cycle 24'][1]).to_datetime()

    mpl.rcParams['hatch.linewidth'] = 0.2
    ax.axvspan(cycle23_start, cycle23_end,
               facecolor='k',
               alpha=0.2,
               )


    ax.text(cycle23_max + timedelta(days=2*365), 150, s='Cycle 23',
            fontsize=12)

    predicted_cycle24_end = Time('2019-06-01', format='iso').to_datetime()
    ax.axvspan(cycle23_end, predicted_cycle24_end,
               facecolor='r',
               alpha=0.2,
               )
    ax.text(cycle24_max - timedelta(days=365), 150, s='Cycle 24',
            fontsize=12)

    operational_coverage_start = Time('1994-01-01', format='iso').to_datetime()
    ax.axvspan(operational_coverage_start, predicted_cycle24_end,
               facecolor='r',
               alpha=0.0,hatch='/'
               )

    ax.set_xlim((date_min.to_datetime(), date_max.to_datetime()))
    ax.set_xlabel('Date')
    ax.set_ylabel('$R_I$')
    leg = ax.legend(loc='best')
    ax.set_ylim(0, noaa_df['sunspot RI'].max() + 30)

    # ax.set_title('International Sunspot Number')
    if save:
        fout = os.path.join(APJ_PLOT_DIR, 'solar_cycle.png')
        fig.savefig(fout,format='png', dpi=300,bbox_inches='tight')
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=20, ha='left')


def main():
    pass


if __name__ == '__main__':
    main()
