#!/usr/bin/env python

import glob
from astropy.time import Time
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve, convolve_fft, Box2DKernel
from astropy.visualization import LinearStretch, ZScaleInterval,\
    AsinhStretch, SqrtStretch, ImageNormalize
import costools
from itertools import chain
import numpy as np

from matplotlib import rc
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.basemap import Basemap
import pandas as pd
import plot_stats as ps
from PIL import Image

from sunpy.timeseries import TimeSeries
from scipy.ndimage import median_filter



def read_cr_data():
    flist_wfc = glob.glob('./../data/ACS/acs_cr_rate_?.hdf5')
    flist_hrc = glob.glob('./../data/ACS/acs_hrc_cr_rate_?.hdf5')
    flist_wfpc2 = glob.glob('./../data/WFPC2/wfpc2_cr_rate_?.hdf5')
    flist_wfc3 = glob.glob('./../data/WFC3/wfc3_cr_rate_?.hdf5')
    flist_stis = glob.glob('./../data/STIS/stis_cr_rate.hdf5')

    ccd_data_in = {'ACS_WFC':flist_wfc,
               'ACS_HRC':flist_hrc,
               'WFPC2':flist_wfpc2,
               'WFC3_UVIS': flist_wfc3,
               'STIS_CCD':flist_stis
            }
    data_obj = {}
    for key in ccd_data_in.keys():
        print(key)
        data_obj[key] = ps.PlotData(fname=None,
                                    flist=ccd_data_in[key],
                                    instr=key,
                                    subgrp='incident_cr_rate')
        data_obj[key].read_rate()
    return data_obj


def exptime_comparison(instrument_data):
    exptime_data = {}
    for key in instrument_data.keys():
        exptime_data[key] = instrument_data[key].data_df['exptime']
    df = pd.DataFrame(exptime_data)
    sampled = df.resample(rule='120D').sum()
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(9,7))

    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                      '#f781bf', '#a65628', '#984ea3',
                      '#999999', '#e41a1c', '#dede00']

    for i, col in enumerate(sampled.columns):
        no_nan = sampled[col].dropna()

        ax.semilogy(no_nan.index.values,
                no_nan,drawstyle='steps-mid',
                label='{}'.format(col.replace('_','/')),
                color=CB_color_cycle[i], fillstyle='full')
        print(col)
        ax.fill_between(no_nan.index.values,
                        no_nan,
                        0,
                        step='mid',
                        color=CB_color_cycle[i])
    ax.legend(loc='best')

    ax.set_ylabel('Cumulative Exposure Time [s]')
    ax.set_title('Temporal Coverage of Dark Exposures')
    ax.set_ylim(1e3, 1e7)
    fig.savefig('cumulative_exptime.png', format='png', dpi=350)
    plt.show()

    return no_nan


def ccd_substrate_model():
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 10))
    ax.set_facecolor('white')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.text(1, 17,'ACS/WFC CCD Substrate Layers', fontsize='x-large')



    # top layer
    text_x = 8
    ax.text(6, 12, 'CCD Housing Environment', fontsize='large')

    arrow_w=0.05

    ax.text(0, 15.2,'Incoming Light')
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
    fig.savefig('ccd_substrate_example.png',
                format='png',
                dpi=350,
                bbox_inches='tight',
                transparent=False,
                frameon=False)
    plt.show()
    # rc(useTex=False)



def thickness_plot(fname=None, fname_comp=None, fout=None, instr=None):
    """

    Parameters
    ----------
    fname
    fname_comp

    Returns
    -------

    """
    rc('text', usetex=True)
    if 'fits' in fname_comp:
        comp_data = fits.getdata(fname_comp)
        astrofits=True
    else:
        comp_data = Image.open(fname_comp)
        astrofits=False

    with fits.open(fname) as hdu:
        data = hdu[0].data

    # mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    smoothed = median_filter(data, size=5)
    # norm = ImageNormalize(data,
    #                       stretch=LinearStretch(),
    #                       vmin=np.min(data), vmax=np.max(data))

    uvis = (140, 240)
    wfc = (50, 110)
    hrc = (130,200)
    norm = ImageNormalize(data,
                          stretch=LinearStretch(),
                          vmin=hrc[0], vmax=hrc[1])
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11,6))
    im1 = ax1.imshow(smoothed, norm=norm, cmap='gray', origin='lower')

    # Add a colorbar to show the image scaling
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes('bottom', size='5%', pad=0.1)
    cbar1 = fig.colorbar(im1, cax=cax1, orientation='horizontal')
    cbar1.ax.set_xticklabels(cbar1.ax.get_xticklabels(), rotation=45)
    cbar1.set_label('Cosmic Ray Strikes')
    if not astrofits:
        norm1 = ImageNormalize(comp_data,
                               stretch=LinearStretch(),
                               interval=ZScaleInterval())
        im2 = ax2.imshow(comp_data, norm=norm1, cmap='gray')
    else:
        norm1 = ImageNormalize(comp_data,
                               stretch=LinearStretch(),
                               vmin=12.49, vmax=16)
        im2 = ax2.imshow(comp_data, cmap='gray', norm=norm1, origin='lower')
    # Add a colorbar to show the image scaling
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes('bottom', size='5%', pad=0.1)
    cbar2 = fig.colorbar(im2, cax=cax2, orientation='horizontal')
    cbar2.ax.set_xticklabels(cbar2.ax.get_xticklabels(), rotation=45)
    cbar2.set_label(r'Thickness $[\mu m]$')
    ax1.grid(False)
    ax2.grid(False)
    ax1.set_title('Cosmic Ray Incidence Heat Map')
    ax2.set_title('Fringing Thickness Map')
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    fig.suptitle(instr,
                 x=0.5, y=0.95,
                 horizontalalignment='center',
                 size=16, weight='bold')
    fig.savefig(fout, format='png', dpi=350)
    plt.show()

def read_solar_data():
    noaa = TimeSeries('./../data/RecentIndices.txt', source='NOAAIndices')
    df = noaa.to_dataframe()
    return df


def stis_saa_plot(instrument_data, i=5):



    stis = instrument_data['STIS_CCD'].data_df['1997-02-01':'1997-02-28']

    saa_eastern = (39.0, -30.0)  # lon/lat
    saa_western = (267.0, -20.0)
    saa_northern = (312.0, 1.0)
    saa_southern = (300.0,-60.0)

    mask = (stis['latitude'] < saa_northern[1]) #& (stis['incident_cr_rate'] < 20)

    stis_saa_cut = stis[mask]
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(9,8))

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

    hst_lon, hst_lat = stis_saa_cut['longitude'], stis_saa_cut['latitude']
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
            m.scatter(lon, lat,
                    marker='o', s=10,c='r',
                    latlon=True)

            ax1.annotate('{}'.format(j-4),
                         xy=(x_coord[j], y_coord[j]),
                         xycoords='data')

    ax2.scatter([k - 4 for k in indices],
                stis_saa_cut['incident_cr_rate'][indices])
    ax1.legend(loc='best')
    ax2.set_xticks([k - 4 for k in indices])
    ax2.set_ylabel('Cosmic Ray Flux [CR/s/cm^2]')
    ax2.set_xlabel('Observation Number')
    fig.savefig('stis_saa_crossing.png', format='png', dpi=350)
    plt.show()
    # with open('stis_saa_darks.txt', 'a') as fobj:
    #     for indx in indices:
    #         fobj.write('{}\n'.format(stis_saa_cut['date'][indx]))



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


def plot(cr_data, noaa_data):
    grid = plt.GridSpec(2, 1, wspace=0.1, hspace=0.25)
    fig = plt.figure(figsize=(9, 10))
    ax1 = fig.add_subplot(grid[0, 0])
    ax2 = fig.add_subplot(grid[1, 0], sharex=ax1)

    for i, key in enumerate(cr_data.keys()):
        # if key == 'WFPC2':
        #     continue
        mask = cr_data[key].perform_SAA_cut()
        cr_data[key].plot_rate_vs_time(window='120D',
                                       min_periods=100,
                                       ax=ax1,
                                       i=i, saa_exclude=False)
    ax2.plot(noaa_data.index.values, noaa_data['sunspot RI'], label='Daily RI Sunspot Number')
    ax2.plot(noaa_data.index.values,
             noaa_data['sunspot RI smooth'],
             label='Smoothed RI Sunspot Number')

    solar_cycle = get_solar_min_and_max(noaa_data)
    print(solar_cycle)
    ax1_legend = ax1.legend(loc='best',
                            ncol=3,
                            labelspacing=0.2,
                            columnspacing=0.5)
    for i in range(len(ax1_legend.legendHandles)):
        ax1_legend.legendHandles[i]._sizes = [30]

    for cycle in solar_cycle.keys():
        # Min
        ax1.axvline(solar_cycle[cycle][0], ls='--', color='k')
        ax2.axvline(solar_cycle[cycle][0],ls='--', color='k')

        # Max
        ax1.axvline(solar_cycle[cycle][1], ls='--', color='k')
        ax2.axvline(solar_cycle[cycle][1], ls='--', color='k')

    ax2_legend = ax2.legend(loc='best')
    date_min = Time('1991-12-01', format='iso')
    date_max = Time('2019-01-01', format='iso')
    ax1.set_xlim((date_min.to_datetime(), date_max.to_datetime()))
    ax2.set_title('International Sunspot Number')
    ax2.set_ylabel('Number of Sunspots')
    fig.savefig('cr_rate_vs_time.png',format='png',dpi=350)
    plt.show()

def main():
    data = read_cr_data()
    noaa_data = read_solar_data()
    plot(data, noaa_data)

if __name__ == '__main__':
    main()