#!/usr/bin/env python

from datetime import timedelta
import glob
import os
from astropy.io import fits
from astropy.time import Time
from astropy.visualization import LinearStretch, ZScaleInterval,\
    AsinhStretch, SqrtStretch, ImageNormalize


from matplotlib import rc
from matplotlib import ticker
import matplotlib as mpl
mpl.use('qt5agg')
# rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from mpl_toolkits.axes_grid1 import make_axes_locatable


import sunpy.net
from sunpy.timeseries import TimeSeries
from scipy.ndimage import median_filter, gaussian_filter

import visualize



APJ_PLOT_DIR = '../../APJ_plots/'

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


def rate_vs_time(hrc, stis, wfc, wfpc2, uvis):
    v= visualize.Visualizer()
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1,
                       figsize=(10,8),
                       sharex=False)
    smooth_type = 'rolling'
    window='120D'
    min_periods=80

    stis_plot_params = {
        'df': stis,
        'legend_label': 'STIS/CCD',
        'ax': ax1,
        'i':0,
        'smooth_type': smooth_type,
        'window': window,
        'min_periods': min_periods
    }
    hrc_plot_params = {
        'df': hrc,
        'legend_label': 'ACS/HRC',
        'ax': ax1,
        'i':1,
        'smooth_type': smooth_type,
        'window': window,
        'min_periods': min_periods
    }
    wfc_plot_params = {
        'df': wfc,
        'legend_label': 'ACS/WFC',
        'ax': ax1,
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
        'smooth_type': smooth_type,
        'window': window,
        'min_periods': min_periods
    }
    uvis_plot_params = {
        'df': uvis,
        'legend_label': 'WFC3/UVIS',
        'ax': ax1,
        'i':4,
        'smooth_type': smooth_type,
        'window': window,
        'min_periods': min_periods
    }
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    # datasets = [stis_plot_params, wfpc2_plot_params, wfc_plot_params]
    datasets = [stis_plot_params, wfpc2_plot_params, hrc_plot_params,
                wfc_plot_params, uvis_plot_params]
    for dset in datasets:
        fig, ax1 = v.plot_cr_rate_vs_time(**dset)



    solar_df = read_solar_data()
    solar_cycle = get_solar_min_and_max(solar_df)
    ax2.plot(solar_df.index.values, 
            solar_df['sunspot RI'], 
            label='Monthly Mean',
            c='#1E88E5')

    ax2.plot(solar_df.index.values,
             solar_df['sunspot RI smooth'],
             label='Smoothed',
             c='#D81B60')

    

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
    date_max = Time('2019-04-01', format='iso')
    ax1.set_xlim((date_min.to_datetime(), date_max.to_datetime()))

    ax2.set_ylabel('$R_I$', fontsize=14)
    ax2.set_xlabel('Date', fontsize=14)
    fout = os.path.join(APJ_PLOT_DIR,'cr_rate_vs_time.png')
    fig.savefig(fout, format='png', dpi=350, bbox_inches='tight')
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