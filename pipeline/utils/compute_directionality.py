#!/usr/bin/env python
"""
Objectives for this script:

1) Generate labels for cosmic rays
2) Compute the centroids of the cosmic rays
3) Compute the centroids of the labels
4) Generate 20 x 20 pixel cutouts showing the labels, the SCI data,
   and the two centroids

"""
import os
import sys
sys.path.append('/Users/nmiles/hst_cosmic_rays/pipeline')

from astropy.table import Table

from astropy.visualization import ImageNormalize, LinearStretch, \
    ZScaleInterval, SqrtStretch, LogStretch


import matplotlib as mpl
# mpl.use('Qt5Agg')
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as patch

from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec


from label import labeler
from stat_utils import statshandler

import numpy as np
import scipy.ndimage as ndimage


def save_to_pdf(data1, sources1, data2, sources2, dirname=None):
    """ Save the results of the matched stars to a multipage pdf

    Using the common sources, save 40 by 40 pixel cutouts of each matched
    source. There will be a total of 16 cutouts per page, that are split
    into two groups of 8. Within each group of 8, the matching sources will
    be displayed column-wise. The top row of 4 stars will all be distinct
    and their matches will be displayed in the row immediately below.

    Parameters
    ----------
    data1 : numpy.ndarray
        FITS data from image 1
    sources1 : astropy.table.Table
        Catalog of matched sources found in image 1
    data2 : numpy.ndarray
        Label for all cosmic rays in data1
    sources2 : astropy.table.Table
        Catalog of matches sources found in image 2
    dirname :
        Path to the directory containing the images that were analyzed

    Returns
    -------
    None
        This function will save a pdf to the results directory containing
        the matched stars used in the analysis
    """

    ncolors = np.max(data2) + 1
    prng = np.random.RandomState(1234)
    h = prng.uniform(low=0.0, high=1.0, size=ncolors)
    s = prng.uniform(low=0.2, high=0.7, size=ncolors)
    v = prng.uniform(low=0.5, high=1.0, size=ncolors)
    hsv = np.dstack((h, s, v))

    rgb = np.squeeze(colors.hsv_to_rgb(hsv))
    rgb[0] = (0, 0, 0)
    cmap = colors.ListedColormap(rgb)

    mk_patch = lambda xy, r, c, lw, fill: patch.Circle(xy=xy,
                                             radius=r,
                                             color=c,
                                             fill=fill,
                                             lw=lw)
    norm = ImageNormalize(data1,
                          stretch=LogStretch(),
                          vmin=0,
                          vmax=1000)
    # interval=ZScaleInterval())

    outname = 'centroid_comparison.pdf'
    print('Total number of sources {}'.format(len(sources1)))
    with PdfPages(outname) as pdf:
        num_pages = int(np.ceil(len(sources1) / 16))
        # num_pages = 20
        start_idx = 0
        for i in range(num_pages):
            start_idx += 8
            sources1_to_plot = sources1[start_idx: start_idx + 8]
            sources2_to_plot = sources2[start_idx: start_idx + 8]
            # Initalize the plot, each axes list will contain 8 plots
            # The plots should be organize by columns, i.e. two plots in
            # the same column correspond to the same image
            fig, axes00, axes10 = mk_grid()



            # i will run from 0 to 15 (i.e. 16 elements)
            for j in range(len(sources1_to_plot)):
                cr1 = sources1_to_plot[j]
                cr2 = sources2_to_plot[j]
                # limits for the first star
                # print(j)
                if j < 4:
                    # print(j%4)
                    ax1 = axes00[j]
                else:
                    # print(j%4)
                    ax1 = axes10[j % 4]

                cutout_size = 40
                # Limits for the first plot
                xlimits1 = cr1['xcenter'] - cutout_size / 2, \
                           cr1['xcenter'] + cutout_size / 2
                ylimits1 = cr1['ycenter'] - cutout_size / 2, \
                           cr1['ycenter'] + cutout_size / 2

                if j < 4:
                    # print(j%4 + 4)
                    ax2 = axes00[j % 4 + 4]
                else:
                    # print(j%4 + 4)
                    ax2 = axes10[j % 4 + 4]

                # Limits for the second plot
                xlimits2 = cr2['xcenter'] - cutout_size/2, \
                           cr2['xcenter'] + cutout_size/2
                ylimits2 = cr2['ycenter'] - cutout_size/2, \
                           cr2['ycenter'] + cutout_size/2



                im1 = ax1.imshow(data1, norm=norm, cmap='gray', origin='lower')
                im2 = ax2.imshow(data2, cmap=cmap, origin='lower')

                # Add axes for color bar to show scale
                divider = make_axes_locatable(ax1)
                cax1 = divider.append_axes("right", size="8%", pad=0.05)


                cbar = fig.colorbar(im1, cax=cax1)
                n = len(cbar.ax.get_yticklabels())

                labels_to_hide = [n-3, n - 2]
                loop = zip(cbar.ax.get_yticklabels(),
                           cbar.ax.yaxis.get_major_ticks())
                for i, (label, tick) in enumerate(loop):
                    if i in labels_to_hide:
                        label.set_visible(False)
                        tick.set_visible(False)

                cbar.ax.set_yticklabels(cbar.ax.get_yticklabels(),
                                             rotation=10,
                                             horizontalalignment='left',
                                             verticalalignment='center',
                                             fontsize=5
                                             )
                cbar.update_ticks()



                flux_max_patch = mk_patch((cr1['xmax'], cr1['ymax']),
                                          r=0.5,
                                          c='magenta',
                                          lw=1,
                                          fill=True)
                ax1.add_patch(flux_max_patch)

                flux_center_patch = mk_patch((cr1['xcenter'], cr1['ycenter']),
                                             r=0.5,
                                             c='red',
                                             lw=1.,
                                             fill=True)
                ax1.add_patch(flux_center_patch)
                geo_center_patch = mk_patch((cr2['xcenter'], cr2['ycenter']),
                                            r=0.5,
                                            c='blue',
                                            lw=1.,
                                            fill=True)

                ax1.add_patch(geo_center_patch)

                current_label_patch = mk_patch((cr2['xcenter'], cr2['ycenter']),
                                               r=4,
                                               c='white',
                                               lw=1.5,
                                               fill=False)


                ax2.add_patch(current_label_patch)

                ax1.set_title('Red: flux-weight\n '
                              'Blue: uniform-weight \n'
                              'Magneta: max value',
                              fontsize='medium')
                ax2.set_title('Label', fontsize='medium')

                # Set the plot limits for star 1
                ax1.set_xlim(xlimits1[0], xlimits1[1])
                ax1.set_ylim(ylimits1[0], ylimits1[1])

                # Set the plot limits for star 2
                ax2.set_xlim(xlimits2[0], xlimits2[1])
                ax2.set_ylim(ylimits2[0], ylimits2[1])

                ax1.grid(False)
                ax2.grid(False)
            # add colorbar to outer grid

            pdf.savefig(fig)
            plt.close()


def mk_grid():
    """Convenience function for making a grid of axes to plot 16 images cuts

    Returns
    -------

    """
    fig = plt.figure(figsize=(10, 8))
    gs0 = gridspec.GridSpec(nrows=2, ncols=1, hspace=0.45)
    gs00 = gridspec.GridSpecFromSubplotSpec(nrows=2,
                                            ncols=4,
                                            subplot_spec=gs0[0],
                                            wspace=0.35, hspace=0.5)

    gs01 = gridspec.GridSpecFromSubplotSpec(nrows=2,
                                            ncols=4, subplot_spec=gs0[1],
                                            wspace=0.35, hspace=0.5)

    axes00 = []
    axes10 = []
    for i in range(2):
        for j in range(4):
            axes00.append(fig.add_subplot(gs00[i, j]))
            axes10.append(fig.add_subplot(gs01[i, j]))
    return fig, axes00, axes10


def main():
    fname = '/Users/nmiles/hst_cosmic_rays/crrejtab/ACS/mastDownload/HST/j8ba20foq/j8ba20foq_flt.fits'

    instr='ACS_WFC'
    cr_label = labeler.CosmicRayLabel(fname=fname)
    cr_label.get_data(ext='dq')
    cr_label.get_data(ext='sci')
    cr_label.get_label()
    fig, ax1, ax2 = cr_label.plot(show=False)

    stats_obj = statshandler.ComputeStats(fname=fname,
                             instr=instr,
                             label=cr_label.label,
                             sci=cr_label.sci,
                             integration_time=1000.0)



    # Get the sizes of each cosmic ray
    cr_sizes = stats_obj.compute_size()
    sizes_in_sig = list(cr_sizes.values())
    sizes_in_sig = np.asarray(sizes_in_sig)
    large_crs_sig = np.where(sizes_in_sig > 0.75)

    # Get the size in pixels of each cosmic ray
    cr_label_ids = cr_label.label.ravel()  # Returns a flattened label

    # get the unique labels
    cr_label_idx = np.unique(cr_label_ids)[1:]

    # Count up the number of pixels associated with each unique label
    sizes_in_pixels = np.bincount(cr_label_ids)[1:]
    # arg_max = np.argmax(sizes_in_pixels)
    # largest object is the background, set it to 0
    # sizes_in_pixels[arg_max] = 0
    large_crs_pix = sizes_in_pixels > 30

    # Compute the position where the most energy is deposited
    maxima = ndimage.maximum_position(cr_label.sci,
                                               labels=cr_label.label,
                                               index=cr_label_idx)
    maxima = np.asarray(maxima)

    # for idx in cr_label_idx:
    #     maxima.append(ndimage.maximum_position(cr_label.sci,
    #                                            labels=cr_label.label,
    #                                            index=idx))


    # Compute the flux weighted centroids,
    # Format is ([col1, row1], ...,[col2, row2])
    flux_weighted_centroid = stats_obj.compute_first_moment()

    flux_weighted_centroid = flux_weighted_centroid[large_crs_pix]
    maxima = maxima[large_crs_pix]



    flux_weight_table = Table([flux_weighted_centroid[:, 1],
                               flux_weighted_centroid[:, 0],
                               maxima[:, 1],
                               maxima[:, 0]],
                              names=['xcenter','ycenter', 'xmax','ymax'])


    # Compute the uniformly weighted centroids (i.e. geometric centers)
    uniform_weight = np.ones(shape=cr_label.sci.shape)

    geometric_centroid = stats_obj.compute_first_moment(sci=uniform_weight)
    geometric_centroid = geometric_centroid[large_crs_pix]

    uniform_weight_table = Table([geometric_centroid[:, 1],
                                  geometric_centroid[:, 0]],
                                 names=['xcenter', 'ycenter'])

    save_to_pdf(cr_label.sci, flux_weight_table,
                cr_label.label, uniform_weight_table)








if __name__ == '__main__':
    main()