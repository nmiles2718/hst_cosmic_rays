#!/usr/bin/env python

from astropy.io import fits
from astropy.visualization import LinearStretch, ZScaleInterval
from astropy.visualization.mpl_normalize import ImageNormalize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage


def main(fname):
    with fits.open(fname) as hdu:
        data = hdu[0].data
    smoothed_data = ndimage.filters.gaussian_filter(data,
                                                    sigma=3,
                                                    mode='nearest')
    norm = ImageNormalize(data,
                          stretch=LinearStretch(),
                          interval=ZScaleInterval())
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,5))
    ax1.set_title('Cosmic Ray Incidence Map')
    ax2.set_title('Smoothed Cosmic Ray Incidence Map')
    im1 = ax1.imshow(data, norm=norm, cmap='gray', origin='lower')
    im2 = ax2.imshow(smoothed_data, norm=norm, cmap='gray', origin='lower')
    divider = make_axes_locatable(ax2)
    for ax in [ax1, ax2]:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    cax = fig.add_axes([0.2, 0.15, 0.6, 0.04])
    cbar = fig.colorbar(im2, cax=cax, orientation='horizontal')
    cbar.set_label('CR Incidences', fontweight='bold')
    fig.savefig('cr_incidence_plot_gauss5sig_smoothed_ACS_WFC.png',
                format='png',
                dpi=350,
                bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    main()