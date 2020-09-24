#!/usr/bin/env python

from collections import defaultdict
from collections import Iterable
import datetime as dt
import logging
from itertools import chain
from datetime import timedelta
import glob
import os
import sys
sys.path.append('/ifs/missions/projects/plcosmic/hst_cosmic_rays/pipeline/')
import time

import astropy.units as u
import astropy.constants as physical_constants
from astropy.stats import sigma_clip, sigma_clipped_stats, gaussian_sigma_to_fwhm
from astropy.timeseries import LombScargle
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS
from astropy.visualization import LinearStretch, ZScaleInterval,\
    LogStretch, SqrtStretch, ImageNormalize
# import costools
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import dask
import dask.array as da
import dask.dataframe as ddf
import h5py
import iminuit
from iminuit.cost import LeastSquares
import matplotlib as mpl
mpl.rcParams['hatch.linewidth'] = 0.6
import dask.array as da
import dask.dataframe as ddf
from matplotlib import rc
import matplotlib.dates as mdates
from matplotlib.legend import Legend
from matplotlib.dates import DateFormatter
from matplotlib import ticker
import matplotlib as mpl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
# mpl.use('qt5agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.gridspec as gridspec
plt.style.use('ggplot')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import  inset_axes, zoomed_inset_axes
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import pmagpy.ipmag as ipmag
from pylandau import landau, langau
import statsmodels.api as sm
import sunpy.net
from sunpy.timeseries import TimeSeries
import scipy.ndimage as ndimage
from scipy.ndimage import median_filter, gaussian_filter
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import LeaveOneOut, KFold
from tqdm import tqdm

# Local packages
import datahandler as dh


OPTIMAL_BANDWIDTHS = {
    'ACS/WFC': 2.25,
    'ACS/HRC': 4.75, # took 35 minutes for 5-fold cross-validation
    'STIS/CCD': 2.25,
    'WFPC2': 1
}

def compute_track_angle(label, box, idx):
    yslice, xslice = box
    tan_theta = (yslice.stop - yslice.start)/(xslice.stop - xslice.start)
    theta = np.arctan(tan_theta) * 180/np.pi
    label_coords = np.where(label[box]== idx)
    label_coords = list(zip(label_coords[0], label_coords[1]))
    label_coords.sort(key=lambda val: val[1])
    label_coords = np.array(label_coords)

    avg_yediff1d = np.sum(np.ediff1d(label_coords[:, 0]))/ len(label_coords[:, 0])

    if avg_yediff1d < 0:
        theta = 360 - theta
        
    return theta


def compute_track_length(box):
    yslice, xslice = box
    return np.sqrt((xslice.stop - xslice.start)**2 + (yslice.stop - yslice.start)**2)


def compute_angle_of_incidience(track_length, thickness, pixel_size):
    tan_phi = thickness/(pixel_size * track_length)
    return 90 - np.arctan(tan_phi)*180/np.pi

def parallel_track_params(label, box, i, thickness, pixel_size):
    tl = compute_track_length(box)
    aoi = compute_angle_of_incidience(tl, thickness, pixel_size)
    trka = compute_track_angle(label, box, i+1)
    return tl, aoi, trka

def find_peaks(hist, bins, break_point):
    bins = 0.5*(bins[:-1] + bins[1:])
    bins_flag1 = bins < break_point
    bins_flag2 = bins > break_point
    hist_l = hist[bins_flag1]
    bins_l = bins[bins_flag1]
    bins_u = bins[bins_flag2]
    hist_u = hist[bins_flag2]
    peak1 = bins_l[np.argmax(hist_l)]
    print(f'Peak 1 {peak1}')
    peak2 = bins_u[np.argmax(hist_u)]
    print(f'Peak 2 {peak2}')
    return peak1, peak2

def compute_track_params(cr_pixels, pixel_size, thickness, detector_shape):
    try:
        x_coords = np.array(cr_pixels)[:, 1].astype(np.int16)
    except IndexError as e:
        return
    y_coords = np.array(cr_pixels)[:, 0].astype(np.int16)
    array = np.zeros(shape=detector_shape, dtype=np.int16)
    try:
        array[y_coords, x_coords] = 1
    except IndexError as e:
        return 
    label, num_sources = ndimage.label(array, structure=np.ones((3,3)))
    box_slices = ndimage.find_objects(label)
    
    track_lengths = []
    track_angles = []
    track_aois = []
    delayed_obj = [
        dask.delayed(parallel_track_params(label, box, i, thickness, pixel_size) 
                     for i, box in enumerate(box_slices))
    ]
    results = np.asarray(dask.compute(*delayed_obj, schedulers='threads', num_workers=os.cpu_count())[0])

    track_lengths = results[:, 0]
    track_aois = results[:, 1]
    track_angles = results[:, 2]       
    return label, num_sources, np.array(track_lengths), np.array(track_angles), np.array(track_aois)


def get_energy_bins(
    path_length, 
    energy_deposited,
    path_intervals = [],
    nbins=125,
    hrange=(0,500),
    density=False
):
    binned_energies = {}   
    keys = [f'interval{i+1}' for i in range(len(path_intervals))]
    for interval, key in zip(path_intervals, keys):
        path_cut = (path_length >= interval[0]) & (path_length <= interval[1])
        if density:
            hist, edges = np.histogram(energy_deposited[path_cut].value, range=hrange, bins=nbins, density=density)
        else: 
            hist, edges = np.histogram(energy_deposited[path_cut], range=hrange, bins=nbins, density=density)
        centers = 0.5*(edges[1:] + edges[:-1])
        binned_energies[key] = (hist, centers)
        binned_energies[f"{key}_range"] = interval
        binned_energies[f"{key}_nsources"] = len(energy_deposited[path_cut])
    return binned_energies


def determine_bw(
    path_length,
    energy_deposited,
    interval=(280*u.micrometer, 300*u.micrometer), 
    E_cut=1000*u.keV,
    nsubsets=2
):
    st = time.time()
    path_cut = (path_length >= interval[0]) & (path_length < interval[1])
    energy_path_cut = energy_deposited[path_cut]
    # Trim off the high energy tail to reduce biasing from rare collisions
    data_subset = energy_path_cut[energy_path_cut < E_cut] 
    kde = KernelDensity(kernel='gaussian')
    grid = GridSearchCV(
        kde, 
        param_grid={'bandwidth':np.linspace(0, 60, 61)},
        cv=KFold(nsubsets),
        verbose=1,
        n_jobs=-1
    )
    grid.fit(data_subset[:, np.newaxis])
    et = time.time()
    print(f'Time to evaluate grid: {(et - st)/60:0.3f} minutes')
    return grid
    

def perform_KDE(
    path_length, 
    energy_deposited,
    path_intervals=[],
    kernel='gau',
    bw=4,
    E_cut=500 *u.keV
):
    kde_energies = {}
    keys = [f'interval{i+1}' for i in range(len(path_intervals))]
    
    # Configure the settings for the max-likelihood, cross-eval estimator
    estimator_settings = sm.nonparametric.EstimatorSettings(
            efficient=True, randomize=True, 
            n_res=1000, n_sub=300, return_median=True, 
            return_only_bw=False, n_jobs=-1
        )
    for interval, key in zip(path_intervals, keys):
        path_cut = (path_length >= interval[0]) & (path_length < interval[1])
        energy_path_cut = energy_deposited[path_cut]
        # Trim off the high energy tail to reduce biasing from rare collisions
        data_subset = energy_path_cut[energy_path_cut < E_cut]
#         data_subset = energy_path_cut
#         hist, edges = np.histogram(energy_deposited[path_cut], range=(0, 980), bins=250, density=False)
        ml_bw = sm.nonparametric.KDEMultivariate(
            data_subset, 
            'c', 
            bw = 'cv_ml', 
            defaults=estimator_settings).bw

        print(f'Optimal BW from statsmodel: {ml_bw}')
        print(f'Optimal BW from sklearn: {bw}')
        kde = sm.nonparametric.KDEUnivariate(energy_deposited[path_cut])
        kde.fit(bw=ml_bw, kernel=kernel, cut=0)
        kde_energies[key] = kde
        kde_energies[f"{key}_range"] = interval
        kde_energies[f"{key}_ml_bw"] = ml_bw
        kde_energies[f"{key}_nsources"] = len(energy_deposited[path_cut])
    return kde_energies


def auto_fit_hist(
    dout, 
    uselandau=False,
    path_intervals=[(280*u.micrometer, 300*u.micrometer)], 
    density=True
):
    """Run the fitting routine and auto compute estimated values"""
    
    binned_energies = get_energy_bins(
        dout['track_path_length'],
        dout['energy_deposited'],
        path_intervals=path_intervals,
        density=density,
        nbins=60,
        hrange=(0,300)
    )
    # Compute an esimtate of xi
    z=1
    Z=14
    A=28.0855*u.g/u.mol
    v = 0.7*physical_constants.c
    c = physical_constants.c
    K = 0.307075 * u.megaelectronvolt * u.cm**2 / u.mol
    m_e = physical_constants.m_e
    beta = 0.75
    estimated_path = 290 * u.micrometer
    si_density = 2.329 * u.g/u.cm**3
    xi = K/2 * Z/A * z**2 *  1/(beta)**2 * si_density *estimated_path
    xi = xi.to('keV')
    
    # Run the fit on only one of the intervals
    hist = binned_energies['interval1'][0]
    # xbins corresponds to the bin centers
    xbins = binned_energies['interval1'][1]
    nsources = binned_energies['interval1_nsources']

    if density:
        mpv_guess = xbins[hist.argmax()]
    else:
        mpv_guess = xbins[hist.argmax()].value
    limit_mpv = (0.5*mpv_guess, 2*mpv_guess)
    
    xi_guess = xi.value
    limit_xi = (1, 3*xi_guess)
    
    sigma_guess = 10
    limit_sigma = (0, 3*sigma_guess)
    try:
        fit_results = fit_energy_distribution(
            xbins,
            hist,
            mpv_guess = mpv_guess, 
            limit_mpv = (0, None),
            xi_guess = xi_guess,
            limit_xi = limit_xi,
            sigma_guess = sigma_guess,
            limit_sigma = limit_sigma,
            uselandau=uselandau,
        )
    except KeyError as e:
        fit_results=None
    return xbins, hist, fit_results, nsources


def auto_fit_kde(
    dout, 
    uselandau=False, 
    path_intervals=[(280*u.micrometer, 300*u.micrometer)], 
    density=True,
    kde_scaling=1e5,
    label=None
):
    """Run the fitting routine and auto compute estimated values"""
    if label is not None:
        bw = OPTIMAL_BANDWIDTHS[label]
    
    # Generate the KDE distribution
    binned_energies = perform_KDE(
        dout['track_path_length'],
        dout['energy_deposited'],
        path_intervals=path_intervals,
        kernel='gau',
        bw=bw,
    )
    # Compute an esimtate of xi
    z=1
    Z=14
    A=28.0855*u.g/u.mol
    v = 0.7*physical_constants.c
    c = physical_constants.c
    K = 0.307075 * u.megaelectronvolt * u.cm**2 / u.mol
    m_e = physical_constants.m_e
    beta = 0.75
    estimated_path = 290 * u.micrometer
    si_density = 2.329 * u.g/u.cm**3
    xi = K/2 * Z/A * z**2 *  1/(beta)**2 * si_density *estimated_path
    xi = xi.to('keV')
    
    # Compute the FWHM of the KDE
    fwhm = estimate_fwhm(
        binned_energies['interval1'].support,
        binned_energies['interval1'].density
    )
    
    # Equally spaced bins separated by 0.5
    xbins = np.linspace(1, 300, 599)
    y_obs = binned_energies['interval1'].evaluate(xbins)
    # Run the fit on only one of the intervals
    # Scale the KDE before the fitting procedure.
    print('Scaling the original KDE by an arbitrary constant')
    print(f'Previous max {y_obs[y_obs.argmax()]:.3f}')
    y_obs_scaled = kde_scaling * y_obs
#     hist = kde_scaling*binned_energies['interval1'].evaluate(xbins)
    print(f'New max: {y_obs_scaled[y_obs_scaled.argmax()]:.3f}')
#     hist = binned_energies['interval1'].density[:-1]
#     xbins = binned_energies['interval1'].support
    nsources = binned_energies['interval1_nsources']

    if density:
        mpv_guess = xbins[y_obs_scaled.argmax()]
    else:
        mpv_guess = xbins[y_obs_scaled.argmax()].value

    limit_mpv = (1, None)
    
    xi_guess = xi.value
    limit_xi = (1, 3*xi_guess)
    
    sigma_guess = 10
    limit_sigma = (0, 20)
#     print(xbins.shape, hist.shape)
#     if not label=='WFPC2':
    fit_results = fit_energy_distribution(
        xbins[y_obs_scaled>0],
        y_obs_scaled[y_obs_scaled>0],
        mpv_guess = mpv_guess, 
        limit_mpv = (0, None),
        xi_guess = xi_guess,
        limit_xi = limit_xi,
        sigma_guess = sigma_guess,
        limit_sigma = limit_sigma,
        uselandau=uselandau,
        scaling_factor=kde_scaling
    )
    fit_results['fwhm'] = fwhm
#     fit_results['coeff'][-1] = fit_results['coeff'][-1]/kde_scaling
#     fit_results['coeff_errors'][-1] = (
#         fit_results['coeff_errors'][-1][0]/kde_scaling,
#         fit_results['coeff_errors'][-1][1]/kde_scaling
#     )
#     fit_results['rms_err'] = fit_results['rms_err']/kde_scaling
#     else: 
#         fit_results = None
    return xbins, y_obs, fit_results, nsources


def estimate_fwhm(x, y):
    
    ymax = y[y.argmax()]
    half_ymax = ymax / 2
    
    # find all values within 1/1000 of half_ymax
    flag = [False]
    tol = 1e-4
    while sum(flag) < 2:
        flag = np.isclose(half_ymax, y, atol=tol)
        tol += 0.0005
    xpoints = x[flag]
    fwhm = xpoints[-1] - xpoints[0]
    return fwhm
    


def fit_energy_distribution(
    bins,
    hist,
    mpv_guess = 40, 
    limit_mpv = (20, 150),
    xi_guess = 5,
    limit_xi = (2,20),
    sigma_guess = 8,
    limit_sigma = (0, 20),
    uselandau=False,
    scaling_factor=None
):
    A_guess = hist.max()
    limit_A = (0, None)
    # Drop the units and convert to double precisions
    try:
        bins = bins.value.astype('double')
    except AttributeError:
        bins = bins.astype('double')
#     x = 0.5*(bins[1:] + bins[:-1])
    x = bins
    y = hist
    if uselandau:
        p0 = [mpv_guess, xi_guess, A_guess]
        coeffs, errors, errors3, iminuit_fit = fit_landau_leastsq(
            x,
            y,
            p0=p0,
            limit_mpv=limit_mpv,
            limit_xi=limit_xi,
            limit_A=limit_A
        )
        residuals = y - landau(x, *coeffs)
    else:
        p0 = [mpv_guess, xi_guess, sigma_guess, A_guess]
#         print(x.shape, y.shape)
        coeffs, errors, errors3, iminuit_fit = fit_langau_leastsq(
            x,
            y,
            p0=p0,
            limit_mpv=limit_mpv,
            limit_xi=limit_xi,
            limit_sigma=limit_sigma,
            limit_A=limit_A
        )
        yfit = langau(x, *coeffs)

    if scaling_factor:
        yfit /= scaling_factor
        y /= scaling_factor
        coeffs[-1] /= scaling_factor
        errors[-1][0] /= scaling_factor
        errors[-1][1] /= scaling_factor
        errors3[-1][0]  /= scaling_factor
        errors3[-1][1] /= scaling_factor
#     avg_resid = np.nanmean(residuals)
#     rms_err = np.sqrt(np.mean(np.square(residuals)))
    residuals = y - yfit
    flag = x <= 300
    percentiles = np.percentile(y, q=[25, 75])
    iqr = percentiles[1] - percentiles[0]
#     std = np.nanstd(y[flag])
    avg_resid = np.nanmedian(residuals[flag])
    rms_err = np.sqrt(np.mean(np.square(residuals[flag])))
    fractional = residuals[flag]/y[flag]
    avg_frac_resid = np.nanmedian(fractional)
    # Compute the normalized rms
    normalized = residuals[flag]/iqr
    n_filt = normalized[np.isfinite(normalized)]
    nrms = np.sqrt(np.mean(np.square(n_filt)))
    fit_results = {
        'coeff': coeffs,
        'coeff_errors': errors,
        'coeff_errors3': errors3,
        'residuals': residuals,
        'avg_resid': avg_resid,
        'rms_err': rms_err,
        'avg_frac_resid': avg_frac_resid,
        'frac_residuals': fractional,
        'frac_rms_err': nrms,
        'iminuit_fit': iminuit_fit
    }
    return fit_results


def fit_langau_leastsq(
    x, 
    y_obs, 
    p0, 
    limit_mpv=None,
    limit_xi=None,
    limit_sigma=None, 
    limit_A=None
):
    """Perform the least squares fitting"""
    least_squares = LeastSquares(
        x=x,
        y=y_obs,
        yerror=np.sqrt(y_obs), # assume poisson errors
        model=langau,
        loss='soft_l1'
    )
    m = iminuit.Minuit(
        least_squares, # Objective funciton to minimize
        errordef=1,
        mpv=p0[0],
        limit_mpv=limit_mpv,
        eta=p0[1],
        limit_eta=limit_xi,
        sigma=p0[2],
        limit_sigma=limit_sigma,
        A=p0[2],
        limit_A=limit_A,
        pedantic=True,
        scale_langau=True,
        fix_scale_langau=True
    )
    m.migrad()
    
    if not m.valid:
        raise RuntimeError('Fit did not converge')
    
    # Compute 1-sigma minos errors
    m.minos(sigma=1)
    
    fitted_params = m.values
    fitted_params = np.array([
        fitted_params['mpv'],
        fitted_params['eta'],
        fitted_params['sigma'],
        fitted_params['A']
    ])
    param_errors = m.merrors
    param_errors = np.array(
        [(param_errors['mpv'].lower, param_errors['mpv'].upper),
         (param_errors['eta'].lower, param_errors['eta'].upper),
         (param_errors['sigma'].lower, param_errors['sigma'].upper),
         (param_errors['A'].lower, param_errors['A'].upper)]
    )
    # Compute 3-sigma errors for plotting
    m.minos(sigma=3)
    param_errors3 = m.merrors
    param_errors3 = np.array(
        [(param_errors3['mpv'].lower, param_errors3['mpv'].upper),
         (param_errors3['eta'].lower, param_errors3['eta'].upper),
         (param_errors3['sigma'].lower, param_errors3['sigma'].upper),
         (param_errors3['A'].lower, param_errors3['A'].upper)]
    )
    for val, err in zip(fitted_params, param_errors):
        print(f"{val+err[0]:.3f} < {val:0.3f} < {val+err[1]:0.3f}")
    return fitted_params, param_errors, param_errors3, m


def read_subset(
    energy, 
    size, 
    shapes, 
    cr_pixels, 
    pixel_size, 
    thickness,
    detector_shape=(1024, 1024), 
    date_range=('2003-01-01', '2005-01-01')
):
    """Read in a subset of the data for a given time period"""
    flist_tuple = list(zip(energy.hdf5_files, size.hdf5_files, shapes.hdf5_files, cr_pixels.hdf5_files))
    dout = defaultdict(list)
    min_date = Time(date_range[0], format='iso')
    max_date = Time(date_range[1], format='iso')
    end = False
    
    for (f1, f2, f3, f4) in flist_tuple:
        fobj1 = h5py.File(f1, mode='r')
        grp1 = fobj1['/energy_deposited']
        # Get the dates for the first and last dataset
        # Compare the coverage to the interval we are analyzing
        grp_list = list(grp1.keys())
        first_dset = grp1[grp_list[0]].attrs['date']
        last_dset = grp1[grp_list[-1]].attrs['date']
        
        if min_date > last_dset:
            print('Dataset not contained in this file, moving to next')
            continue
        
        fobj2 = h5py.File(f2, mode='r')
        grp2 = fobj2['/sizes']
        
        fobj3 = h5py.File(f3, mode='r')
        grp3 = fobj3['/shapes']
        
        fobj4 = h5py.File(f4, mode='r')
        grp4 = fobj4['/cr_affected_pixels']
        
        for i, key in tqdm(enumerate(grp1.keys()), total=len(grp1.keys())):
            energy_dset = grp1[key]
            image_date = Time(energy_dset.attrs['date'], format='iso')
            # The data are stored chronological order
            if image_date < min_date:
                continue
            elif image_date > max_date:
                # If the image date > max date for the current image
                # it will also be greater for every image after that, so we break the loop
                end = True
                break

            missing = False
            
            try:
                size_dset = grp2[key]
            except IndexError as e:
                print(e)
                missing = True
            
            try:
                shape_dset = grp3[key]
            except IndexError as e:
                print(e)
                missing = True
                
            try:
                cr_pixels = grp4[key]
            except IndexError as e:
                print(e)
                missing = True

            if missing:
                continue
            track_params = compute_track_params(
                cr_pixels, thickness=thickness, pixel_size=pixel_size, detector_shape=detector_shape
            )
            if track_params is None:
#                 print(key)
                continue
            label, num_sources, track_path_lengths, track_angles, track_aois = track_params      
            dout['energy_deposited'].append(da.from_array(energy_dset.value))
            dout['size_pix'].append(da.from_array(size_dset[:][1]))
            dout['size_sig'].append(da.from_array(size_dset[:][0]))
            dout['shape'].append(da.from_array(shape_dset.value))
#             dout['cr_pixels'] += list(cr_pixels.value)
            dout['proj_track_path_length'].append(da.from_array(track_path_lengths))
            dout['track_angles'].append(da.from_array(track_angles))
            dout['track_aoi'].append(da.from_array(track_aois))
                
        if end:
            break
    for key in dout.keys():
        dout[key] = np.asarray(da.concatenate(dout[key], axis=0))
            
    fobj1.close()
    fobj2.close()
    fobj3.close()
    fobj4.close()

#     attach physical units to the results values
    dout['energy_deposited'] = dout['energy_deposited'] * 3.71 * u.eV
    dout['energy_deposited'] = dout['energy_deposited'].to('keV')
    dout['track_path_length'] = np.sqrt(
        (dout['proj_track_path_length']*u.micrometer * pixel_size)**2 + (thickness*u.micrometer)**2
    ) 
    return dout


def read_ddf(fname):
    df = ddf.read_csv(fname, header=0)
    data = {}
    data['energy_deposited'] = df['energy_deposited'].compute().values * u.keV
    data['track_path_length'] = df['track_path_length'].compute().values * u.micrometer
    return data, df


def generate_dataset(
    instr='STIS_CCD',
    date_range=None,
    detector_shape=(1024, 1024)
):
    if date_range is None:
        print('Must give a date range')
        return
    instr_params = {
        'STIS_CCD' : {'pixel_size':21, 'thickness': 14.04},
        'ACS_WFC': {'pixel_size':15, 'thickness': 14.85},
        'ACS_HRC': {'pixel_size': 21, 'thickness': 14.26},
        'WFC3_UVIS': {'pixel_size': 15, 'thickness': 15.75},
        'WFPC2':{'pixel_size':15,'thickness':10}
    }
    # Instantiate the reader for each statistics we need
    reader_energy = dh.DataReader(instr=instr, statistic='energy_deposited')
    reader_energy.find_hdf5()

    reader_size = dh.DataReader(instr=instr, statistic='sizes')
    reader_size.find_hdf5()

    reader_shape = dh.DataReader(instr=instr, statistic='shapes')
    reader_shape.find_hdf5()

    reader_cr_pixels = dh.DataReader(instr=instr, statistic='cr_affected_pixels')
    reader_cr_pixels.find_hdf5()
    st = time.time()
    dout = read_subset(
        reader_energy,
        reader_size, 
        reader_shape, 
        reader_cr_pixels,
        pixel_size=instr_params[instr]['pixel_size'],
        thickness=instr_params[instr]['thickness'],
        date_range=date_range,
        detector_shape=detector_shape
    )
    et = time.time()
    duration = et - st
    units = 'sec'
    if duration > 60: 
        duration /= 60
        units = 'min'
    print(f'read time: {duration:.2f} {units}')
    return dout

def track_length_pdf_shielding(x, A, n):
    return  A/(x**(n+2))


def fit_path_length(data, keyname='proj_track_path_length', scaling=1e7):
#     hist, edges = np.histogram(data[keyname], range=(60,1000), bins=40, density=False)
    hist, edges = np.histogram(data[keyname], range=(60,1000), bins=40, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    # scale the data by a constant prior to fitting
    least_squares = LeastSquares(
        x=centers[1:],
        y=scaling*hist[1:],
        yerror=np.sqrt(scaling*hist[1:]), # assume poisson errors
        model=track_length_pdf,
        loss='soft_l1'
    )
    A_guess = hist[hist.argmax()]
    print(A_guess)
    m = iminuit.Minuit(
        least_squares, # Objective funciton to minimize
        errordef=1,
        A=A_guess,
        n=2,
        pedantic=True
    )
    m.migrad()

    return m
    
    
def plot_path_lengths(stis_ddf, hrc_ddf, wfc_ddf, wfpc2_ddf, uvis_ddf, keyname='proj_track_path_length'):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5.5,4))
    
    datasets = [stis_ddf, hrc_ddf, wfc_ddf, wfpc2_ddf, uvis_ddf]
    labels = ['STIS/CCD','ACS/HRC', 'ACS/WFC', 'WFPC2', 'WFC3/UVIS']
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                        '#f781bf', '#a65628', '#984ea3',
                        '#999999', '#e41a1c', '#dede00']    
    
    path_hists = []
    xbins = []
    for data in datasets:
        hist, edges = np.histogram(data[keyname], range=(60,1000), bins=40, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
        path_hists.append(hist)
        xbins.append(centers)
    
    print(centers[1] - centers[0])
    iterator = zip(xbins, path_hists, labels, CB_color_cycle)
    
    # Best-fit of the track length PDF
    params = [16070394.619957544, 2.6145341044273054]
    avg_params = [2816165.711095254, 2.2618257672640687]
    all_params = [3825790.1825247086, 2.3347399172885512]
    
    for x, y, label, c in iterator:
        ax.step(x, y, label=label, c=c, lw=1.2)
    
    ax.plot(x, track_length_pdf_shielding(x, *all_params), ls='--', c='k')
#     ax.plot(x, track_length_pdf_shielding(x, params[0], 1), ls='-', c='k')
    
    ax.set_xlim(0, 1000)
    ax.set_ylim(1e-7, 1e-1)
    ax.set_yscale('log')
    ax.legend(loc='upper right', edgecolor='k', fontsize=10)
    ax.set_xlabel('Path Length [$\mu$m]')
    ax.xaxis.set_major_locator(plt.MultipleLocator(100))
    ax.xaxis.set_minor_locator(plt.MultipleLocator(25))
    fig.savefig('/user/nmiles/path_lengths_with_fit.jpg', format='jpg', dpi=250, bbox_inches='tight')
    plt.show()
    


def energy_plot_apj(stis_dict, hrc_dict, wfc_dict, wfpc2_dict, 
                    every_nth_point=5, plot_fit=True):
    """ Plot the energy distribution for each instrument
    
    The inputs are all dictionaries with the following keys
    - bins
    - hist
    - x value
    - kde pdf
    - best-fit Landau+Gauss pdf
    - RMS error
    """
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                        '#f781bf', '#a65628', '#984ea3',
                        '#999999', '#e41a1c', '#dede00']
    labels = ['STIS/CCD','ACS/HRC', 'ACS/WFC', 'WFPC2', 'WFC3/UVIS']
    datasets =[stis_dict, hrc_dict, wfc_dict, wfpc2_dict]
    fig, axes = plt.subplots(
        nrows=2, ncols=2, gridspec_kw={'hspace':0.5, 'wspace':0.5},
        figsize=(6.75,4.75)
    )
    axes = axes.flatten()
    axis_label_size = 9
    tick_label_size = 8
    for data, label, c, ax in zip(datasets, labels, CB_color_cycle, axes):
        bin_centers = data['bins']
#         bin_centers = 0.5*(bins[:-1] + bins[1:])
        bin_width = bin_centers[1] - bin_centers[0]
        x = bin_centers.astype('double')
        y = data['hist']
        num_sources = data['nsources']
        print(label, x[y.argmax()])

        obsplt = ax.bar(
            x,
            y, 
            align='center', 
            width=bin_width, 
            fill=True, 
            alpha=0.6, 
            facecolor='tab:gray', 
            edgecolor='k', 
            label=f'N={num_sources:,}'
        )
#         for bar_artist in obsplt:
#             bar_artist.set_linewidth(1.1)
            
        leg = ax.legend(loc='upper right', edgecolor='k', fontsize=6.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(50))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(10))
        ax.set_xlim((0,250))
        
        
#         xkde = 0.5*(data['xkde'][:-1] + data['xkde'][1:])
        xkde = data['xkde']
        kde_pdf = data['kde_pdf']
        kdeplt = ax.plot(
            xkde,
            kde_pdf,
            c='k',
            lw=1.,
#             label='KDE',
            ls='-',
            zorder=2
        )
        ax.tick_params(
            which='both',
            axis='both',
            labelsize=tick_label_size
        )
        ax.set_xlabel('Energy Loss [keV]', fontsize=axis_label_size)
        ax.set_ylabel('Probability Density', fontsize=axis_label_size)
        ax.set_title(f'{label}',fontsize=10)
        if plot_fit:
            ylangau = langau(xkde, *data['fit_coeff'])
            ax.errorbar(
                xkde[::every_nth_point],
                ylangau[::every_nth_point],
                yerr=data['rms_err'],
                marker='D',
                lw=0.75,
                barsabove=True,
                ms=0.85,
                capthick=0.75,
                capsize=1,
                c='tab:red',
                label='Landau+Gaussian',zorder=3
            )
        ax.set_ylim(0, 0.025)
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.005))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.001))
    fig.savefig(
        '/user/nmiles/energy_loss_distribution_all.jpg',
        format='jpg', 
        dpi=250,
        bbox_inches='tight'
    )
    
def energy_plot_with_fit_apj(stis_dict, hrc_dict, wfc_dict, 
                    every_nth_point=5, plot_fit=True):
    """ Plot the energy distribution for each instrument
    
    The inputs are all dictionaries with the following keys
    - bins
    - hist
    - x value
    - kde pdf
    - best-fit Landau+Gauss pdf
    - RMS error
    """
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                        '#f781bf', '#a65628', '#984ea3',
                        '#999999', '#e41a1c', '#dede00']
    labels = ['STIS/CCD','ACS/HRC', 'ACS/WFC', 'WFPC2', 'WFC3/UVIS']
    datasets =[stis_dict, hrc_dict, wfc_dict ]
    fig, axes = plt.subplots(
        nrows=2, ncols=3, gridspec_kw={'hspace':0.1, 'wspace':0.1},
        figsize=(6.75,4.5), sharex=True
    )
    axes = axes.flatten()
    axis_label_size = 9
    tick_label_size = 8
    counter = 0
    for data, label, c in zip(datasets, labels, CB_color_cycle):
        bins = data['bins']
        bin_centers = bins
        bin_width = bins[1] - bins[0]
        x = bin_centers.astype('double')
        y = data['hist']
        num_sources = data['nsources']
        print(label, x[y.argmax()])
            
        #obsplt = ax[counter].bar(
        #    x,
        #    y, 
        #    align='center', 
        #    width=bin_width, 
        #    fill=True, 
        #    alpha=0.6, 
        #    facecolor='tab:gray', 
        #    edgecolor='tab:gray', 
#             label=f'{label} (N={num_sources:,})'
        #)
        axes[counter].xaxis.set_major_locator(plt.MultipleLocator(50))
        axes[counter].xaxis.set_minor_locator(plt.MultipleLocator(10))
        axes[counter].set_xlim((0,250))
        axes[counter].yaxis.set_major_locator(plt.MultipleLocator(0.005))
        axes[counter].yaxis.set_minor_locator(plt.MultipleLocator(0.001))
        axes[counter].set_ylim((0,0.02))        
      
        axes[counter+3].xaxis.set_major_locator(plt.MultipleLocator(50))
        axes[counter+3].xaxis.set_minor_locator(plt.MultipleLocator(10))
        axes[counter+3].set_xlim((0,250))        
        xkde = data['xkde']
        kde_pdf = data['kde_pdf']
        kdeplt = axes[counter].plot(
            xkde,
            kde_pdf,
            c='k',
            lw=.6,
#             label='KDE',
            ls='-',
            zorder=10
        )
        axes[counter].tick_params(
            which='both',
            axis='both',
            labelsize=tick_label_size
        )
        axes[counter+3].tick_params(
            which='both',
            axis='both',
            labelsize=tick_label_size
        )
        #axes[counter].set_xlabel('Energy Loss [keV]', fontsize=axis_label_size)
        axes[0].set_ylabel('Probability Density', fontsize=axis_label_size)
        axes[counter].set_title(f'{label}',fontsize=10)
        if plot_fit:
            lower_bound = data['fit_coeff'] + data['fit_error'][:, 0]
            upper_bound = data['fit_coeff'] + data['fit_error'][:, 1]
#             axes[counter].plot(
#                 xkde[::every_nth_point], 
#                 langau(xkde,*lower_bound)[::every_nth_point], 
#                 label='lower')
#             axes[counter].plot(
#                 xkde[::every_nth_point],
#                 langau(xkde, *upper_bound)[::every_nth_point], 
#                 label='upper')
#             axes[counter].fill_between(
#                 xkde[::every_nth_point],
#                 y1=langau(xkde,*lower_bound)[::every_nth_point],
#                 y2=langau(xkde, *upper_bound)[::every_nth_point],
#                 color='k',
#                 alpha=0.25
#             )
            axes[counter].axvline(data['fit_coeff'][0], lw=0.65, ls='--', color='k')
            ylangau = langau(xkde, *data['fit_coeff'])
            axes[counter].errorbar(
                xkde[::every_nth_point],
                ylangau[::every_nth_point],
                yerr=data['rms_err'],
                marker='D',
                lw=0.,
                elinewidth=0.8,
                barsabove=True,
                ms=0.7,
                capthick=0.75,
                capsize=1,
                c='tab:red',
                label='Landau+Gaussian',zorder=1
            )
            residuals = (ylangau - kde_pdf)
            frac_residuals = residuals/kde_pdf
            axes[counter+3].errorbar(
                xkde[::every_nth_point],
                frac_residuals[::every_nth_point],
                yerr=data['frac_rms_err'],
                marker='D',
                lw=0,
                elinewidth=0.75,
                barsabove=True,
                ms=0.7,
                capthick=0.75,
                capsize=1,
                c='tab:red',
            )
            axes[counter+3].fill_between(xkde, -0.1, 0.1, alpha=0.2,color='k')            
            axes[counter+3].yaxis.set_major_locator(plt.MultipleLocator(0.25))
            axes[counter+3].yaxis.set_minor_locator(plt.MultipleLocator(0.05))
            axes[counter+3].set_xlabel('Energy Loss [keV]', fontsize=axis_label_size)
        axes[3].set_ylabel('Fractional Residual', fontsize=axis_label_size)
        counter += 1
        #leg = ax.legend(loc='upper right', edgecolor='k', fontsize=6.5)
    axes[3].set_ylim(-1, 0.65)
    axes[4].set_ylim(-1, 0.65)
    axes[5].set_ylim(-1, 0.65)
    
    axes[1].tick_params(which='both', axis='y', labelleft=False)
    axes[2].tick_params(which='both', axis='y', labelleft=False)
    axes[4].tick_params(which='both', axis='y', labelleft=False)
    axes[5].tick_params(which='both', axis='y', labelleft=False)
    
    #axes[3].get_shared_y_axes().join(axes[3], axes[4])
    #axes[3].get_shared_y_axes().join(axes[3], axes[5])
    fig.savefig(
        '/user/nmiles/langau_fit_all.jpg',
        format='jpg', 
        dpi=250,
        bbox_inches='tight'
    )
     
def plot_best_fit(stis_dict, hrc_dict, wfc_dict, nsigma=3):
    
    CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                        '#f781bf', '#a65628', '#984ea3',
                        '#999999', '#e41a1c', '#dede00']
    labels = ['STIS/CCD','ACS/HRC', 'ACS/WFC', 'WFPC2', 'WFC3/UVIS']
    datasets =[stis_dict, hrc_dict, wfc_dict ]
    fig, axes = plt.subplots(
        nrows=1, ncols=1, gridspec_kw={'hspace':0.1, 'wspace':0.1},
        figsize=(5.25,4.), sharex=True
    )
    fig1, axes1 = plt.subplots(
        nrows=1, ncols=1, gridspec_kw={'hspace':0.1, 'wspace':0.1},
        figsize=(5.25,4.), sharex=True
    )
#     axes = axes.flatten()
    axis_label_size = 10 
    tick_label_size = 10
    counter = 0

    axes.xaxis.set_major_locator(plt.MultipleLocator(50))
    axes.xaxis.set_minor_locator(plt.MultipleLocator(10))
    axes.set_xlim((0,250))
    axes.yaxis.set_major_locator(plt.MultipleLocator(0.005))
    axes.yaxis.set_minor_locator(plt.MultipleLocator(0.001))
    axes.set_ylim((0,0.02)) 
    for data, label, c in zip(datasets, labels, CB_color_cycle):
        xkde = data['xkde']
        ylangau = langau(xkde, *data['fit_coeff'])
        # recompute 3-sigma errors

        lower_bound = data['fit_coeff'] + data['fit_error3'][:, 0]
        lower = langau(xkde, *lower_bound)
        
        upper_bound = data['fit_coeff'] + data['fit_error3'][:, 1]
        upper = langau(xkde, *upper_bound)
#         kde_pdf = data['kde_pdf']
        axes.plot(
            xkde,
            ylangau,
            c=c,
            lw=1.15,
            label=label,
            ls='-',
            zorder=10
        )
        axes.fill_between(
            xkde,
            y1=lower,
            y2=upper,
            color=c,
            alpha=0.5
        )
        axes.tick_params(
            which='both',
            axis='both',
            labelsize=tick_label_size
        )
        axes.tick_params(
            which='both',
            axis='both',
            labelsize=tick_label_size
        )
    axes.set_xlabel('Energy Loss [keV]', fontsize=11)
    axes.set_ylabel('Probability Density', fontsize=11)
    axes.legend(loc='upper right', edgecolor='k', fontsize=9)
    fig.savefig('/user/nmiles/langau_fit_comparisons.jpg', format='jpeg', dpi=250, bbox_inches='tight')
    
        
def compile_fit_results(stis_dict, hrc_dict, wfc_dict):
    dout = {
        'instr':[],
        'mpv':[],
        'mpv_err_minos_l':[],
        'mpv_err_minos_u':[],
        'fwhm':[],
        'xi': [],
        'xi_err_minos_l': [],
        'xi_err_minos_u': [],
        'sigma': [],
        'sigma_err_minos_l': [],
        'sigma_err_minos_u': [],
        'A': [],
        'A_err_minos_l': [],
        'A_err_minos_u': [],
        'frac_rms_err':[],
        'rms_err':[],
        'beta_l':[],
        'beta_u':[],
    }
    compute_beta = lambda xi, t: np.sqrt((0.017825 * t)/xi)
    for data, instr in zip([stis_dict, hrc_dict, wfc_dict],['STIS/CCD','ACS/HRC','ACS/WFC']):
        dout['instr'].append(instr)
        dout['mpv'].append(data['fit_coeff'][0])
        dout['mpv_err_minos_l'].append(data['fit_coeff'][0] + data['fit_error'][0][0])
        dout['mpv_err_minos_u'].append(data['fit_coeff'][0] + data['fit_error'][0][1])
        dout['fwhm'].append(data['fwhm'])
        dout['xi'].append(data['fit_coeff'][1])
        dout['xi_err_minos_l'].append(data['fit_coeff'][1] + data['fit_error'][1][0])
        dout['xi_err_minos_u'].append(data['fit_coeff'][1] + data['fit_error'][1][1])
        dout['sigma'].append(data['fit_coeff'][2])
        dout['sigma_err_minos_l'].append(data['fit_coeff'][2] + data['fit_error'][2][0])
        dout['sigma_err_minos_u'].append(data['fit_coeff'][2] + data['fit_error'][2][1])
        dout['A'].append(data['fit_coeff'][3])
        dout['A_err_minos_l'].append(data['fit_coeff'][3] + data['fit_error'][3][0])
        dout['A_err_minos_u'].append(data['fit_coeff'][3] + data['fit_error'][3][1])
        dout['frac_rms_err'].append(data['frac_rms_err'])
        dout['rms_err'].append(data['rms_err'])
        beta_l = compute_beta(data['fit_coeff'][1], 280)
        beta_u = compute_beta(data['fit_coeff'][1], 300)
        dout['beta_l'].append(beta_l)
        dout['beta_u'].append(beta_u)

    
    df = pd.DataFrame(dout)
    return df
