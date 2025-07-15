import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import scipy.interpolate as spi
import scipy.stats as stats
from scipy.interpolate import interp1d
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from matplotlib.path import Path
from mpl_toolkits import axes_grid1
import imageio
import warnings
warnings.filterwarnings("ignore", message='Ignoring specified arguments in '
                                          'this call because figure with num')

from .source_pos import gauss
from ..lib import util, plots


@plots.apply_style
def lc_nodriftcorr(meta, wave_1d, optspec, optmask=None, scandir=None,
                   mad=None, order=None):
    '''Plot a 2D light curve without drift correction. (Fig 3101+3102)

    Fig 3101 uses a linear wavelength x-axis, while Fig 3102 uses a linear
    detector pixel x-axis.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    wave_1d : Xarray Dataset
        Wavelength array with trimmed edges depending on xwindow and ywindow
        which have been set in the S3 ecf
    optspec : Xarray Dataset
        The optimally extracted spectrum.
    optmask : Xarray DataArray; optional
        A boolean mask array to use if optspec is not a masked array. Defaults
        to None in which case only the invalid values of optspec will be
        masked. Will mask the values where the mask value is set to True.
    scandir : ndarray; optional
        For HST spatial scanning mode, 0=forward scan and 1=reverse scan.
        Defaults to None which is fine for JWST data, but must be provided
        for HST data (can be all zero values if not spatial scanning mode).
    mad : float; optional
        Median absolution deviation. Default is None.
    order : int; optional
        Spectral order. Default is None
    '''
    normspec = util.normalize_spectrum(meta, optspec.values,
                                       optmask=optmask.values,
                                       scandir=scandir)

    # Save the wavelength units as the copy below will erase them
    wave_units = wave_1d.wave_units

    # For plotting purposes, extrapolate NaN wavelengths
    wave_1d = np.ma.masked_invalid(np.copy(wave_1d))
    if np.any(wave_1d.mask):
        masked = np.where(wave_1d.mask)[0]
        inds = np.arange(len(wave_1d))
        wave_1d_valid = np.delete(wave_1d, masked)
        inds_valid = np.delete(inds, masked)
        # Do a spline extrapolation of third order
        interp_fn = interp1d(inds_valid, wave_1d_valid, kind='cubic',
                             fill_value="extrapolate", assume_sorted=True)
        wave_1d[masked] = interp_fn(masked)
    wave_1d = wave_1d.data

    wmin = np.nanmin(wave_1d)
    wmax = np.nanmax(wave_1d)
    # Don't do min and max because MIRI is backwards
    # Correctly place label at center of pixel
    if meta.inst == 'miri':
        pmin = optspec.x[0].values+0.5
        pmax = optspec.x[-1].values-0.5
    else:
        pmin = optspec.x[0].values-0.5
        pmax = optspec.x[-1].values+0.5

    cmap = plt.cm.RdYlBu_r.copy()
    fig1 = plt.figure(3101)
    fig1.set_size_inches(8, 8, forward=True)
    fig1.clf()
    fig2 = plt.figure(3102)
    fig2.set_size_inches(8, 8, forward=True)
    fig2.clf()
    ax1 = fig1.gca()
    ax2 = fig2.gca()
    if meta.time_axis == 'y':
        im1 = ax1.pcolormesh(wave_1d, np.arange(meta.n_int)+0.5,
                             normspec, vmin=meta.vmin, vmax=meta.vmax,
                             cmap=cmap)
        im2 = ax2.imshow(normspec, origin='lower', aspect='auto',
                         extent=[pmin, pmax, 0, meta.n_int], vmin=meta.vmin,
                         vmax=meta.vmax, cmap=cmap)
        ax1.set_xlim(wmin, wmax)
        ax2.set_xlim(pmin, pmax)
        ax1.set_ylim(0, meta.n_int)
        ax2.set_ylim(0, meta.n_int)
        ax1.set_ylabel('Integration Number')
        ax2.set_ylabel('Integration Number')
        ax1.set_xlabel(f'Wavelength ({wave_units})')
        ax2.set_xlabel('Detector Pixel Position')
    else:
        im1 = ax1.pcolormesh(np.arange(meta.n_int), wave_1d,
                             normspec.swapaxes(0, 1), vmin=meta.vmin,
                             vmax=meta.vmax, cmap=cmap)
        im2 = ax2.imshow(normspec.swapaxes(0, 1), origin='lower',
                         aspect='auto', extent=[0, meta.n_int, pmin, pmax],
                         vmin=meta.vmin, vmax=meta.vmax, cmap=cmap)
        ax1.set_ylim(wmin, wmax)
        ax2.set_ylim(pmin, pmax)
        ax1.set_xlim(0, meta.n_int)
        ax2.set_xlim(0, meta.n_int)
        ax1.set_ylabel(f'Wavelength ({wave_units})')
        ax2.set_ylabel('Detector Pixel Position')
        ax1.set_xlabel('Integration Number')
        ax2.set_xlabel('Integration Number')

    ax1.minorticks_on()
    ax2.minorticks_on()
    if mad is not None:
        ax1.set_title(f"MAD = {mad:.0f} ppm")
        ax2.set_title(f"MAD = {mad:.0f} ppm")
    fig1.colorbar(im1, ax=ax1, label='Normalized Flux')
    fig2.colorbar(im2, ax=ax2, label='Normalized Flux')

    if order is None:
        orderkey = ''
    else:
        orderkey = f'_order{order}'
    fname1 = f'figs{os.sep}fig3101{orderkey}_2D_LC'+plots.get_filetype()
    fname2 = f'figs{os.sep}fig3102{orderkey}_2D_LC'+plots.get_filetype()
    fig1.savefig(meta.outputdir+fname1, dpi=300)
    fig2.savefig(meta.outputdir+fname2, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def image_and_background(data, meta, log, m, order=None, group=None):
    '''Make image+background plot. (Figs 3301)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    m : int
        The file number.
    order : int; optional
        Spectral order. Default is None.
    group : int; optional
        The group number (only applies to Stage 1).  Default is None.
    '''
    log.writelog('  Creating figures for background subtraction...',
                 mute=(not meta.verbose))

    # If need be, transpose array so that largest dimension is on x axis
    if len(data.x) < len(data.y):
        data = data.transpose('time', 'x', 'y')
        xmin, xmax, ymin, ymax = get_bounds(data.flux.y.values,
                                            data.flux.x.values)
    else:
        xmin, xmax, ymin, ymax = get_bounds(data.flux.x.values,
                                            data.flux.y.values)

    intstart = data.attrs['intstart']
    subdata = np.ma.masked_invalid(data.flux.values)
    subbg = np.ma.masked_invalid(data.bg.values)
    subdata = np.ma.masked_where(data.mask.values, subdata)
    subbg = np.ma.masked_where(data.mask.values, subbg)

    # Determine bounds for subdata
    stddev = np.ma.std(subdata)
    vmin = -3*stddev
    vmax = 5*stddev
    # Determine bounds for BG frame
    median = np.ma.median(subbg)
    std = np.ma.std(subbg)
    # Set bad pixels to plot as black
    cmap = plt.cm.plasma.copy()
    cmap.set_bad('k', 1.)
    iterfn = range(meta.int_start, meta.int_end)
    if meta.verbose:
        iterfn = tqdm(iterfn)
    for n in iterfn:
        fig = plt.figure(3301)
        fig.set_size_inches(8, 8, forward=True)
        fig.clf()
        plt.suptitle(f'Integration {intstart + n}')
        plt.subplot(211)
        plt.title('Background-Subtracted Frame')
        plt.imshow(subdata[n], origin='lower', aspect='auto', cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation='nearest',
                   extent=[xmin, xmax, ymin, ymax])
        plt.colorbar()
        plt.ylabel('Detector Pixel Position')
        plt.subplot(212)
        plt.title('Subtracted Background')
        plt.imshow(subbg[n], origin='lower', aspect='auto', cmap=cmap,
                   vmin=median-3*std, vmax=median+3*std,
                   interpolation='nearest',
                   extent=[xmin, xmax, ymin, ymax])
        plt.colorbar()
        plt.ylabel('Detector Pixel Position')
        plt.xlabel('Detector Pixel Position')

        file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))
                                       + 1))
        int_number = str(intstart + n).zfill(int(np.floor(
            np.log10(meta.n_int))+1))
        if order is None:
            orderkey = ''
        else:
            orderkey = f'_order{order}'
        if group is None:
            groupkey = ''
        else:
            groupkey = f'_group{group}'
        fname = (f'figs{os.sep}fig3301_file{file_number}_int{int_number}' +
                 f'{orderkey}' + f'{groupkey}' + '_' + meta.bg_dir +
                 '_ImageAndBackground' + plots.get_filetype())
        plt.savefig(meta.outputdir+fname, dpi=300)
        if not meta.hide_plots:
            plt.pause(0.2)


@plots.apply_style
def drift_2D(data, meta):
    '''Plot the fitted 2D drift. (Fig 3106)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    '''
    fig = plt.figure(3106)
    fig.set_size_inches(8, 6, forward=True)
    fig.clf()
    plt.subplot(211)
    for p in range(2):
        iscans = np.where(data.scandir.values == p)[0]
        if len(iscans) > 0:
            if p == 0:
                label = "Direction 0 (Forward)"
            else:
                label = "Direction 1 (Reverse)"
            plt.plot(iscans, data.centroid_y[iscans], '.', label=label)
    plt.ylabel(f'Drift Along y ({data.centroid_y.units})')

    plt.subplot(212)
    for p in range(2):
        iscans = np.where(data.scandir.values == p)[0]
        if len(iscans) > 0:
            if p == 0:
                label = "Direction 0 (Forward)"
            else:
                label = "Direction 1 (Reverse)"
            plt.plot(iscans, data.centroid_x[iscans], '.', label=label)
    plt.ylabel(f'Drift Along x ({data.centroid_x.units})')
    plt.xlabel('Integration Number')

    fname = f'figs{os.sep}fig3106_Drift2D{plots.get_filetype()}'
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def optimal_spectrum(data, meta, n, m):
    '''Make optimal spectrum plot. (Figs 3302)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    n : int
        The integration number.
    m : int
        The file number.
    '''
    xmin, xmax = get_bounds(data.x.values)
    intstart, stdspec, optspec, opterr = (data.attrs['intstart'],
                                          data.stdspec.values,
                                          data.optspec.values,
                                          data.opterr.values)

    fig = plt.figure(3302)
    fig.clf()
    plt.suptitle(f'1D Spectrum - Integration {intstart + n}')
    if meta.orders is None:
        plt.semilogy(data.stdspec.x.values, stdspec[n], '-', color='C1',
                     label='Standard Spec')
        plt.errorbar(data.stdspec.x.values, optspec[n], yerr=opterr[n],
                     fmt='-', color='C2', ecolor='C2', label='Optimal Spec')
    else:
        norders = len(meta.orders)
        for j in range(norders):
            order = meta.orders[j]
            inotnan = np.where(~np.isnan(data.wave_1d[:, j]))[0]
            plt.semilogy(data.x.values[inotnan], stdspec[n, inotnan, j], '-',
                         label=f'Standard Spec - Order {order}')
            plt.errorbar(data.stdspec.x.values, optspec[n, :, j],
                         yerr=opterr[n, :, j], fmt='-',
                         label=f'Optimal Spec - Order {order}')
    plt.xlim(xmin, xmax)
    plt.ylim(np.nanmin(optspec[n])/2, np.nanmax(optspec[n])*2)
    plt.ylabel('Flux')
    plt.xlabel('Detector Pixel Position')
    plt.legend(loc='best')

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(intstart + n).zfill(int(np.floor(
        np.log10(meta.n_int))+1))
    fname = (f'figs{os.sep}fig3302_file{file_number}_int{int_number}' +
             '_Spectrum'+plots.get_filetype())
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def source_position(meta, x_dim, pos_max, m, n,
                    isgauss=False, x=None, y=None, popt=None,
                    isFWM=False, y_pixels=None, sum_row=None, y_pos=None):
    '''Plot source position for MIRI data. (Figs 3103)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    x_dim : int
        The number of pixels in the y-direction in the image.
    pos_max : float
        The brightest row.
    m : int
        The file number.
    n : int
        The integration number.
    isgauss : bool; optional
        Used a guassian centring method.
    x : type; optional
        Unused.
    y : type; optional
        Unused.
    popt : list; optional
        The fitted Gaussian terms.
    isFWM : bool; optional
        Used a flux-weighted mean centring method.
    y_pixels : 1darray; optional
        The indices of the y-pixels.
    sum_row : 1darray; optional
        The sum over each row.
    y_pos : float; optional
        The FWM central position of the star.

    Notes
    -----
    History:

    - 2021-07-14: Sebastian Zieba
        Initial version.
    - Oct 15, 2021: Taylor Bell
        Tidied up the code a bit to reduce repeated code.
    '''
    fig = plt.figure(3103)
    fig.clf()
    plt.plot(y_pixels, sum_row, 'o', label='Data')
    if isgauss:
        x_gaussian = np.linspace(0, x_dim, 500)
        gaussian = gauss(x_gaussian, *popt)
        plt.plot(x_gaussian, gaussian, '-', label='Gaussian Fit')
        plt.axvline(popt[1], ls=':', label='Gaussian Center', c='C2')
        plt.xlim(pos_max-meta.spec_hw, pos_max+meta.spec_hw)
    elif isFWM:
        plt.axvline(y_pos, ls='-', label='Weighted Row')
    plt.axvline(pos_max, ls='--', label='Brightest Row', c='C3')
    plt.ylabel('Row Flux')
    plt.xlabel('Row Relative Pixel Position')
    plt.legend()

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = (f'figs{os.sep}fig3103_file{file_number}_int{int_number}' +
             '_source_pos'+plots.get_filetype())
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def profile(meta, profile, submask, n, m, order=None):
    '''Plot weighting profile from optimal spectral extraction routine.
    (Figs 3303)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    profile : ndarray
        Fitted profile in the same shape as the data array.
    submask : ndarray
        Outlier mask, where outliers are marked with the value True.
    n : int
        The current integration number.
    m : int
        The file number.
    order : int; optional
        Spectral order. Default is None
    '''
    profile = np.ma.masked_invalid(profile)
    profile = np.ma.masked_where(submask, profile)
    vmin = np.ma.min(profile)
    vmax = vmin + 0.3*np.ma.max(profile)
    cmap = plt.cm.viridis.copy()
    fig = plt.figure(3303)
    fig.set_size_inches(8, 4, forward=True)
    fig.clf()
    plt.title(f"Optimal Profile - Integration {n}")
    plt.imshow(profile, aspect='auto', origin='lower',
               vmax=vmax, vmin=vmin, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    plt.ylabel('Relative Pixel Position')
    plt.xlabel('Relative Pixel Position')

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    if order is None:
        orderkey = ''
    else:
        orderkey = f'_order{order}'
    fname = (f'figs{os.sep}fig3303_file{file_number}_int{int_number}' +
             f'{orderkey}_Profile' + plots.get_filetype())
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def subdata(meta, i, n, m, subdata, submask, expected, loc, variance):
    '''Show 1D view of profile for each column. (Figs 3501)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    i : int
        The column number.
    n : int
        The current integration number.
    m : int
        The file number.
    subdata : ndarray
        Background subtracted data.
    submask : ndarray
        Outlier mask, where outliers are marked with the value True.
    expected : ndarray
        Expected profile
    loc : int
        Location of worst outlier.
    variance : ndarray
        Variance of background subtracted data.
    '''
    ny, nx = subdata.shape
    fig = plt.figure(3501)
    fig.clf()
    plt.suptitle(f'Integration {n}, Columns {i}/{nx}')
    plt.errorbar(np.arange(ny)[np.where(~submask[:, i])[0]],
                 subdata[np.where(~submask[:, i])[0], i],
                 np.sqrt(variance[np.where(~submask[:, i])[0], i]),
                 fmt='.', color='b')
    plt.plot(np.arange(ny)[np.where(~submask[:, i])[0]],
             expected[np.where(~submask[:, i])[0], i], 'g-')
    plt.plot((loc), (subdata[loc, i]), 'ro')
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    col_number = str(i).zfill(int(np.floor(np.log10(nx))+1))
    fname = (f'figs{os.sep}fig3501_file{file_number}_int{int_number}' +
             f'_col{col_number}_subdata'+plots.get_filetype())
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.1)


@plots.apply_style
def driftypos(data, meta, m):
    '''Plot the spatial jitter. (Fig 3104)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.

    Notes
    -----
    History:

    - 2022-07-11 Caroline Piaulet
        First version of this function
    '''
    fig = plt.figure(3104)
    fig.set_size_inches(8, 4, forward=True)
    fig.clf()
    plt.plot(np.arange(meta.n_int), data["centroid_y"].values, '.')
    plt.ylabel('Spectrum spatial profile center')
    plt.xlabel('Integration Number')

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    fname = (f'figs{os.sep}fig3104_file{file_number}_DriftYPos' +
             plots.get_filetype())
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def driftywidth(data, meta, m):
    '''Plot the spatial profile's fitted Gaussian width. (Fig 3105)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.

    Notes
    -----
    History:

    - 2022-07-11 Caroline Piaulet
        First version of this function
    '''
    fig = plt.figure(3105)
    fig.set_size_inches(8, 4, forward=True)
    fig.clf()
    plt.plot(np.arange(meta.n_int), data["centroid_sy"].values, '.')
    plt.ylabel('Spectrum spatial profile width')
    plt.xlabel('Integration Number')

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    fname = (f'figs{os.sep}fig3105_file{file_number}_DriftYWidth' +
             plots.get_filetype())
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def residualBackground(data, meta, m, vmin=None, vmax=None,
                       flux=None, order=None, ap_y=None, bg_y=None):
    '''Plot the median, BG-subtracted frame to study the residual BG region and
    aperture/BG sizes. (Fig 3304)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.
    vmin : int; optional
        Minimum value of colormap. Default is None.
    vmax : int; optional
        Maximum value of colormap. Default is None.
    flux : 2D array; optional
        Median flux array. Default is None
    order : int; optional
        Spectral order. Default is None
    ap_y : list; optional
        Two-element list indicating the outer edges of the aperture region
    bg_y : list; optional
        Two-element list indicating the inner edges of the background region
    '''
    xmin, xmax, ymin, ymax = get_bounds(data.x.values, data.y.values)

    if flux is None:
        # Median flux of segment
        flux = data.medflux.values
    if ap_y is None:
        ap_y = [meta.src_ypos - meta.spec_hw,
                meta.src_ypos + meta.spec_hw + 1]
    if bg_y is None:
        bg_y = [meta.bg_y1, meta.bg_y2]
    # Compute vertical slice of width 10 columns
    flux_slice = np.nanmedian(flux[:, meta.subnx//2-5:meta.subnx//2+5], axis=1)
    # Replace NaNs with zeros to enable interpolation
    flux_slice = np.nan_to_num(flux_slice, copy=False, nan=0.0)
    # Interpolate to 0.01-pixel resolution
    f = spi.interp1d(np.arange(ymin+0.5, ymax), flux_slice, 'cubic',
                     fill_value="extrapolate", axis=0)
    ny_hr = np.arange(ymin, ymax, 0.01)
    flux_hr = f(ny_hr)
    # Set vmin and vmax
    if vmin is None:
        vmin = np.min((0, np.nanmin(flux_hr)))
    if vmax is None:
        vmax = np.nanmax(flux_hr)/3
    # Set bad pixels to plot as black
    cmap = plt.cm.plasma.copy()
    cmap.set_bad('k', 1.)

    fig = plt.figure(3304)
    fig.set_size_inches(8, 4, forward=True)
    fig.clf()
    fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]},
                                 num=3304)
    a0.imshow(flux, origin='lower', aspect='auto', vmax=vmax, vmin=vmin,
              cmap=cmap, interpolation='nearest',
              extent=[xmin, xmax, ymin, ymax])
    a0.hlines([ymin+bg_y[0], ymin+bg_y[1]], xmin, xmax, color='orange')
    a0.hlines([ymin+ap_y[0], ymin+ap_y[1]], xmin,
              xmax, color='mediumseagreen', linestyle='dashed')
    a0.axes.set_ylabel("Detector Pixel Position")
    a0.axes.set_xlabel("Detector Pixel Position")
    a1.scatter(flux_hr, ny_hr, 5, flux_hr, cmap=cmap,
               norm=plt.Normalize(vmin, vmax))
    a1.vlines([0], ymin, ymax, color='0.5', linestyle='dotted')
    a1.hlines([ymin+bg_y[0], ymin+bg_y[1]], vmin, vmax, color='orange',
              linestyle='solid', label='bg'+str(meta.bg_hw))
    a1.hlines([ymin+ap_y[0], ymin+ap_y[1]], vmin,
              vmax, color='mediumseagreen', linestyle='dashed',
              label='ap'+str(meta.spec_hw))
    a1.legend(loc='upper right', fontsize=8)
    a1.axes.set_xlabel("Flux")
    a1.axes.set_xlim(vmin, vmax)
    a1.axes.set_ylim(ymin, ymax)
    a1.axes.set_yticklabels([])
    a1.axes.set_xticks(np.linspace(vmin, vmax, 3))
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax),
                 cmap=cmap), ax=a1)

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    if order is None:
        orderkey = ''
    else:
        orderkey = f'_order{order}'
    fname = (f'figs{os.sep}fig3304_file{file_number}{orderkey}' +
             '_ResidualBG'+plots.get_filetype())
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.1)


@plots.apply_style
def curvature(meta, column_coms, smooth_coms, int_coms, m):
    '''Plot the measured, smoothed, and integer correction from the measured
    curvature. (Fig 3107)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    column_coms : 1D array
        Measured center of mass (light) for each pixel column
    smooth_coms : 1D array
        Smoothed center of mass (light) for each pixel column
    int_coms : 1D array
        Integer-rounded center of mass (light) for each pixel column
    m : int
        The file number.

    Notes
    -----
    History:

    - 2022-07-31 KBS
        Initial version
    '''
    cmap = plt.cm.viridis.copy()

    fig = plt.figure(3107)
    fig.clf()
    plt.title("Trace Curvature")
    plt.plot(column_coms+meta.ywindow[0], '.', label='Measured',
             color=cmap(0.25))
    plt.plot(smooth_coms+meta.ywindow[0], '-', label='Smoothed',
             color=cmap(0.98))
    plt.plot(int_coms+meta.ywindow[0], 's', label='Integer',
             color=cmap(0.7), ms=2)
    plt.legend()
    plt.ylabel('Detector Pixel Position')
    plt.xlabel('Detector Pixel Position')

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    fname = (f'figs{os.sep}fig3107_file{file_number}_Curvature' +
             plots.get_filetype())
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.1)


@plots.apply_style
def median_frame(data, meta, m, medflux, order=None):
    '''Plot the cleaned time-median frame. (Fig 3308)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.
    medflux : masked array
        The cleaned median flux array.
    order : int; optional
        Spectral order. Default is None
    '''
    xmin, xmax, ymin, ymax = get_bounds(data.flux.x.values, data.flux.y.values)
    vmin = medflux.min()
    vmax = medflux.max()/3
    cmap = plt.cm.plasma.copy()

    fig = plt.figure(3308)
    fig.set_size_inches(8, 4, forward=True)
    fig.clf()
    plt.title("Cleaned Median Frame")
    plt.imshow(medflux, origin='lower', aspect='auto',
               vmin=vmin, vmax=vmax, interpolation='nearest',
               extent=[xmin, xmax, ymin, ymax], cmap=cmap)
    plt.colorbar()
    plt.ylabel('Detector Pixel Position')
    plt.xlabel('Detector Pixel Position')

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    if order is None:
        orderkey = ''
    else:
        orderkey = f'_order{order}'
    fname = (f'figs{os.sep}fig3308_file{file_number}{orderkey}' +
             '_MedianFrame' + plots.get_filetype())
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.1)


# Photometry
@plots.apply_style
def phot_lc(data, meta):
    """
    Plots the flux as determined by the photometry routine as a function of
    time. (Fig 3108)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Notes
    -----
    History:

    - 2022-08-02 Sebastian Zieba
        Initial version
    """
    fig = plt.figure(3108)
    fig.clf()
    plt.suptitle('Photometric light curve')
    plt.errorbar(data.time, data['aplev'], yerr=data['aperr'], c='k', fmt='.')
    plt.ylabel('Flux')
    plt.xlabel('Time')

    fname = (f'figs{os.sep}fig3108-1D_LC' + plots.get_filetype())
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def phot_bg(data, meta):
    """
    Plots the background flux as determined by the photometry routine as a
    function of time. (Fig 3305)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Notes
    -----
    History:

    - 2022-08-02 Sebastian Zieba
        Initial version
    """
    if not meta.skip_apphot_bg:
        fig = plt.figure(3305)
        fig.clf()
        plt.suptitle('Photometric background light curve')
        plt.errorbar(data.time, data['skylev'], yerr=data['skyerr'],
                     c='k', fmt='.')
        plt.ylabel('Flux')
        plt.xlabel('Time')

        fname = (f'figs{os.sep}fig3305-1D_LC_BG' + plots.get_filetype())
        plt.savefig(meta.outputdir+fname, dpi=300)
        if not meta.hide_plots:
            plt.pause(0.2)


@plots.apply_style
def phot_centroid(data, meta):
    """
    Plots the (x, y) centroids and (sx, sy) the Gaussian 1-sigma half-widths
    as a function of time. (Fig 3109)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Notes
    -----
    History:

    - 2022-08-02 Sebastian Zieba
        Initial version
    - 2023-02-24 Isaac Edelman
        Enchanced graph layout,
        added sig display values for sy,sx,
        and fixed issue with ax[2] displaying sy instead of sx.
    - 2023-04-21 Isaac Edelman
        Added flat "0" lines to plots.
    """
    fig = plt.figure(3109)
    fig.set_size_inches(10, 6, forward=True)
    fig.clf()
    fig, ax = fig.subplots(4, 1, sharex=True)
    plt.suptitle('Centroid positions over time')

    cx = data.centroid_x.values
    cx_rms = np.sqrt(np.nanmean((cx - np.nanmedian(cx)) ** 2))
    cy = data.centroid_y.values
    cy_rms = np.sqrt(np.nanmean((cy - np.nanmedian(cy)) ** 2))
    csx = data.centroid_sx.values
    csx_rms = np.sqrt(np.nanmean((csx - np.nanmedian(csx)) ** 2))
    csy = data.centroid_sy.values
    csy_rms = np.sqrt(np.nanmean((csy - np.nanmedian(csy)) ** 2))

    ax[0].plot(data.time, data.centroid_x-np.nanmean(data.centroid_x),
               label=r'$\sigma$x = {0:.4f} pxls'.format(cx_rms))
    ax[0].axhline(y=0, linestyle=':', c='r')
    ax[0].set_ylabel('Delta x')
    ax[0].legend(bbox_to_anchor=(1.03, 0.5), loc=6)

    ax[1].plot(data.time, data.centroid_y-np.nanmean(data.centroid_y),
               label=r'$\sigma$y = {0:.4f} pxls'.format(cy_rms))
    ax[1].axhline(y=0, linestyle=':', c='r')
    ax[1].set_ylabel('Delta y')
    ax[1].legend(bbox_to_anchor=(1.03, 0.5), loc=6)

    ax[2].plot(data.time, data.centroid_sx-np.nanmean(data.centroid_sx),
               label=r'$\sigma$sx = {0:.4f} pxls'.format(csx_rms))
    ax[2].axhline(y=0, linestyle=':', c='r')
    ax[2].set_ylabel('Delta sx')
    ax[2].legend(bbox_to_anchor=(1.03, 0.5), loc=6)

    ax[3].plot(data.time, data.centroid_sy-np.nanmean(data.centroid_sy),
               label=r'$\sigma$sy = {0:.4f} pxls'.format(csy_rms))
    ax[3].axhline(y=0, linestyle=':', c='r')
    ax[3].set_ylabel('Delta sy')
    ax[3].set_xlabel('Time')
    ax[3].legend(bbox_to_anchor=(1.03, 0.5), loc=6)

    fig.get_layout_engine().set(hspace=0.02)
    fig.align_ylabels()

    fname = (f'figs{os.sep}fig3109-Centroid' + plots.get_filetype())
    plt.savefig(meta.outputdir + fname, dpi=250)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def phot_npix(data, meta):
    """
    Plots the number of pixels within the target aperture and within the
    background annulus as a function of time. (Fig 3502)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Notes
    -----
    History:

    - 2022-08-02 Sebastian Zieba
        Initial version
    """
    fig = plt.figure(3502)
    fig.clf()
    plt.suptitle('Aperture sizes over time')
    plt.subplot(211)
    plt.plot(range(len(data.nappix)), data.nappix)
    plt.ylabel('nappix')
    plt.subplot(212)
    plt.plot(range(len(data.nskypix)), data.nskypix)
    plt.ylabel('nskypix')
    plt.xlabel('Time')

    fname = (f'figs{os.sep}fig3502_aperture_size' + plots.get_filetype())
    plt.savefig(meta.outputdir + fname, dpi=250)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def phot_centroid_fgc(img, mask, x, y, sx, sy, i, m, meta):
    """
    Plot of the gaussian fit to the centroid cutout. (Fig 3309)

    Parameters
    ----------
    img : 2D numpy array
        Cutout of the center of the target which is used to determine the
        centroid position.
    mask: 2D numpy array
        A False indicates the corresponding element of Data is good, a
        True indicates it is bad, same shape as data.
    x : float
        Centroid position in x direction.
    y : float
        Centroid position in y direction.
    sx : float
        Gaussian Sigma of Centroid position in x direction.
    sy : float
        Gaussian Sigma of Centroid position in y direction.
    i : int
        The integration number.
    m : int
        The file number.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    """
    img = np.ma.copy(img)
    img = np.ma.masked_where(mask, img)

    fig = plt.figure(3309)
    fig.set_size_inches(8, 8, forward=True)
    fig.clf()
    ax = fig.subplots(2, 2)

    # Title
    plt.suptitle('Centroid gaussian fit')

    # Image of source
    ax[1, 0].imshow(img, origin='lower', aspect='auto')

    # X gaussian plot
    norm_x_factor = np.ma.sum(np.ma.sum(img, axis=0))
    ax[0, 0].plot(range(len(np.ma.sum(img, axis=0))),
                  np.ma.sum(img, axis=0)/norm_x_factor)
    x_plot = np.linspace(0, len(np.ma.sum(img, axis=0)))
    norm_distr_x = stats.norm.pdf(x_plot, x, sx)
    norm_distr_x_scaled = \
        norm_distr_x/np.nanmax(norm_distr_x)*np.nanmax(np.ma.sum(img, axis=0))
    ax[0, 0].plot(x_plot, norm_distr_x_scaled/norm_x_factor,
                  linestyle='dashed')
    ax[0, 0].set_xlabel('x position')
    ax[0, 0].set_ylabel('Normalized Flux')

    # Y gaussian plot
    norm_y_factor = np.ma.sum(np.ma.sum(img, axis=0))
    ax[1, 1].plot(np.ma.sum(img, axis=1)/norm_y_factor,
                  range(len(np.ma.sum(img, axis=1))))
    y_plot = np.linspace(0, len(np.ma.sum(img, axis=1)))
    norm_distr_y = stats.norm.pdf(y_plot, y, sy)
    norm_distr_y_scaled = \
        norm_distr_y/np.nanmax(norm_distr_y)*np.nanmax(np.ma.sum(img, axis=1))
    ax[1, 1].plot(norm_distr_y_scaled/norm_y_factor, y_plot,
                  linestyle='dashed')
    ax[1, 1].set_ylabel('y position')
    ax[1, 1].set_xlabel('Normalized Flux')

    # Last plot in (0,1) not used
    ax[0, 1].set_axis_off()

    # Naming figure
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(i).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = (f'figs{os.sep}fig3309_file{file_number}_int{int_number}'
             f'_Centroid_Fit' + plots.get_filetype())
    plt.savefig(meta.outputdir + fname, dpi=250)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    Taken from:
    https://stackoverflow.com/
    questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


@plots.apply_style
def make_artists(meta, centroid_x, centroid_y):
    """Make the aperture shapes for the photometry plots.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    centroid_x : float
        The x-coordinate of the centroid.
    centroid_y : float
        The y-coordinate of the centroid.

    Returns
    -------
    ap1 : matplotlib.patches.PathPatch
        The target aperture.
    ap2 : matplotlib.patches.PathPatch
        The inner circle of the sky annulus.
    ap3 : matplotlib.patches.PathPatch
        The outer circle of the sky annulus.
    """
    # Plot proper aperture shapes
    if meta.aperture_shape == "hexagon":
        # to make a hexagon, make the vertices and add them into a path
        # need to add extraneous vertex to close path, for some reason
        xvert = centroid_x - meta.photap*np.sin(2*np.pi*np.arange(7)/6)
        yvert = centroid_y + meta.photap*np.cos(2*np.pi*np.arange(7)/6)
        hex1 = Path(np.vstack((xvert, yvert)).T)

        # make patch of hexagon
        ap1 = patches.PathPatch(hex1, color='r',
                                fill=False, lw=3, alpha=0.7,
                                label='target aperture')

        xvert = centroid_x - meta.skyin*np.sin(2*np.pi*np.arange(7)/6)
        yvert = centroid_y + meta.skyin*np.cos(2*np.pi*np.arange(7)/6)
        hex2 = Path(np.vstack((xvert, yvert)).T)

        ap2 = patches.PathPatch(hex2, color='w',
                                fill=False, lw=4, alpha=0.8,
                                label='sky aperture')

        xvert = centroid_x - meta.skyout*np.sin(2*np.pi*np.arange(7)/6)
        yvert = centroid_y + meta.skyout*np.cos(2*np.pi*np.arange(7)/6)
        hex3 = Path(np.vstack((xvert, yvert)).T)

        ap3 = patches.PathPatch(hex3, color='w',
                                fill=False, lw=4, alpha=0.8)
    elif meta.aperture_shape == "ellipse":
        # elliptical apertures

        skyin_b = meta.skyin*(meta.photap_b/meta.photap)
        skyout_b = meta.skyout*(meta.photap_b/meta.photap)

        ap1 = patches.Ellipse((centroid_x, centroid_y), 2*meta.photap,
                              2*meta.photap_b, angle=meta.photap_theta,
                              color='r', fill=False, lw=3,
                              alpha=0.7, label='target aperture')
        ap2 = patches.Ellipse((centroid_x, centroid_y), 2*meta.skyin,
                              2*skyin_b, angle=meta.photap_theta,
                              color='w', fill=False, lw=4, alpha=0.8,
                              label='sky aperture')
        ap3 = patches.Ellipse((centroid_x, centroid_y), 2*meta.skyout,
                              2*skyout_b, angle=meta.photap_theta,
                              color='w', fill=False, lw=4, alpha=0.8)
    elif meta.aperture_shape == "rectangle":
        # rectangular apertures

        skyin_b = meta.skyin*(meta.photap_b/meta.photap)
        skyout_b = meta.skyout*(meta.photap_b/meta.photap)

        ap1 = patches.Rectangle((centroid_x-meta.photap,
                                 centroid_y-meta.photap_b),
                                2*meta.photap, 2*meta.photap_b,
                                angle=meta.photap_theta,
                                rotation_point='center',
                                color='r', fill=False, lw=3,
                                alpha=0.7, label='target aperture')
        ap2 = patches.Rectangle((centroid_x-meta.skyin,
                                 centroid_y-skyin_b),
                                2*meta.skyin, 2*skyin_b,
                                angle=meta.photap_theta,
                                rotation_point='center',
                                color='w', fill=False, lw=4, alpha=0.8,
                                label='sky aperture')
        ap3 = patches.Rectangle((centroid_x-meta.skyout,
                                 centroid_y-skyout_b),
                                2*meta.skyout, 2*skyout_b,
                                angle=meta.photap_theta,
                                rotation_point='center',
                                color='w', fill=False, lw=4, alpha=0.8)
    else:
        # circular apertures
        ap1 = plt.Circle((centroid_x, centroid_y), meta.photap, color='r',
                         fill=False, lw=3, alpha=0.7, label='target aperture')
        ap2 = plt.Circle((centroid_x, centroid_y), meta.skyin, color='w',
                         fill=False, lw=4, alpha=0.8, label='sky aperture')
        ap3 = plt.Circle((centroid_x, centroid_y), meta.skyout, color='w',
                         fill=False, lw=4, alpha=0.8)

    return ap1, ap2, ap3


@plots.apply_style
def phot_2d_frame(data, meta, m, i):
    """
    Plots the 2D frame together with the centroid position, the target aperture
    and the background annulus. (Fig 3306) If meta.isplots_S3 >= 5, this
    function will additionally create another figure - Fig 3504 - but it
    only includes the target area. (Fig 3306 and 3504)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    i : int
        The integration number.
    m : int
        The file number.

    Notes
    -----
    History:

    - 2022-08-02 Sebastian Zieba
        Initial version
    - 2024-04-06 Yoni Brande
        Added hexagonal aperture plotting
    """
    fig = plt.figure(3306)
    fig.set_size_inches(8, 8, forward=True)
    fig.clf()

    flux = data.flux.values[i]
    centroid_x = data.centroid_x.values[i]
    centroid_y = data.centroid_y.values[i]

    xmin = data.flux.x.min().values-meta.xwindow[0]
    xmax = data.flux.x.max().values-meta.xwindow[0]
    ymin = data.flux.y.min().values-meta.ywindow[0]
    ymax = data.flux.y.max().values-meta.ywindow[0]

    vmax = np.nanmedian(flux)+8*np.nanstd(flux)
    vmin = np.nanmedian(flux)-3*np.nanstd(flux)

    im = plt.imshow(flux, vmin=vmin, vmax=vmax, origin='lower', aspect='equal',
                    extent=[xmin, xmax, ymin, ymax])
    plt.scatter(centroid_x, centroid_y, marker='x', s=25, c='r',
                label='centroid')
    plt.title('Full 2D frame\nwith centroid and apertures')
    plt.ylabel('y pixels')
    plt.xlabel('x pixels')

    ap1, ap2, ap3 = make_artists(meta, centroid_x, centroid_y)
    plt.gca().add_patch(ap1)
    plt.gca().add_patch(ap2)
    plt.gca().add_patch(ap3)

    add_colorbar(im, label='Flux (electrons)')
    plt.xlim(0, flux.shape[1])
    plt.ylim(0, flux.shape[0])
    plt.xlabel('x pixels')
    plt.ylabel('y pixels')
    plt.legend(loc=1)

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(i).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = (f'figs{os.sep}fig3306_file{file_number}_int{int_number}_2D_Frame'
             + plots.get_filetype())
    plt.savefig(meta.outputdir + fname, dpi=250)
    if not meta.hide_plots:
        plt.pause(0.2)

    if meta.isplots_S3 >= 5:
        fig = plt.figure(3504)
        fig.set_size_inches(6, 5, forward=True)
        fig.clf()
        plt.title('Zoomed-in 2D frame\nwith centroid and apertures')

        im = plt.imshow(flux, vmin=vmin, vmax=vmax, origin='lower',
                        aspect='equal', extent=[xmin, xmax, ymin, ymax])
        plt.scatter(centroid_x, centroid_y, marker='x', s=25, c='r',
                    label='centroid')
        plt.ylabel('y pixels')
        plt.xlabel('x pixels')

        ap1, ap2, ap3 = make_artists(meta, centroid_x, centroid_y)
        plt.gca().add_patch(ap1)
        plt.gca().add_patch(ap2)
        plt.gca().add_patch(ap3)

        add_colorbar(im, label='Flux (electrons)')
        xlim_min = max(0, centroid_x - meta.skyout - 10)
        xlim_max = min(centroid_x + meta.skyout + 10, flux.shape[1])
        ylim_min = max(0, centroid_y - meta.skyout - 10)
        ylim_max = min(centroid_y + meta.skyout + 10, flux.shape[0])

        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)
        plt.xlabel('x pixels')
        plt.ylabel('y pixels')
        plt.legend(loc=1)

        fname = (f'figs{os.sep}fig3504_file{file_number}_int{int_number}'
                 f'_2D_Frame_Zoom' + plots.get_filetype())
        plt.savefig(meta.outputdir + fname, dpi=250)
        if not meta.hide_plots:
            plt.pause(0.2)


@plots.apply_style
def phot_2d_frame_oneoverf(data, meta, m, i, flux_w_oneoverf):
    """
    Plots the 2D frame with a low vmax so that the background is well visible.
    The top panel is before the 1/f correction, the lower panel shows the 2D
    frame after the 1/f correction. The typical "stripy" structure for each
    row should have been mitigated after the 1/f correction in Stage 3.
    (Fig 3307)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    i : int
        The integration number.
    m : int
        The file number.
    flux_w_oneoverf : 2D numpy array
        The 2D frame before the 1/f correction

    Notes
    -----
    History:

    - 2022-08-02 Sebastian Zieba
        Initial version
    """
    fig = plt.figure(3307)
    fig.set_size_inches(8.2, 4.2, forward=True)
    fig.clf()
    ax = fig.subplots(2, 1, gridspec_kw={'hspace': 0.0})

    cmap = plt.cm.viridis.copy()
    ax[0].imshow(flux_w_oneoverf, origin='lower',
                 norm=LogNorm(vmin=0.1, vmax=40), cmap=cmap)
    ax[0].set_ylabel('y pixels')

    flux = data.flux.values[i]
    im1 = ax[1].imshow(flux, origin='lower',
                       norm=LogNorm(vmin=0.1, vmax=40), cmap=cmap)
    ax[1].set_title('After 1/f correction')
    ax[1].set_xlabel('x pixels')
    ax[1].set_ylabel('y pixels')

    fig.get_layout_engine().set(hspace=0.3)
    cbar = fig.colorbar(im1, ax=ax)
    cbar.ax.set_ylabel('Flux (electrons)')

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(i).zfill(int(np.floor(np.log10(meta.n_int))+1))
    ax[0].set_title(f'Segment {file_number}, Integration {int_number}\n\n'
                    'Before 1/f correction')
    fname = (f'figs{os.sep}fig3307_file{file_number}_int{int_number}'
             f'_2D_Frame_OneOverF' + plots.get_filetype())
    plt.savefig(meta.outputdir + fname, dpi=250)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def phot_2d_frame_diff(data, meta):
    """
    Plots the difference between to consecutive 2D frames. This might be
    helpful in order to investigate flux changes due to mirror tilts
    which have been observed during commissioning. (Fig 3505)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Notes
    -----
    History:

    - 2022-08-02 Sebastian Zieba
        Initial version
    """
    nplots = meta.nplots
    if nplots == meta.n_int:
        # Need to reduce by 1 since we're doing differences
        nplots -= 1

    for i in range(nplots):
        fig = plt.figure(3505)
        fig.clf()
        flux1 = data.flux.values[i]
        flux2 = data.flux.values[i+1]
        im = plt.imshow(flux2-flux1, origin='lower', vmin=-600, vmax=600)
        plt.xlabel('x pixels')
        plt.ylabel('y pixels')
        add_colorbar(im, label='Delta Flux (electrons)')

        int_number = str(i).zfill(int(np.floor(np.log10(meta.n_int)) + 1))
        plt.title('2D frame differences\n'
                  f'Integration {int_number}')
        fname = (f'figs{os.sep}fig3505_int{int_number}_2D_Frame_Diff'
                 + plots.get_filetype())
        plt.savefig(meta.outputdir + fname, dpi=250)
        if not meta.hide_plots:
            plt.pause(0.2)


@plots.apply_style
def stddev_profile(meta, n, m, stdevs, p7thresh):
    """
    Plots the difference between the data and optimal profile in units
    of standard deviations.  The scale goes from 0 to p7thresh. (Fig 3506)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    n : int
        The current integration number.
    m : int
        The file number.
    stdevs : 2D array
        Difference between data and profile in standard deviations
    p7thresh : int
        X-sigma threshold for outlier rejection during optimal spectral
        extraction

    Notes
    -----
    History:

    - 2022-12-29 Kevin Stevenson
        Initial version
    """
    fig = plt.figure(3506)
    fig.set_size_inches(8, 4, forward=True)
    fig.clf()
    cmap = plt.cm.viridis.copy()
    plt.title(f'Std. Dev. from Optimal Profile - Integration {n}')
    plt.imshow(stdevs, origin='lower', aspect='auto',
               vmax=p7thresh, vmin=0, cmap=cmap,
               interpolation='nearest')
    plt.ylabel('Relative Pixel Position')
    plt.xlabel('Relative Pixel Position')
    plt.colorbar()

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = (f'figs{os.sep}fig3506_file{file_number}_int{int_number}' +
             '_Std_Dev_Profile'+plots.get_filetype())
    plt.savefig(meta.outputdir + fname, dpi=200)
    if not meta.hide_plots:
        plt.pause(0.1)


@plots.apply_style
def tilt_events(meta, data, log, m, position, saved_refrence_tilt_frame):
    """
    Plots the mirror tilt events by divinding
    an integrations' flux values by a
    median frames' flux values.
    Creates .pngs and a .gif (Fig 3507a, Fig 3507b, Fig 3507c)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    data : Xarray Dataset
        The Dataset object.
    log : logedit.Logedit
        The current log.
    m : int
        The file number.
    position : ndarray
        The y, x position of the star.
    saved_refrence_tilt_frame : ndarray
        The median of the first 10 integrations.

    Returns
    -------
    ndarray
        A median frame of the first 10 integrations.

    Notes
    -----
    History:

    - 2023-03-22 Isaac Edelman
        Initial implementation.
    """
    images = []
    cmap = plt.cm.viridis.copy()

    # Crop out noisy background pixels
    delta_x = 70
    delta_y = 80
    minx = -delta_x+int(position[1])
    maxx = delta_x+int(position[1])
    miny = -delta_y+int(position[0])
    maxy = delta_y+int(position[0])
    maxy = np.min([maxy, np.argmax(data.y.values)])
    miny = np.max([miny, np.argmin(data.y.values)])
    maxx = np.min([maxx, np.argmax(data.x.values)])
    minx = np.max([minx, np.argmin(data.x.values)])
    asb_xpos_min = data.x.values[minx]
    asb_xpos_max = data.x.values[maxx]
    asb_ypos_min = data.y.values[miny]
    asb_ypos_max = data.y.values[maxy]

    # Create median frame
    if saved_refrence_tilt_frame is None:
        refrence_tilt_frame = ((np.nanmedian(data.flux.values[:10],
                                             axis=0))[miny:maxy, minx:maxx])
    else:
        refrence_tilt_frame = saved_refrence_tilt_frame

    # Plot each integration
    for i in tqdm(range(len(data.time)), desc='  Creating tilt event figures'):

        # Caluculate flux ratio
        flux_tilt = (data.flux.values[i, miny:maxy,
                                      minx:maxx] / refrence_tilt_frame)

        # Create plot
        fig = plt.figure(3507)
        fig.set_size_inches(6, 6, forward=True)
        fig.clf()

        # Plot figure
        im = plt.imshow(flux_tilt, origin='lower', aspect='equal',
                        vmin=0.98, vmax=1.02, cmap=cmap)

        # Figure settings
        plt.xticks(np.arange(0, flux_tilt.shape[1], 1),
                   (np.arange(asb_xpos_min, asb_xpos_max, 1)),
                   rotation='vertical')
        plt.yticks(np.arange(0, flux_tilt.shape[0], 1),
                   (np.arange(asb_ypos_min, asb_ypos_max, 1)),
                   rotation='horizontal')
        add_colorbar(im, label='Flux Ratio')
        plt.locator_params(nbins=11)
        plt.tick_params(labelsize='small')
        plt.xlabel('x pixels')
        plt.ylabel('y pixels')

        # Create file names
        tilt_events = os.path.join(meta.outputdir + 'figs', 'tilt_events')
        if not os.path.exists(tilt_events):
            os.mkdir(tilt_events)
        file_number = str(m).zfill(int(np.floor(np.log10(
            meta.num_data_files))+1))
        int_number = str(i).zfill(int(np.floor(np.log10(meta.n_int))+1))
        plt.title('Tilt Identification\n'
                  f'Batch {file_number}, Integration {int_number}')
        fname = (f'figs{os.sep}tilt_events{os.sep}'
                 f'fig3507a_file{file_number}_int{int_number}'
                 f'_tilt_events' + plots.get_filetype())
        # Save figure
        plt.savefig(meta.outputdir + fname, dpi=250, bbox_inches='tight')
        if not meta.hide_plots:
            plt.pause(0.2)

        # Create list of figure names to pull from later to create .gif
        images.append(imageio.v2.imread(meta.outputdir + fname))

    # Figure fig3507b
    # Create .gif per batch
    if meta.nbatch > 1:
        log.writelog('  Creating batch tilt event GIF',
                     mute=(not meta.verbose))
        imageio.mimsave(meta.outputdir + f'figs{os.sep}' +
                        f'fig3507b_tilt_event_batch_{file_number}.gif',
                        images, fps=20)

    # Figure fig3507c
    # Create .gif of all tilt event segments combined
    if not meta.testing_S3 and (m + 1 == meta.nbatch):
        log.writelog('  Creating all segment tilt event GIF',
                     mute=(not meta.verbose))

        # Create list of all .png tilt images in tilt_event folder
        all_images = []
        in_filenames = []
        for file in os.listdir(meta.outputdir +
                               f'figs{os.sep}tilt_events{os.sep}'):
            if file.endswith(".png"):
                in_filenames.append(os.path.join(meta.outputdir +
                                                 f'figs{os.sep}' +
                                                 f'tilt_events{os.sep}', file))
        in_filenames.sort()

        # Create list of all figure names to pull from later to create .gif
        for fname in in_filenames:
            all_images.append(imageio.v2.imread(fname))

        # Create .gif of all tilt event segments
        imageio.mimsave(meta.outputdir + f'figs{os.sep}' +
                        'fig3507c_tilt_events_all_segments.gif',
                        all_images, fps=60)

    return refrence_tilt_frame


def get_bounds(x, y=None):
    """
    Define bounds by adding half pixel to all edges

    Parameters
    ----------
    x : 1D array
        Pixel indices along x axis.
    y : 1D array, optional
        Pixel indices along y axis.

    Returns
    -------
    xmin
        Minimum x bound
    xmax
        Maximum x bound
    ymin, optional
        Minimum y bound
    ymax, optional
        Maximum y bound

    """
    xmin, xmax = x[0], x[-1]
    if xmin < xmax:
        # NIR instruments
        xmin -= 0.5
        xmax += 0.5
    else:
        # MIRI
        xmin += 0.5
        xmax -= 0.05
    if y is not None:
        ymin, ymax = y[0], y[-1]
        if ymin < ymax:
            # All instruments
            ymin -= 0.5
            ymax += 0.5
        else:
            # Possible future use
            ymin += 0.5
            ymax -= 0.05
        return xmin, xmax, ymin, ymax
    else:
        return xmin, xmax
