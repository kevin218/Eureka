import numpy as np
import os
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.interpolate as spi
from .source_pos import gauss
from ..lib import util
from ..lib.plots import figure_filetype
import scipy.stats as stats
from matplotlib.colors import LogNorm
from mpl_toolkits import axes_grid1


def lc_nodriftcorr(meta, wave_1d, optspec, optmask=None):
    '''Plot a 2D light curve without drift correction. (Fig 3101)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    wave_1d : ndarray
        Wavelength array with trimmed edges depending on xwindow and ywindow
        which have been set in the S3 ecf
    optspec : ndarray
        The optimally extracted spectrum.
    optmask : ndarray (1D); optional
        A mask array to use if optspec is not a masked array. Defaults to None
        in which case only the invalid values of optspec will be masked.
    '''
    normspec = util.normalize_spectrum(meta, optspec, optmask=optmask)
    wmin = np.ma.min(wave_1d)
    wmax = np.ma.max(wave_1d)
    if not hasattr(meta, 'vmin') or meta.vmin is None:
        meta.vmin = 0.97
    if not hasattr(meta, 'vmax') or meta.vmin is None:
        meta.vmax = 1.03
    if not hasattr(meta, 'time_axis') or meta.time_axis is None:
        meta.time_axis = 'y'
    elif meta.time_axis not in ['y', 'x']:
        print("WARNING: meta.time_axis is not one of ['y', 'x']!"
              " Using 'y' by default.")
        meta.time_axis = 'y'

    plt.figure(3101, figsize=(8, 8))
    plt.clf()
    if meta.time_axis == 'y':
        plt.imshow(normspec, origin='lower', aspect='auto',
                   extent=[wmin, wmax, 0, meta.n_int], vmin=meta.vmin,
                   vmax=meta.vmax, cmap=plt.cm.RdYlBu_r)
        plt.ylabel('Integration Number')
        plt.xlabel(r'Wavelength ($\mu m$)')
    else:
        plt.imshow(normspec.swapaxes(0, 1), origin='lower', aspect='auto',
                   extent=[0, meta.n_int, wmin, wmax], vmin=meta.vmin,
                   vmax=meta.vmax, cmap=plt.cm.RdYlBu_r)
        plt.ylabel(r'Wavelength ($\mu m$)')
        plt.xlabel('Integration Number')

    plt.title(f"MAD = {np.round(meta.mad_s3, 0).astype(int)} ppm")
    plt.colorbar(label='Normalized Flux')
    plt.tight_layout()
    fname = f'figs{os.sep}fig3101-2D_LC'+figure_filetype
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def image_and_background(data, meta, log, m):
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
    '''
    log.writelog('  Creating figures for background subtraction...',
                 mute=(not meta.verbose))

    intstart = data.attrs['intstart']
    subdata = np.ma.masked_where(~data.mask.values, data.flux.values)
    subbg = np.ma.masked_where(~data.mask.values, data.bg.values)

    xmin, xmax = data.flux.x.min().values, data.flux.x.max().values
    ymin, ymax = data.flux.y.min().values, data.flux.y.max().values

    # Commented out vmax calculation is sensitive to unflagged hot pixels
    # vmax = np.ma.max(np.ma.masked_invalid(subdata))/40
    vmin = -200
    vmax = 1000
    median = np.ma.median(subbg)
    std = np.ma.std(subbg)
    # Set bad pixels to plot as black
    cmap = mpl.cm.get_cmap("plasma").copy()
    cmap.set_bad('k', 1.)
    iterfn = range(meta.int_end-meta.int_start)
    if meta.verbose:
        iterfn = tqdm(iterfn)
    for n in iterfn:
        plt.figure(3301, figsize=(8, 8))
        plt.clf()
        plt.suptitle(f'Integration {intstart + n}')
        plt.subplot(211)
        plt.title('Background-Subtracted Frame')
        plt.imshow(subdata[n], origin='lower', aspect='auto', cmap=cmap,
                   vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax])
        plt.colorbar()
        plt.ylabel('Detector Pixel Position')
        plt.subplot(212)
        plt.title('Subtracted Background')
        plt.imshow(subbg[n], origin='lower', aspect='auto', cmap=cmap,
                   vmin=median-3*std, vmax=median+3*std,
                   extent=[xmin, xmax, ymin, ymax])
        plt.colorbar()
        plt.ylabel('Detector Pixel Position')
        plt.xlabel('Detector Pixel Position')
        plt.tight_layout()
        file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))
                                       + 1))
        int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
        fname = (f'figs{os.sep}fig3301_file{file_number}_int{int_number}' +
                 '_ImageAndBackground'+figure_filetype)
        plt.savefig(meta.outputdir+fname, dpi=300)
        if not meta.hide_plots:
            plt.pause(0.2)


def drift_2D(data, meta):
    '''Plot the fitted 2D drift. (Fig 3105)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    '''
    plt.figure(3105, figsize=(8, 6))
    plt.clf()
    plt.subplot(211)
    for p in range(2):
        iscans = np.where(data.scandir.values == p)[0]
        plt.plot(iscans, data.drift2D[iscans, 1], '.')
    plt.ylabel(f'Drift Along y ({data.drift2D.drift_units})')
    plt.subplot(212)
    for p in range(2):
        iscans = np.where(data.scandir.values == p)[0]
        plt.plot(iscans, data.drift2D[iscans, 0], '.')
    plt.ylabel(f'Drift Along x ({data.drift2D.drift_units})')
    plt.xlabel('Integration Number')
    plt.tight_layout()
    fname = f'figs{os.sep}fig3105_Drift2D{figure_filetype}'
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


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
    intstart, stdspec, optspec, opterr = (data.attrs['intstart'],
                                          data.stdspec.values,
                                          data.optspec.values,
                                          data.opterr.values)

    plt.figure(3302)
    plt.clf()
    plt.suptitle(f'1D Spectrum - Integration {intstart + n}')
    plt.semilogy(data.stdspec.x.values, stdspec[n], '-', color='C1',
                 label='Standard Spec')
    plt.errorbar(data.stdspec.x.values, optspec[n], yerr=opterr[n], fmt='-',
                 color='C2', ecolor='C2', label='Optimal Spec')
    plt.ylabel('Flux')
    plt.xlabel('Detector Pixel Position')
    plt.legend(loc='best')
    plt.tight_layout()
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = (f'figs{os.sep}fig3302_file{file_number}_int{int_number}' +
             '_Spectrum'+figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def source_position(meta, x_dim, pos_max, m, n,
                    isgauss=False, x=None, y=None, popt=None,
                    isFWM=False, y_pixels=None, sum_row=None, y_pos=None):
    '''Plot source position for MIRI data. (Figs 3102)

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
    plt.figure(3102)
    plt.clf()
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
    plt.xlabel('Row Pixel Position')
    plt.legend()
    plt.tight_layout()
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = (f'figs{os.sep}fig3102_file{file_number}_int{int_number}' +
             '_source_pos'+figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def profile(meta, profile, submask, n, m):
    '''Plot weighting profile from optimal spectral extraction routine. (Figs 3303)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    profile : ndarray
        Fitted profile in the same shape as the data array.
    submask : ndarray
        Outlier mask.
    n : int
        The current integration number.
    m : int
        The file number.
    '''
    profile = np.ma.masked_invalid(profile)
    submask = np.ma.masked_invalid(submask)
    mask = np.logical_or(np.ma.getmaskarray(profile),
                         np.ma.getmaskarray(submask))
    profile = np.ma.masked_where(mask, profile)
    submask = np.ma.masked_where(mask, submask)
    vmin = np.ma.min(profile*submask)
    vmax = vmin + 0.05*np.ma.max(profile*submask)
    plt.figure(3303)
    plt.clf()
    plt.suptitle(f"Profile - Integration {n}")
    plt.imshow(profile*submask, aspect='auto', origin='lower',
               vmax=vmax, vmin=vmin)
    plt.ylabel('Relative Pixel Position')
    plt.xlabel('Relative Pixel Position')
    plt.tight_layout()
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = (f'figs{os.sep}fig3303_file{file_number}_int{int_number}_Profile' +
             figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def subdata(meta, i, n, m, subdata, submask, expected, loc):
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
        Outlier mask.
    expected : ndarray
        Expected profile
    loc : ndarray
        Location of worst outliers.
    '''
    ny, nx = subdata.shape
    plt.figure(3501)
    plt.clf()
    plt.suptitle(f'Integration {n}, Columns {i}/{nx}')
    plt.plot(np.arange(ny)[np.where(submask[:, i])[0]],
             subdata[np.where(submask[:, i])[0], i], 'bo')
    plt.plot(np.arange(ny)[np.where(submask[:, i])[0]],
             expected[np.where(submask[:, i])[0], i], 'g-')
    plt.plot((loc[i]), (subdata[loc[i], i]), 'ro')
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    col_number = str(i).zfill(int(np.floor(np.log10(nx))+1))
    fname = (f'figs{os.sep}fig3501_file{file_number}_int{int_number}' +
             f'_col{col_number}_subdata'+figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.1)


def driftypos(data, meta):
    '''Plot the spatial jitter. (Fig 3103)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Notes
    -----
    History:

    - 2022-07-11 Caroline Piaulet
        First version of this function
    '''
    plt.figure(3103, figsize=(8, 4))
    plt.clf()
    plt.plot(np.arange(meta.n_int), data["driftypos"].values, '.')
    plt.ylabel('Spectrum spatial profile center')
    plt.xlabel('Integration Number')
    plt.tight_layout()
    fname = 'figs'+os.sep+'fig3103_DriftYPos'+figure_filetype
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def driftywidth(data, meta):
    '''Plot the spatial profile's fitted Gaussian width. (Fig 3104)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Notes
    -----
    History:

    - 2022-07-11 Caroline Piaulet
        First version of this function
    '''
    plt.figure(3104, figsize=(8, 4))
    plt.clf()
    plt.plot(np.arange(meta.n_int), data["driftywidth"].values, '.')
    plt.ylabel('Spectrum spatial profile width')
    plt.xlabel('Integration Number')
    plt.tight_layout()
    fname = 'figs'+os.sep+'fig3104_DriftYWidth'+figure_filetype
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def residualBackground(data, meta, m, vmin=-200, vmax=1000):
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
        Minimum value of colormap. Default is -200.
    vmax : int; optional
        Maximum value of colormap. Default is 1000.

    Notes
    -----
    History:

    - 2022-07-29 KBS
        Initial version
    '''
    xmin, xmax = data.flux.x.min().values, data.flux.x.max().values
    ymin, ymax = data.flux.y.min().values, data.flux.y.max().values

    # Median flux of segment
    subdata = np.ma.masked_where(~data.mask.values, data.flux.values)
    flux = np.ma.median(subdata, axis=0)
    # Compute vertical slice of with 10 columns
    slice = np.nanmedian(flux[:, meta.subnx//2-5:meta.subnx//2+5], axis=1)
    # Interpolate to 0.01-pixel resolution
    f = spi.interp1d(np.arange(ymin, ymax+1), slice, 'cubic')
    ny_hr = np.arange(ymin, ymax, 0.01)
    flux_hr = f(ny_hr)
    # Set bad pixels to plot as black
    cmap = mpl.cm.get_cmap("plasma").copy()
    cmap.set_bad('k', 1.)

    plt.figure(3304, figsize=(8, 3.5))
    plt.clf()
    fig, (a0, a1) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]},
                                 num=3304, figsize=(8, 3.5))
    a0.imshow(flux, origin='lower', aspect='auto', vmax=vmax, vmin=vmin,
              cmap=cmap, extent=[xmin, xmax, ymin, ymax])
    a0.hlines([ymin+meta.bg_y1, ymin+meta.bg_y2], xmin, xmax, color='orange')
    a0.hlines([ymin+meta.src_ypos+meta.spec_hw,
              ymin+meta.src_ypos-meta.spec_hw], xmin,
              xmax, color='mediumseagreen', linestyle='dashed')
    a0.axes.set_ylabel("Detector Pixel Position")
    a0.axes.set_xlabel("Detector Pixel Position")
    a1.scatter(flux_hr, ny_hr, 5, flux_hr, cmap='plasma',
               norm=plt.Normalize(vmin, vmax))
    a1.vlines([0], ymin, ymax, color='0.5', linestyle='dotted')
    a1.hlines([ymin+meta.bg_y1, ymin+meta.bg_y2], vmin, vmax, color='orange',
              linestyle='solid', label='bg'+str(meta.bg_hw))
    a1.hlines([ymin+meta.src_ypos+meta.spec_hw,
              ymin+meta.src_ypos-meta.spec_hw], vmin,
              vmax, color='mediumseagreen', linestyle='dashed',
              label='ap'+str(meta.spec_hw))
    a1.legend(loc='upper right', fontsize=8)
    a1.axes.set_xlabel("Flux [e-]")
    a1.axes.set_xlim(vmin, vmax)
    a1.axes.set_ylim(ymin, ymax)
    a1.axes.set_yticklabels([])
    # a1.yaxis.set_visible(False)
    a1.axes.set_xticks(np.linspace(vmin, vmax, 3))
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(vmin, vmax),
                 cmap='plasma'), ax=a1)
    fig.subplots_adjust(top=0.97,
                        bottom=0.155,
                        left=0.08,
                        right=0.925,
                        hspace=0.2,
                        wspace=0.08)
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    fname = (f'figs{os.sep}fig3304_file{file_number}' +
             '_ResidualBG'+figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.1)


def curvature(meta, column_coms, smooth_coms, int_coms):
    '''Plot the measured, smoothed, and integer correction from the measured
    curvature. (Fig 3106)

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

    Notes
    -----
    History:

    - 2022-07-31 KBS
        Initial version
    '''
    colors = mpl.cm.viridis

    plt.figure(3106)
    plt.clf()
    plt.title("Trace Curvature")
    plt.plot(column_coms, '.', label='Measured', color=colors(0.25))
    plt.plot(smooth_coms, '-', label='Smoothed', color=colors(0.98))
    plt.plot(int_coms, 's', label='Integer', color=colors(0.7), ms=2)
    plt.legend()
    plt.ylabel('Relative Pixel Position')
    plt.xlabel('Relative Pixel Position')
    plt.tight_layout()

    fname = (f'figs{os.sep}fig3106_Curvature'+figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.1)


def median_frame(data, meta):
    '''Plot the cleaned time-median frame. (Fig 3401)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Notes
    -----
    History:

    - 2022-08-06 KBS
        Initial version
    '''
    xmin, xmax = data.flux.x.min().values, data.flux.x.max().values
    ymin, ymax = data.flux.y.min().values, data.flux.y.max().values
    vmin = data.medflux.min().values
    vmax = vmin + 2000

    plt.figure(3401)
    plt.clf()
    plt.title("Cleaned Median Frame")
    plt.imshow(data.medflux, origin='lower', aspect='auto',
               vmin=vmin, vmax=vmax, extent=[xmin, xmax, ymin, ymax])
    plt.ylabel('Detector Pixel Position')
    plt.xlabel('Detector Pixel Position')
    plt.tight_layout()

    fname = (f'figs{os.sep}fig3401_MedianFrame'+figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.1)


# Photometry
def phot_lc(data, meta):
    """
    Plots the flux as determined by the photometry routine as a function of time. (Fig 3107)

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
    plt.figure(3107)
    plt.clf()
    plt.suptitle('Photometric light curve')
    plt.errorbar(data.time, data['aplev'], yerr=data['aperr'], c='k', fmt='.')
    plt.ylabel('Flux')
    plt.xlabel('Time')
    plt.tight_layout()
    fname = (f'figs{os.sep}fig3107-1D_LC' + figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def phot_bg(data, meta):
    """
    Plots the background flux as determined by the photometry routine as a function of time.
    (Fig 3305)

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
        plt.figure(3305)
        plt.clf()
        plt.suptitle('Photometric background light curve')
        plt.errorbar(data.time, data['skylev'], yerr=data['skyerr'], c='k', fmt='.')
        plt.ylabel('Flux')
        plt.xlabel('Time')
        plt.tight_layout()
        fname = (f'figs{os.sep}fig3305-1D_LC_BG' + figure_filetype)
        plt.savefig(meta.outputdir+fname, dpi=300)
        if not meta.hide_plots:
            plt.pause(0.2)


def phot_centroid(data, meta):
    """
    Plots the (x, y) centroids and (sx, sy) the Gaussian 1-sigma half-widths as a function of time.
    (Fig 3108)

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
    plt.figure(3108)
    plt.clf()
    fig, ax = plt.subplots(4, 1, sharex=True)
    plt.suptitle('Centroid positions over time')

    cx = data.centroid_x.values
    cx_rms = np.sqrt(np.mean((cx - np.median(cx)) ** 2))
    cy = data.centroid_y.values
    cy_rms = np.sqrt(np.mean((cy - np.median(cy)) ** 2))

    ax[0].plot(data.time, data.centroid_x-np.mean(data.centroid_x),
               label=r'$\sigma$x = {0:.4f} pxls'.format(cx_rms))
    ax[0].set_ylabel('Delta x')
    ax[0].legend()

    ax[1].plot(data.time, data.centroid_y-np.mean(data.centroid_y),
               label=r'$\sigma$y = {0:.4f} pxls'.format(cy_rms))
    ax[1].set_ylabel('Delta y')
    ax[1].legend()

    ax[2].plot(data.time, data.centroid_sy-np.mean(data.centroid_sx))
    ax[2].set_ylabel('Delta sx')

    ax[3].plot(data.time, data.centroid_sy-np.mean(data.centroid_sy))
    ax[3].set_ylabel('Delta sy')
    ax[3].set_xlabel('Time')

    fig.subplots_adjust(hspace=0.02)

    plt.tight_layout()
    fname = (f'figs{os.sep}fig3108-Centroid' + figure_filetype)
    plt.savefig(meta.outputdir + fname, dpi=250)
    if not meta.hide_plots:
        plt.pause(0.2)


def phot_npix(data, meta):
    """
    Plots the number of pixels within the target aperture and within the background annulus
    as a function of time. (Fig 3502)

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
    plt.figure(3502)
    plt.clf()
    plt.suptitle('Aperture sizes over time')
    plt.subplot(211)
    plt.plot(range(len(data.nappix)), data.nappix)
    plt.ylabel('nappix')
    plt.subplot(212)
    plt.plot(range(len(data.nskypix)), data.nskypix)
    plt.ylabel('nskypix')
    plt.xlabel('Time')
    plt.tight_layout()
    fname = (f'figs{os.sep}fig3502_aperture_size' + figure_filetype)
    plt.savefig(meta.outputdir + fname, dpi=250)
    if not meta.hide_plots:
        plt.pause(0.2)


def phot_centroid_fgc(img, x, y, sx, sy, i, m, meta):
    """
    Plot of the gaussian fit to the centroid cutout. (Fig 3503)

    Parameters
    ----------
    img : 2D numpy array
        Cutout of the center of the target which is used to determine the centroid position.
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

    Notes
    -----
    History:

    - 2022-08-02 Sebastian Zieba
        Initial version
    """
    plt.figure(3503)
    plt.clf()

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    plt.suptitle('Centroid gaussian fit')
    fig.delaxes(ax[1, 1])
    ax[0, 0].imshow(img, vmax=5e3, origin='lower', aspect='auto')

    ax[1, 0].plot(range(len(np.sum(img, axis=0))), np.sum(img, axis=0))
    x_plot = np.linspace(0, len(np.sum(img, axis=0)))
    norm_distr_x = stats.norm.pdf(x_plot, x, sx)
    norm_distr_x_scaled = norm_distr_x/np.max(norm_distr_x)*np.max(np.sum(img, axis=0))
    ax[1, 0].plot(x_plot, norm_distr_x_scaled)
    ax[1, 0].set_xlabel('x position')
    ax[1, 0].set_ylabel('Flux (electrons)')

    ax[0, 1].plot(np.sum(img, axis=1), range(len(np.sum(img, axis=1))))
    y_plot = np.linspace(0, len(np.sum(img, axis=1)))
    norm_distr_y = stats.norm.pdf(y_plot, y, sy)
    norm_distr_y_scaled = norm_distr_y/np.max(norm_distr_y)*np.max(np.sum(img, axis=1))
    ax[0, 1].plot(norm_distr_y_scaled, y_plot)
    ax[0, 1].set_ylabel('y position')
    ax[0, 1].set_xlabel('Flux (electrons)')

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(i).zfill(int(np.floor(np.log10(meta.n_int))+1))
    plt.tight_layout()
    fname = (f'figs{os.sep}fig3503_file{file_number}_int{int_number}_Centroid_Fit'
             + figure_filetype)
    plt.savefig(meta.outputdir + fname, dpi=250)
    if not meta.hide_plots:
        plt.pause(0.2)


def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """
    Add a vertical color bar to an image plot.
    Taken from:
    https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
    """
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)


def phot_2d_frame(data, meta, m, i):
    """
    Plots the 2D frame together with the centroid position, the target aperture
    and the background annulus. (Fig 3306) If meta.isplots_S3 >= 5, this function will additionally
    create another figure - Fig 3504 - but it only includes the target area. (Fig 3306 and 3504)

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
    """
    plt.figure(3306, figsize=(8, 3))
    plt.clf()
    plt.suptitle('2D frame with centroid and apertures')

    flux, centroid_x, centroid_y = data.flux[i], data.centroid_x[i], data.centroid_y[i]

    xmin, xmax = data.flux.x.min().values-meta.xwindow[0], data.flux.x.max().values-meta.xwindow[0]
    ymin, ymax = data.flux.y.min().values-meta.ywindow[0], data.flux.y.max().values-meta.ywindow[0]

    im = plt.imshow(flux, vmin=0, vmax=5e3, origin='lower', aspect='equal',
                    extent=[xmin, xmax, ymin, ymax])
    plt.scatter(centroid_x, centroid_y, marker='x', s=25, c='r', label='centroid')
    plt.title('Full 2D frame')
    plt.ylabel('y pixels')
    plt.xlabel('x pixels')

    circle1 = plt.Circle((centroid_x, centroid_y), meta.photap, color='r',
                         fill=False, lw=3, alpha=0.7, label='target aperture')
    circle2 = plt.Circle((centroid_x, centroid_y), meta.skyin, color='w',
                         fill=False, lw=4, alpha=0.8, label='sky aperture')
    circle3 = plt.Circle((centroid_x, centroid_y), meta.skyout, color='w',
                         fill=False, lw=4, alpha=0.8)
    plt.gca().add_patch(circle1)
    plt.gca().add_patch(circle2)
    plt.gca().add_patch(circle3)
    add_colorbar(im, label='Flux (electrons)')
    plt.xlim(0, flux.shape[1])
    plt.ylim(0, flux.shape[0])
    plt.xlabel('x pixels')
    plt.ylabel('y pixels')

    plt.legend()

    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(i).zfill(int(np.floor(np.log10(meta.n_int))+1))

    fname = (f'figs{os.sep}fig3306_file{file_number}_int{int_number}_2D_Frame'
             + figure_filetype)
    plt.savefig(meta.outputdir + fname, dpi=250)
    if not meta.hide_plots:
        plt.pause(0.2)

    if meta.isplots_S3 >= 5:
        plt.figure(3504, figsize=(6, 5))
        plt.clf()
        plt.suptitle('2D frame with centroid and apertures (zoom-in version)')

        im = plt.imshow(flux, vmin=0, vmax=5e3, origin='lower', aspect='equal',
                        extent=[xmin, xmax, ymin, ymax])
        plt.scatter(centroid_x, centroid_y, marker='x', s=25, c='r', label='centroid')
        plt.title('Zoom into 2D frame')
        plt.ylabel('y pixels')
        plt.xlabel('x pixels')

        circle1 = plt.Circle((centroid_x, centroid_y), meta.photap, color='r',
                             fill=False, lw=3, alpha=0.7, label='target aperture')
        circle2 = plt.Circle((centroid_x, centroid_y), meta.skyin, color='w',
                             fill=False, lw=4, alpha=0.8, label='sky aperture')
        circle3 = plt.Circle((centroid_x, centroid_y), meta.skyout, color='w',
                             fill=False, lw=4, alpha=0.8)
        plt.gca().add_patch(circle1)
        plt.gca().add_patch(circle2)
        plt.gca().add_patch(circle3)

        add_colorbar(im, label='Flux (electrons)')
        xlim_min = max(0, centroid_x - meta.skyout - 10)
        xlim_max = min(centroid_x + meta.skyout + 10, flux.shape[1])
        ylim_min = max(0, centroid_y - meta.skyout - 10)
        ylim_max = min(centroid_y + meta.skyout + 10, flux.shape[0])

        plt.xlim(xlim_min, xlim_max)
        plt.ylim(ylim_min, ylim_max)
        plt.xlabel('x pixels')
        plt.ylabel('y pixels')

        plt.legend()

        fname = (f'figs{os.sep}fig3504_file{file_number}_int{int_number}_2D_Frame_Zoom'
                 + figure_filetype)
        plt.savefig(meta.outputdir + fname, dpi=250)
        if not meta.hide_plots:
            plt.pause(0.2)


def phot_2d_frame_oneoverf(data, meta, m, i, flux_w_oneoverf):
    """
    Plots the 2D frame with a low vmax so that the background is well visible. The top panel
    is before the 1/f correction, the lower panel shows the 2D frame after the 1/f correction.
    The typical "stripy" structure for each row should have been mitigated after the 1/f correction
    in Stage 3. (Fig 3307)

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
    plt.figure(3307)
    plt.clf()
    fig, ax = plt.subplots(2, 1, figsize=(8.2, 4.2))

    ax[0].imshow(flux_w_oneoverf, origin='lower', norm=LogNorm(vmin=0.1, vmax=40), cmap='viridis')
    ax[0].set_title('Before 1/f correction')
    ax[0].set_ylabel('y pixels')

    flux = data.flux.values[i]
    im1 = ax[1].imshow(flux, origin='lower', norm=LogNorm(vmin=0.1, vmax=40), cmap='viridis')
    ax[1].set_title('After 1/f correction')
    ax[1].set_xlabel('x pixels')
    ax[1].set_ylabel('y pixels')

    plt.subplots_adjust(hspace=0.3)

    cbar = fig.colorbar(im1, ax=ax)
    cbar.ax.set_ylabel('Flux (electrons)')
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(i).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fig.suptitle((f'Segment {file_number}, Integration {int_number}'), y=0.99)

    fname = (f'figs{os.sep}fig3307_file{file_number}_int{int_number}_2D_Frame_OneOverF' +
             figure_filetype)
    plt.savefig(meta.outputdir + fname, dpi=250)
    if not meta.hide_plots:
        plt.pause(0.2)


def phot_2d_frame_diff(data, meta):
    """
    Plots the difference between to consecutive 2D frames. This might be helpful in order
    to investigate flux changes due to mirror tilts which have been observed during commissioning.
    (Fig 3505)

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
    for i in range(len(data.aplev.values)-1):
        plt.figure(3505)
        plt.clf()
        plt.suptitle('2D frame differences')
        flux1 = data.flux.values[i]
        flux2 = data.flux.values[i+1]
        plt.imshow(flux2-flux1, origin='lower', vmin=-600, vmax=600)
        plt.xlim(1064-120-512, 1064+120-512)
        plt.ylim(0, flux1.shape[0])
        plt.xlabel('x pixels')
        plt.ylabel('y pixels')
        plt.colorbar(label='Delta Flux (electrons)')
        plt.tight_layout()
        int_number = str(i).zfill(int(np.floor(np.log10(meta.n_int)) + 1))
        plt.suptitle((f'Integration {int_number}'), y=0.99)
        fname = (f'figs{os.sep}fig3505_int{int_number}_2D_Frame_Diff' + figure_filetype)
        plt.savefig(meta.outputdir + fname, dpi=250)
        if not meta.hide_plots:
            plt.pause(0.2)
