import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from .source_pos import gauss
from ..lib import util
from ..lib.plots import figure_filetype


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
    optmask : ndarray (1D), optional
        A mask array to use if optspec is not a masked array. Defaults to None
        in which case only the invalid values of optspec will be masked.

    Returns
    -------
    None
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
              "Using 'y' by default.")
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

    Returns
    -------
    None
    '''
    log.writelog('  Creating figures for background subtraction...',
                 mute=(not meta.verbose))

    intstart = data.attrs['intstart']
    subdata = np.ma.masked_where(~data.mask.values, data.flux.values)
    subbg = np.ma.masked_where(~data.mask.values, data.bg.values)

    xmin, xmax = data.flux.x.min().values, data.flux.x.max().values
    ymin, ymax = data.flux.y.min().values, data.flux.y.max().values

    iterfn = range(meta.n_int)
    if meta.verbose:
        iterfn = tqdm(iterfn)
    for n in iterfn:
        plt.figure(3301, figsize=(8, 8))
        plt.clf()
        plt.suptitle(f'Integration {intstart + n}')
        plt.subplot(211)
        plt.title('Background-Subtracted Flux')
        max = np.ma.max(subdata[n])
        plt.imshow(subdata[n], origin='lower', aspect='auto',
                   vmin=0, vmax=max/10, extent=[xmin, xmax, ymin, ymax])
        plt.colorbar()
        plt.ylabel('Detector Pixel Position')
        plt.subplot(212)
        plt.title('Subtracted Background')
        median = np.ma.median(subbg[n])
        std = np.ma.std(subbg[n])
        plt.imshow(subbg[n], origin='lower', aspect='auto', vmin=median-3*std,
                   vmax=median+3*std, extent=[xmin, xmax, ymin, ymax])
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
    '''Plot the fitted 2D drift. (Fig 3102)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    '''
    plt.figure(3102, figsize=(8, 6))
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
    plt.xlabel('Frame Number')
    plt.tight_layout()
    fname = f'figs{os.sep}fig3102_Drift2D{figure_filetype}'
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

    Returns
    -------
    None
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


def source_position(meta, x_dim, pos_max, m,
                    isgauss=False, x=None, y=None, popt=None,
                    isFWM=False, y_pixels=None, sum_row=None, y_pos=None):
    '''Plot source position for MIRI data. (Figs 3303)

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

    Returns
    -------
    None

    Notes
    -----
    History:

    - 2021-07-14: Sebastian Zieba
        Initial version.
    - Oct 15, 2021: Taylor Bell
        Tidied up the code a bit to reduce repeated code.
    '''
    plt.figure(3303)
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
    fname = ('figs'+os.sep+f'fig3303_file{file_number}_source_pos' +
             figure_filetype)
    plt.savefig(meta.outputdir+fname, dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def profile(meta, profile, submask, n, m):
    '''Plot weighting profile from optimal spectral extraction routine. (Figs 3304)

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

    Returns
    -------
    None
    '''
    profile = np.ma.masked_invalid(profile)
    submask = np.ma.masked_invalid(submask)
    mask = np.logical_or(np.ma.getmaskarray(profile),
                         np.ma.getmaskarray(submask))
    profile = np.ma.masked_where(mask, profile)
    submask = np.ma.masked_where(mask, submask)
    vmax = 0.05*np.ma.max(profile*submask)
    vmin = np.ma.min(profile*submask)
    plt.figure(3304)
    plt.clf()
    plt.suptitle(f"Profile - Integration {n}")
    plt.imshow(profile*submask, aspect='auto', origin='lower',
               vmax=vmax, vmin=vmin)
    plt.ylabel('Relative Pixel Postion')
    plt.xlabel('Relative Pixel Position')
    plt.tight_layout()
    file_number = str(m).zfill(int(np.floor(np.log10(meta.num_data_files))+1))
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = (f'figs{os.sep}fig3304_file{file_number}_int{int_number}_Profile' +
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

    Returns
    -------
    None
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
    '''Plot the spatial jitter. (Fig 3305)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    None
    
    Notes
    -----
    History:
    
    - 2022-07-11 Caroline Piaulet
        First version of this function
    '''
    plt.figure(3305, figsize=(8, 4))
    plt.clf()
    plt.plot(np.arange(meta.n_int), data["driftypos"].values, '.')
    plt.ylabel('Spectrum spatial profile center')
    plt.xlabel('Frame Number')
    plt.tight_layout()
    fname = 'figs'+os.sep+'fig3305_DriftYPos'+figure_filetype
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def driftywidth(data, meta):
    '''Plot the spatial profile's fitted Gaussian width. (Fig 3306)

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    None
    
    Notes
    -----
    History:
    
    - 2022-07-11 Caroline Piaulet
        First version of this function
    '''
    plt.figure(3306, figsize=(8, 4))
    plt.clf()
    plt.plot(np.arange(meta.n_int), data["driftywidth"].values, '.')
    plt.ylabel('Spectrum spatial profile width')
    plt.xlabel('Frame Number')
    plt.tight_layout()
    fname = 'figs'+os.sep+'fig3306_DriftYWidth'+figure_filetype
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)