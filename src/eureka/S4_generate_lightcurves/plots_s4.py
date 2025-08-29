import numpy as np
import os
import matplotlib.pyplot as plt
from ..lib import util
from ..lib import plots
import warnings
warnings.filterwarnings("ignore", message='Ignoring specified arguments in '
                                          'this call because figure with num')


@plots.apply_style
def binned_lightcurve(meta, log, lc, i, white=False):
    '''Plot each spectroscopic light curve. (Figs 4102)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    lc : Xarray Dataset
        The Dataset object containing light curve and time data.
    i : int
        The current bandpass number.
    white : bool; optional
        Is this figure for the additional white-light light curve
    '''
    fig = plt.figure(4102)
    fig.set_size_inches(8, 6, forward=True)
    fig.clf()
    ax = fig.gca()
    if white:
        fig.suptitle(f'White-light Bandpass {i}: {meta.wave_min:.3f} - '
                     f'{meta.wave_max:.3f}')
        # Normalize the light curve
        norm_lcdata, norm_lcerr = util.normalize_spectrum(
            meta, lc.flux_white, lc.err_white,
            scandir=getattr(lc, 'scandir', None))
        i = 0
        fname_tag = 'white'
    elif meta.photometry:
        fig.suptitle(f'Photometric Lightcurve at {meta.phot_wave} microns')
        # Normalize the light curve
        norm_lcdata, norm_lcerr = util.normalize_spectrum(
            meta, lc['data'][i], lc['err'][i],
            scandir=getattr(lc, 'scandir', None))
        fname_tag = 'phot'
    else:
        fig.suptitle(f'Bandpass {i}: {lc.wave_low.values[i]:.3f} - '
                     f'{lc.wave_hi.values[i]:.3f}')
        # Normalize the light curve
        norm_lcdata, norm_lcerr = util.normalize_spectrum(
            meta, lc['data'][i], lc['err'][i],
            scandir=getattr(lc, 'scandir', None))
        ch_number = str(i).zfill(int(np.floor(np.log10(meta.nspecchan))+1))
        fname_tag = f'ch{ch_number}'

    time_modifier = np.floor(np.ma.min(lc.time.values))

    # Plot the normalized light curve
    if meta.inst == 'wfc3':
        for p in range(2):
            iscans = np.nonzero(lc.scandir.values == p)[0]

            if len(iscans) > 0:
                ax.errorbar(lc.time.values[iscans]-time_modifier,
                            norm_lcdata[iscans]+0.005*p,
                            norm_lcerr[iscans], fmt='o', color=f'C{p}',
                            mec=f'C{p}', alpha=0.2)
                mad = util.get_mad_1d(norm_lcdata[iscans])
                meta.mad_s4_binned.append(mad)
                log.writelog(f'    MAD = {np.round(mad).astype(int)} ppm')
                plt.text(0.05, 0.075+0.05*p,
                         f"MAD = {np.round(mad).astype(int)} ppm",
                         transform=ax.transAxes, color=f'C{p}')
    else:
        plt.errorbar(lc.time.values-time_modifier, norm_lcdata, norm_lcerr,
                     fmt='o', color=f'C{i}', mec=f'C{i}', alpha=0.2)
        mad = util.get_mad_1d(norm_lcdata)
        meta.mad_s4_binned.append(mad)
        log.writelog(f'    MAD = {np.round(mad).astype(int)} ppm')
        plt.text(0.05, 0.1, f"MAD = {np.round(mad).astype(int)} ppm",
                 transform=ax.transAxes, color='k')
    plt.ylabel('Normalized Flux')
    time_units = lc.data.attrs['time_units']
    plt.xlabel(f'Time [{time_units} - {time_modifier}]')

    fname = f'figs{os.sep}fig4102_{fname_tag}_1D_LC'+plots.get_filetype()
    fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def binned_background(meta, log, lc, i, white=False):
    '''Plot each spectroscopic background light curve. (Figs 4105)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    lc : Xarray Dataset
        The Dataset object containing light curve and time data.
    i : int
        The current bandpass number.
    white : bool; optional
        Is this figure for the additional white-light light curve
    '''
    fig = plt.figure(4105)
    fig.set_size_inches(8, 6, forward=True)
    fig.clf()
    ax = fig.gca()
    if white:
        fig.suptitle(f'White-light Bandpass {i}: {meta.wave_min:.3f} - '
                     f'{meta.wave_max:.3f}')
        # Normalize the light curve
        norm_bgdata, norm_bgerr = util.normalize_spectrum(
            meta, lc.skylev_white, lc.skyerr_white,
            scandir=getattr(lc, 'scandir', None))
        i = 0
        fname_tag = 'white'
    elif meta.photometry:
        fig.suptitle(f'Photometric Lightcurve at {meta.phot_wave} microns')
        # Normalize the light curve
        norm_bgdata, norm_bgerr = util.normalize_spectrum(
            meta, lc['skylev'][i], lc['skyerr'][i],
            scandir=getattr(lc, 'scandir', None))
        fname_tag = 'phot'
    else:
        fig.suptitle(f'Bandpass {i}: {lc.wave_low.values[i]:.3f} - '
                     f'{lc.wave_hi.values[i]:.3f}')
        # Normalize the light curve
        norm_bgdata, norm_bgerr = util.normalize_spectrum(
            meta, lc['skylev'][i], lc['skyerr'][i],
            scandir=getattr(lc, 'scandir', None))
        ch_number = str(i).zfill(int(np.floor(np.log10(meta.nspecchan))+1))
        fname_tag = f'ch{ch_number}'

    time_modifier = np.floor(np.ma.min(lc.time.values))

    # Plot the normalized light curve
    if meta.inst == 'wfc3':
        for p in range(2):
            iscans = np.nonzero(lc.scandir.values == p)[0]

            if len(iscans) > 0:
                ax.errorbar(lc.time.values[iscans]-time_modifier,
                            norm_bgdata[iscans]+0.005*p,
                            norm_bgerr[iscans], fmt='o', color=f'C{p}',
                            mec=f'C{p}', alpha=0.2)
                mad = util.get_mad_1d(norm_bgdata[iscans])
                meta.mad_s4_binned.append(mad)
                log.writelog(f'    MAD = {np.round(mad).astype(int)} ppm')
                plt.text(0.05, 0.075+0.05*p,
                         f"MAD = {np.round(mad).astype(int)} ppm",
                         transform=ax.transAxes, color=f'C{p}')
    else:
        plt.errorbar(lc.time.values-time_modifier, norm_bgdata, norm_bgerr,
                     fmt='o', color=f'C{i}', mec=f'C{i}', alpha=0.2)
        mad = util.get_mad_1d(norm_bgdata)
        meta.mad_s4_binned_bg.append(mad)
        plt.text(0.05, 0.1, f"MAD = {np.round(mad).astype(int)} ppm",
                 transform=ax.transAxes, color='k')
    plt.ylabel('Normalized Background')
    time_units = lc.data.attrs['time_units']
    plt.xlabel(f'Time [{time_units} - {time_modifier}]')

    fname = f'figs{os.sep}fig4105_{fname_tag}_1D_BG'+plots.get_filetype()
    fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def driftxpos(meta, lc):
    '''Plot the 1D drift/jitter results. (Fig 4103)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    lc : Xarray Dataset
        The light curve object containing drift arrays.
    '''
    fig = plt.figure(4103)
    fig.set_size_inches(8, 4, forward=True)
    fig.clf()
    mask = lc.driftmask
    plt.plot(np.arange(meta.n_int)[~mask],
             lc.centroid_x[~mask], '.',
             label='Good Drift Points')
    plt.plot(np.arange(meta.n_int)[mask],
             lc.centroid_x[mask], '.',
             label='Interpolated Drift Points')
    plt.ylabel('Spectrum Drift Along x')
    plt.xlabel('Frame Number')
    plt.legend(loc='best')

    fname = 'figs'+os.sep+'fig4103_DriftXPos'+plots.get_filetype()
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def driftxwidth(meta, lc):
    '''Plot the 1D drift width results. (Fig 4104)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    lc : Xarray Dataset
        The light curve object containing drift arrays.
    '''
    fig = plt.figure(4104)
    fig.set_size_inches(8, 4, forward=True)
    fig.clf()
    mask = lc.driftmask
    plt.plot(np.arange(meta.n_int)[~mask],
             lc.centroid_sx[~mask], '.',
             label='Good Drift Points')
    plt.plot(np.arange(meta.n_int)[mask],
             lc.centroid_sx[mask], '.',
             label='Interpolated Drift Points')
    plt.ylabel('Spectrum Drift CC Width Along x')
    plt.xlabel('Frame Number')
    plt.legend(loc='best')

    fname = 'figs'+os.sep+'fig4104_DriftXWidth'+plots.get_filetype()
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def lc_driftcorr(meta, wave_1d, optspec_in, optmask=None, scandir=None):
    '''Plot a 2D light curve with drift correction. (Fig 4101)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    wave_1d : ndarray
        Wavelength array with trimmed edges depending on xwindow and ywindow
        which have been set in the S3 ecf.
    optspec_in : Xarray DataArray
        The optimally extracted spectrum.
    optmask : Xarray DataArray; optional
        A mask array to use if optspec is not a masked array. Defaults to None
        in which case only the invalid values of optspec will be masked.
    scandir : ndarray; optional
        For HST spatial scanning mode, 0=forward scan and 1=reverse scan.
        Defaults to None which is fine for JWST data, but must be provided
        for HST data (can be all zero values if not spatial scanning mode).
    '''
    optspec = np.ma.masked_invalid(optspec_in.values)
    optspec = np.ma.masked_where(optmask.values, optspec)

    wmin = meta.wave_min
    wmax = meta.wave_max
    iwmin = np.nanargmin(np.abs(wave_1d-wmin))
    iwmax = np.nanargmin(np.abs(wave_1d-wmax))

    # Normalize the light curve
    norm_lcdata = util.normalize_spectrum(meta, optspec[:, iwmin:iwmax],
                                          scandir=scandir)

    if meta.time_axis not in ['y', 'x']:
        print("WARNING: meta.time_axis is not one of ['y', 'x']!"
              " Using 'y' by default.")
        meta.time_axis = 'y'

    cmap = plt.cm.RdYlBu_r
    fig = plt.figure(4101)
    fig.set_size_inches(8, 8, forward=True)
    fig.clf()
    if meta.time_axis == 'y':
        plt.pcolormesh(wave_1d[iwmin:iwmax], np.arange(meta.n_int)+0.5,
                       norm_lcdata, vmin=meta.vmin, vmax=meta.vmax,
                       cmap=cmap)
        plt.xlim(meta.wave_min, meta.wave_max)
        plt.ylim(0, meta.n_int)
        plt.ylabel('Integration Number')
        plt.xlabel(r'Wavelength ($\mu m$)')
        plt.colorbar(label='Normalized Flux')

        if len(meta.wave) > 1 and len(wave_1d) != meta.nspecchan:
            # Insert vertical dashed lines at spectroscopic channel edges
            secax = plt.gca().secondary_xaxis('top')
            xticks = np.unique(np.concatenate([meta.wave_low, meta.wave_hi]))
            xticks_labels = [f'{np.round(xtick, 3):.3f}' for xtick in xticks]
            secax.set_xticks(xticks, xticks_labels, rotation=90,
                             fontsize='xx-small')
            plt.vlines(xticks, 0, meta.n_int, '0.3', 'dashed')
    else:
        plt.pcolormesh(np.arange(meta.n_int), wave_1d[iwmin:iwmax],
                       norm_lcdata.swapaxes(0, 1), vmin=meta.vmin,
                       vmax=meta.vmax, cmap=cmap)
        plt.ylim(meta.wave_min, meta.wave_max)
        plt.xlim(0, meta.n_int)
        plt.ylabel(r'Wavelength ($\mu m$)')
        plt.xlabel('Integration Number')
        plt.colorbar(label='Normalized Flux', pad=0.075)

        if len(meta.wave_low) > 1 and len(wave_1d) != meta.nspecchan:
            # Insert vertical dashed lines at spectroscopic channel edges
            secax = plt.gca().secondary_yaxis('right')
            yticks = np.unique(np.concatenate([meta.wave_low, meta.wave_hi]))
            yticks_labels = [f'{np.round(ytick, 3):.3f}' for ytick in yticks]
            secax.set_yticks(yticks, yticks_labels, rotation=0,
                             fontsize='xx-small')
            plt.hlines(yticks, 0, meta.n_int, '0.3', 'dashed')

    plt.minorticks_on()

    plt.title(f"MAD = {np.round(meta.mad_s4).astype(int)} ppm")

    fname = 'figs'+os.sep+'fig4101_2D_LC'+plots.get_filetype()
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.2)

    return


@plots.apply_style
def mad_outliers(meta, pp):
    '''Plot spectroscopic MAD values and identify outliers. (Figs 4106)
    Outliers will be appended to `mask_columns` in the Stage 4 ECF.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    pp : Dictionary
        A dictionary of plotting parameters from outliers.get_outliers().
    '''
    # Unpack dictionary of plotting parameters
    x = pp["x"]
    x_mask = pp["x_mask"]
    x_mad_outliers = pp["x_mad_outliers"]
    x_dev_outliers = pp["x_dev_outliers"]
    mad = pp["mad"]
    dev = pp["dev"]
    masked_mad = pp["masked_mad"]
    masked_dev = pp["masked_dev"]
    smoothed_mad = pp["smoothed_mad"]
    residual_mad = pp["residual_mad"]
    smoothed_dev = pp["smoothed_dev"]
    residual_dev = pp["residual_dev"]

    # Plot spectroscopic MAD values
    alpha = 0.5
    fig = plt.figure(4106)
    fig.set_size_inches(8, 8, forward=True)
    fig.clf()
    plt.subplot(211)
    plt.plot(x, mad, '.', color='b', zorder=1,
             label="Unbinned LC MAD", alpha=alpha)
    plt.plot(x, dev, '.', color='gold', zorder=2,
             label="Deviation from white LC (Scaled)", alpha=alpha)
    plt.plot(x_mask, smoothed_mad, '-', color='r', lw=2, zorder=5,
             label="Unbinned LC MAD (Smoothed)")
    plt.plot(x_mask, smoothed_dev, '--', color='0.3', lw=2, zorder=6,
             label="Deviation from white LC (Smoothed)")
    plt.ylabel('MAD Value (ppm)')
    plt.legend(loc='best')
    plt.subplot(212)
    plt.plot(x_mask, residual_mad, '.', color='b', zorder=1,
             label="Unbinned LC MAD", alpha=alpha)
    plt.plot(x_mask, residual_dev, '.', color='gold', zorder=2,
             label="Deviation from white LC", alpha=alpha)
    plt.plot(x_mad_outliers, residual_mad[masked_mad.mask], '.', color='C3',
             zorder=5)
    plt.plot(x_dev_outliers, residual_dev[masked_dev.mask], '.', color='C3',
             zorder=5, label=rf"{meta.mad_sigma}$\sigma$ Outliers")
    plt.legend(loc='best')
    plt.ylabel('Residuals (ppm)')
    plt.xlabel('Detector Column Number')
    fname = 'figs'+os.sep+'fig4106_MAD_Outliers'+plots.get_filetype()
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.1)

    return


@plots.apply_style
def cc_spec(meta, ref_spec, fit_spec, n):
    '''Compare the spectrum used for cross-correlation with the current
    spectrum (Fig 4301).

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    ref_spec : ndarray (1D)
        The reference spectrum used for cross-correlation.
    fit_spec : ndarray (1D)
        The extracted spectrum for the current integration.
    n : int
        The current integration number.
    '''
    fig = plt.figure(4301)
    fig.set_size_inches(8, 8, forward=True)
    fig.clf()
    plt.title(f'Cross Correlation - Spectrum {n}')
    nx = len(ref_spec)
    plt.plot(np.arange(nx), ref_spec, '-', label='Reference Spectrum')
    plt.plot(np.arange(meta.drift_range, nx-meta.drift_range), fit_spec, '-',
             label='Current Spectrum')
    plt.legend(loc='best')

    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = ('figs'+os.sep+f'fig4301_int{int_number}_CC_Spec' +
             plots.get_filetype())
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def cc_vals(meta, vals, n):
    '''Make the cross-correlation strength plot (Fig 4302).

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    vals : ndarray (1D)
        The cross-correlation strength.
    n : int
        The current integration number.
    '''
    fig = plt.figure(4302)
    fig.set_size_inches(8, 8, forward=True)
    fig.clf()
    plt.title(f'Cross Correlation - Values {n}')
    plt.plot(np.arange(-meta.drift_range, meta.drift_range+1), vals, '.')

    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = ('figs'+os.sep+f'fig4302_int{int_number}_CC_Vals' +
             plots.get_filetype())
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


@plots.apply_style
def plot_extrapolated_throughput(meta, throughput_wavelengths, throughput,
                                 wav_poly, throughput_poly, mode):
    '''Make the extrapolated throughput plot (Fig 4303).

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    throughput_wavelengths : ndarray
        The throughput wavelengths from ExoTiC-LD. Units of Angstroms.
    throughput : ndarray
        The ethroughput wavelengths from ExoTiC-LD. Unitless, spanning 0-1.
    wav_poly : ndarray
        The throughput wavelengths where extrapolating beyond the ExoTiC-LD
        wavelenths. Units of Angstroms.
    throughput_poly : ndarray
        The extrapolated throughput values. Unitless, spanning 0-1.
    mode : str
        The string describing the observatory, instrument, and filter combo,
        from the supported list at
        https://exotic-ld.readthedocs.io/en/latest/views/supported_instruments.html
    '''
    fig = plt.figure(4303)
    fig.set_size_inches(8, 8, forward=True)
    fig.clf()
    plt.title(mode)
    plt.plot(throughput_wavelengths/1e4, 100*throughput,
             label='ExoTiC-LD Throughput')
    plt.plot(wav_poly/1e4, 100*throughput_poly, zorder=1,
             label='Extrapolated Throughput')
    plt.xlim(min([throughput_wavelengths[0], wav_poly[0]])/1e4,
             max([throughput_wavelengths[-1], wav_poly[-1]])/1e4,)
    plt.ylim(0)
    plt.ylabel('Throughput (%)')
    plt.xlabel('Wavelength ($\\mu$m)')
    plt.legend(loc='best')

    fname = ('figs'+os.sep+'fig4303_ExtrapolatedThroughput' +
             plots.get_filetype())
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)
