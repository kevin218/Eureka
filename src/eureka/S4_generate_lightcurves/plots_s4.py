import numpy as np
import os
import matplotlib.pyplot as plt
from ..lib import util
from ..lib.plots import figure_filetype


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
    white : bool, optional
        Is this figure for the additional white-light light curve
    '''
    fig = plt.figure(4102, figsize=(8, 6))
    fig.clf()
    ax = fig.gca()
    if white:
        fig.suptitle(f'White-light Bandpass {i}: {meta.wave_min:.3f} - '
                     f'{meta.wave_max:.3f}')
        # Normalize the light curve
        norm_lcdata, norm_lcerr = util.normalize_spectrum(
            meta, lc.flux_white, lc.err_white)
        i = 0
        fname_tag = 'white'
    else:
        fig.suptitle(f'Bandpass {i}: {lc.wave_low.values[i]:.3f} - '
                     f'{lc.wave_hi.values[i]:.3f}')
        # Normalize the light curve
        norm_lcdata, norm_lcerr = util.normalize_spectrum(meta, lc['data'][i],
                                                          lc['err'][i])
        ch_number = str(i).zfill(int(np.floor(np.log10(meta.nspecchan))+1))
        fname_tag = f'ch{ch_number}'
        
    time_modifier = np.floor(np.ma.min(lc.time.values))
    
    # Plot the normalized light curve
    if meta.inst == 'wfc3':
        for p in range(2):
            iscans = np.where(lc.scandir.values == p)[0]

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

    fig.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.90,
                        hspace=0.20, wspace=0.3)
    fname = f'figs{os.sep}Fig4102_{fname_tag}_1D_LC'+figure_filetype
    fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def driftxpos(meta, lc):
    '''Plot the 1D drift/jitter results. (Fig 4103)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    lc : Xarray Dataset
        The light curve object containing drift arrays.
    
    Notes
    -----
    History:

    - Jul 11, 2022 Caroline Piaulet
        Edited this function to use the new naming convention for drift
    '''
    plt.figure(4103, figsize=(8, 4))
    plt.clf()
    plt.plot(np.arange(meta.n_int)[np.where(~lc.driftmask)],
             lc.driftxpos[np.where(~lc.driftmask)], '.',
             label='Good Drift Points')
    plt.plot(np.arange(meta.n_int)[np.where(lc.driftmask)],
             lc.driftxpos[np.where(lc.driftmask)], '.',
             label='Interpolated Drift Points')
    plt.ylabel('Spectrum Drift Along x')
    plt.xlabel('Frame Number')
    plt.legend(loc='best')
    plt.tight_layout()
    fname = 'figs'+os.sep+'fig4103_DriftXPos'+figure_filetype
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def driftxwidth(meta, lc):
    '''Plot the 1D drift width results. (Fig 4104)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    lc : Xarray Dataset
        The light curve object containing drift arrays.

    Returns
    -------
    None
    
    Notes
    -----
    History:

    - Jul 11, 2022 Caroline Piaulet
        Created this function
    '''
    plt.figure(4104, figsize=(8, 4))
    plt.clf()
    plt.plot(np.arange(meta.n_int)[np.where(~lc.driftmask)],
             lc.driftxwidth[np.where(~lc.driftmask)], '.',
             label='Good Drift Points')
    plt.plot(np.arange(meta.n_int)[np.where(lc.driftmask)],
             lc.driftxwidth[np.where(lc.driftmask)], '.',
             label='Interpolated Drift Points')
    plt.ylabel('Spectrum Drift CC Width Along x')
    plt.xlabel('Frame Number')
    plt.legend(loc='best')
    plt.tight_layout()
    fname = 'figs'+os.sep+'fig4104_DriftXWidth'+figure_filetype
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


def lc_driftcorr(meta, wave_1d, optspec, optmask=None):
    '''Plot a 2D light curve with drift correction. (Fig 4101)

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    wave_1d : ndarray
        Wavelength array with trimmed edges depending on xwindow and ywindow
        which have been set in the S3 ecf.
    optspec : ndarray
        The optimally extracted spectrum.
    optmask : ndarray (1D), optional
        A mask array to use if optspec is not a masked array. Defaults to None
        in which case only the invalid values of optspec will be masked.
    '''
    optspec = np.ma.masked_invalid(optspec)
    optspec = np.ma.masked_where(optmask, optspec)
    
    wmin = meta.wave_min
    wmax = meta.wave_max
    iwmin = np.nanargmin(np.abs(wave_1d-wmin).values)
    iwmax = np.nanargmin(np.abs(wave_1d-wmax).values)

    # Normalize the light curve
    norm_lcdata = util.normalize_spectrum(meta, optspec[:, iwmin:iwmax])

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
    
    plt.figure(4101, figsize=(8, 8))
    plt.clf()
    if meta.time_axis == 'y':
        plt.imshow(norm_lcdata, origin='lower', aspect='auto',
                   extent=[wmin, wmax, 0, meta.n_int], vmin=meta.vmin,
                   vmax=meta.vmax, cmap=plt.cm.RdYlBu_r)
        plt.ylabel('Integration Number')
        plt.xlabel(r'Wavelength ($\mu m$)')
        plt.colorbar(label='Normalized Flux')

        if len(meta.wave_low) > 1:
            # Insert vertical dashed lines at spectroscopic channel edges
            secax = plt.gca().secondary_xaxis('top')
            xticks = np.unique(np.concatenate([meta.wave_low, meta.wave_hi]))
            xticks_labels = [f'{np.round(xtick, 4):.4f}' for xtick in xticks]
            secax.set_xticks(xticks, xticks_labels, rotation=90,
                             fontsize='xx-small')
            plt.vlines(xticks, 0, meta.n_int, '0.3', 'dashed')
    else:
        plt.imshow(norm_lcdata.swapaxes(0, 1), origin='lower', aspect='auto',
                   extent=[0, meta.n_int, wmin, wmax], vmin=meta.vmin,
                   vmax=meta.vmax, cmap=plt.cm.RdYlBu_r)
        plt.ylabel(r'Wavelength ($\mu m$)')
        plt.xlabel('Integration Number')
        plt.colorbar(label='Normalized Flux', pad=0.075)

        if len(meta.wave_low) > 1:
            # Insert vertical dashed lines at spectroscopic channel edges
            secax = plt.gca().secondary_yaxis('right')
            yticks = np.unique(np.concatenate([meta.wave_low, meta.wave_hi]))
            yticks_labels = [f'{np.round(ytick, 4):.4f}' for ytick in yticks]
            secax.set_yticks(yticks, yticks_labels, rotation=0,
                             fontsize='xx-small')
            plt.hlines(yticks, 0, meta.n_int, '0.3', 'dashed')

    plt.title(f"MAD = {np.round(meta.mad_s4).astype(int)} ppm")
    plt.tight_layout()
    fname = 'figs'+os.sep+'fig4101_2D_LC'+figure_filetype
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.2)

    return


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
    plt.figure(4301, figsize=(8, 8))
    plt.clf()
    plt.title(f'Cross Correlation - Spectrum {n}')
    nx = len(ref_spec)
    plt.plot(np.arange(nx), ref_spec, '-', label='Reference Spectrum')
    plt.plot(np.arange(meta.drift_range, nx-meta.drift_range), fit_spec, '-',
             label='Current Spectrum')
    plt.legend(loc='best')
    plt.tight_layout()
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = 'figs'+os.sep+f'fig4301_int{int_number}_CC_Spec'+figure_filetype
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)


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
    plt.figure(4302, figsize=(8, 8))
    plt.clf()
    plt.title(f'Cross Correlation - Values {n}')
    plt.plot(np.arange(-meta.drift_range, meta.drift_range+1), vals, '.')
    plt.tight_layout()
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    fname = 'figs'+os.sep+f'fig4302_int{int_number}_CC_Vals'+figure_filetype
    plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)
