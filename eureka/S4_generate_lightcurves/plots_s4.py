import numpy as np
import matplotlib.pyplot as plt
from ..lib import util
from ..lib.plots import figure_filetype

def binned_lightcurve(meta, i):
    '''Plot each spectroscopic light curve. (Figs 4102)

    Parameters
    ----------
    meta:   MetaClass
        The metadata object.
    i:  int
        The current bandpass number.

    Returns
    -------
    None
    '''
    plt.figure(4102, figsize=(8, 6))
    plt.clf()
    plt.suptitle(f"Bandpass {i}: %.3f - %.3f" % (meta.wave_low[i], meta.wave_hi[i]))
    ax = plt.subplot(111)
    time_modifier = np.ma.floor(meta.time[0])
    # Normalized light curve
    norm_lcdata = meta.lcdata[i] / np.ma.mean(meta.lcdata[i,:])
    norm_lcerr = meta.lcerr[i] / np.ma.mean(meta.lcdata[i,:])
    plt.errorbar(meta.time - time_modifier, norm_lcdata, norm_lcerr, fmt='o', color=f'C{i}', mec=f'C{i}', alpha = 0.2)
    mad = util.get_mad_1d(norm_lcdata)
    plt.text(0.05, 0.1, f"MAD = {np.round(mad).astype(int)} ppm", transform=ax.transAxes, color='k')
    plt.ylabel('Normalized Flux')
    plt.xlabel(f'Time [{meta.time_units} - {time_modifier}]')

    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.90, hspace=0.20, wspace=0.3)
    ch_number = str(i).zfill(int(np.floor(np.log10(meta.nspecchan))+1))
    plt.savefig(meta.outputdir + f'figs/Fig4102_ch{ch_number}_1D_LC'+figure_filetype, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)

def drift1d(meta):
    '''Plot the 1D drift/jitter results. (Fig 4103)

    Parameters
    ----------
    meta:   MetaClass
        The metadata object.

    Returns
    -------
    None
    '''
    plt.figure(4103, figsize=(8, 4))
    plt.clf()
    plt.plot(np.arange(meta.n_int)[np.where(meta.driftmask)], meta.drift1d[np.where(meta.driftmask)], '.')
    plt.ylabel('Spectrum Drift Along x')
    plt.xlabel('Frame Number')
    plt.tight_layout()
    plt.savefig(meta.outputdir + 'figs/Fig4103_Drift'+figure_filetype, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)

def lc_driftcorr(meta, wave_1d, optspec):
    '''Plot a 2D light curve with drift correction. (Fig 4101)

    Parameters
    ----------
    meta:   MetaClass
        The metadata object.
    wave_1d:
        Wavelength array with trimmed edges depending on xwindow and ywindow which have been set in the S3 ecf
    optspec:
        The optimally extracted spectrum.

    Returns
    -------
    None
    '''
    plt.figure(4101, figsize=(8, 8))
    plt.clf()
    wmin = np.ma.min(wave_1d)
    wmax = np.ma.max(wave_1d)
    n_int, nx = optspec.shape
    vmin = 0.97
    vmax = 1.03
    normspec = optspec / np.ma.mean(optspec, axis=0)
    plt.imshow(normspec, origin='lower', aspect='auto', extent=[wmin, wmax, 0, n_int], vmin=vmin, vmax=vmax,
               cmap=plt.cm.RdYlBu_r)
    plt.title("MAD = " + str(np.round(meta.mad_s4).astype(int)) + " ppm")
    if meta.nspecchan > 1:
        # Insert vertical dashed lines at spectroscopic channel edges
        secax = plt.gca().secondary_xaxis('top')
        xticks = np.unique(np.concatenate([meta.wave_low,meta.wave_hi]))
        secax.set_xticks(xticks, np.round(xticks, 6), rotation=90, fontsize='xx-small')
        plt.vlines(xticks,0,n_int,'0.3','dashed')
    plt.ylabel('Integration Number')
    plt.xlabel(r'Wavelength ($\mu m$)')
    plt.colorbar(label='Normalized Flux')
    plt.tight_layout()
    plt.savefig(meta.outputdir + 'figs/Fig4101_2D_LC'+figure_filetype, bbox_inches='tight', dpi=300)
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.2)
    return

def cc_spec(meta, ref_spec, fit_spec, n):
    '''Compare the spectrum used for cross-correlation with the current spectrum (Fig 4301).

    Parameters
    ----------
    meta:   MetaClass
        The metadata object.
    ref_spec:   ndarray (1D)
        The reference spectrum used for cross-correlation.
    fit_spec:   ndarray (1D)
        The extracted spectrum for the current integration.
    n:  int
        The current integration number.

    Returns
    -------
    None
    '''
    plt.figure(4301, figsize=(8, 8))
    plt.clf()
    plt.title(f'Cross Correlation - Spectrum {n}')
    nx = len(ref_spec)
    plt.plot(np.arange(nx), ref_spec, '-', label='Reference Spectrum')
    plt.plot(np.arange(meta.drift_range,nx-meta.drift_range), fit_spec, '-', label='Current Spectrum')
    plt.legend(loc='best')
    plt.tight_layout()
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    plt.savefig(meta.outputdir + f'figs/Fig4301_int{int_number}_CC_Spec'+figure_filetype, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)

def cc_vals(meta, vals, n):
    '''Make the cross-correlation strength plot (Fig 4302).

    Parameters
    ----------
    meta:   MetaClass
        The metadata object.
    vals:   ndarray (1D)
        The cross-correlation strength.
    n:  int
        The current integration number.

    Returns
    -------
    None
    '''
    plt.figure(4302, figsize=(8, 8))
    plt.clf()
    plt.title(f'Cross Correlation - Values {n}')
    plt.plot(np.arange(-meta.drift_range,meta.drift_range+1), vals, '.')
    plt.tight_layout()
    int_number = str(n).zfill(int(np.floor(np.log10(meta.n_int))+1))
    plt.savefig(meta.outputdir + f'figs/Fig4302_int{int_number}_CC_Vals'+figure_filetype, bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)
