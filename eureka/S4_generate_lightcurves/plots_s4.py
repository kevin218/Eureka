import numpy as np
import matplotlib.pyplot as plt


def binned_lightcurve(meta, time, i):
    '''Plot each spectroscopic light curve.

    Parameters
    ----------
    meta:   MetaClass
        The metadata object.
    time:  ndarray (1D)
        The time in meta.time_units of each data point.
    i:  int
        The current bandpass number.

    Returns
    -------
    None
    '''
    plt.figure(4300 + i, figsize=(8, 6))
    plt.clf()
    plt.suptitle(f"Bandpass {i}: %.3f - %.3f" % (meta.wave_low[i], meta.wave_hi[i]))
    ax = plt.subplot(111)
    time_modifier = np.floor(time[0])
    # Normalized light curve
    norm_lcdata = meta.lcdata[i] / meta.lcdata[i, -1]
    norm_lcerr = meta.lcerr[i] / meta.lcdata[i, -1]
    plt.errorbar(time - time_modifier, norm_lcdata, norm_lcerr, fmt='o', color=f'C{i}', mec='w')
    plt.text(0.05, 0.1, "MAD = " + str(np.round(1e6 * np.ma.median(np.abs(np.ediff1d(norm_lcdata))))) + " ppm",
             transform=ax.transAxes, color='k')
    plt.ylabel('Normalized Flux')
    plt.xlabel(f'Time [{meta.time_units} - {time_modifier}]')

    plt.subplots_adjust(left=0.10, right=0.95, bottom=0.10, top=0.90, hspace=0.20, wspace=0.3)
    plt.savefig(meta.outputdir + 'figs/Fig' + str(4300 + i) + '-' + meta.eventlabel + '-1D_LC.png')
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.2)

def drift1d(meta):
    '''Plot the 1D drift/jitter results.

    Parameters
    ----------
    meta:   MetaClass
        The metadata object.

    Returns
    -------
    None
    '''
    plt.figure(4101, figsize=(8,4))
    plt.clf()
    plt.plot(np.arange(meta.n_int)[np.where(meta.driftmask)], meta.drift1d[np.where(meta.driftmask)], '.')
    # plt.subplot(211)
    # for j in range(istart,ev.n_reads-1):
    #     plt.plot(ev.drift2D[:,j,1],'.')
    # plt.ylabel('Spectrum Drift Along y')
    # plt.subplot(212)
    # for j in range(istart,ev.n_reads-1):
    #     plt.plot(ev.drift2D[:,j,0]+ev.drift[:,j],'.')
    plt.ylabel('Spectrum Drift Along x')
    plt.xlabel('Frame Number')
    plt.tight_layout()
    plt.savefig(meta.outputdir + 'figs/Fig4101-Drift.png')
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.2)

def lc_driftcorr(meta, wave_1d, optspec):
    '''Plot a 2D light curve with drift correction.

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
    plt.figure(4102, figsize=(8, 8))  # ev.n_files/20.+0.8))
    plt.clf()
    wmin = wave_1d.min()
    wmax = wave_1d.max()
    n_int, nx = optspec.shape
    vmin = 0.97
    vmax = 1.03
    normspec = optspec / np.mean(optspec, axis=0)
    plt.imshow(normspec, origin='lower', aspect='auto', extent=[wmin, wmax, 0, n_int], vmin=vmin, vmax=vmax,
               cmap=plt.cm.RdYlBu_r)
    ediff = np.zeros(n_int)
    for m in range(n_int):
        ediff[m] = 1e6 * np.median(np.abs(np.ediff1d(normspec[m])))
    plt.title("MAD = " + str(np.round(np.mean(ediff), 0).astype(int)) + " ppm")
    if meta.nspecchan > 1:
        # Insert vertical dashed lines at spectroscopic channel edges
        xticks = np.unique(np.concatenate([meta.wave_low,meta.wave_hi]))
        plt.xticks(xticks, xticks, rotation=90)
        plt.vlines(xticks,0,n_int,'0.3','dashed')
    plt.ylabel('Integration Number')
    plt.xlabel(r'Wavelength ($\mu m$)')
    plt.colorbar(label='Normalized Flux')
    plt.tight_layout()
    plt.savefig(meta.outputdir + 'figs/fig4102-2D_LC.png')
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.2)

def cc_spec(meta, ref_spec, fit_spec, n):
    '''Compare the spectrum used for cross-correlation with the current spectrum.

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
    plt.figure(4500)
    plt.clf()
    plt.title(f'Cross Correlation - Spectrum {n}')
    nx = len(ref_spec)
    plt.plot(range(nx), ref_spec, '-', label='Reference Spectrum')
    plt.plot(range(meta.drift_range,nx-meta.drift_range), fit_spec, '-', label='Current Spectrum')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(meta.outputdir + f'figs/Fig4500-{n}-CC_Spec')
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.2)

def cc_vals(meta, vals, n):
    '''Make the cross-correlation strength plot.

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
    plt.figure(4501)
    plt.clf()
    plt.title(f'Cross Correlation - Values {n}')
    plt.plot(range(-meta.drift_range,meta.drift_range+1), vals, '.')
    plt.tight_layout()
    plt.savefig(meta.outputdir + f'figs/Fig4501-{n}-CC_Vals')
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.2)
