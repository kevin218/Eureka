import numpy as np
import os
import matplotlib.pyplot as plt
from ..lib import util, plots

colors = ['xkcd:bright blue', 'xkcd:soft green', 'orange', 'purple']


def plot_whitelc(optspec, time, meta, i, fig=None, ax=None):
    '''Plot binned white light curve and indicate
    baseline and in-occultation regions.

    Parameters
    ----------
    optspec : Xarray DataArray
        The optimally extracted spectrum.
    time : Xarray DataArray
        The time array.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    i : int
        The occultation number.
    fig : object; optional
        The figure object. Default is None, which creates a new object.
    ax : object; optional
        The axis object. Default is None, which creates a new object.

    Returns
    -------
    fig : object
        The figure object.
    ax : object
        The axis object.
    '''
    toffset = meta.time_offset
    it0, it1, it2, it3, it4, it5 = meta.it

    # Created binned white LC
    lc = np.ma.sum(optspec, axis=1)
    lc /= np.mean(lc)
    lc_bin = util.binData_time(lc, time, nbin=meta.nbin_plot)
    time_bin = util.binData_time(time, time, nbin=meta.nbin_plot)

    if i == 0:
        fig = plt.figure(4202, figsize=(8, 5))
        plt.clf()
        ax = fig.subplots(1, 1)
        ax.plot(time_bin-toffset, lc_bin, '.', color='0.2', alpha=0.8,
                label='Binned White LC')
    ymin, ymax = ax.get_ylim()
    ax.fill_betweenx((ymin, ymax), time[it0]-toffset, time[it1]-toffset,
                     color=colors[1], alpha=0.2)
    ax.fill_betweenx((ymin, ymax), time[it4]-toffset, time[it5]-toffset,
                     color=colors[1], alpha=0.2)
    ax.fill_betweenx((ymin, ymax), time[it2]-toffset, time[it3]-toffset,
                     color=colors[0], alpha=0.2)
    ax.vlines([time[it1]-toffset, time[it4]-toffset,
              time[it0]-toffset, time[it5]-toffset],
              ymin, ymax, color=colors[1], label='Baseline Regions')
    ax.vlines([time[it2]-toffset, time[it3]-toffset],
              ymin, ymax, color=colors[0], label='In-Occultation Region')
    if i == 0:
        ax.set_ylim(ymin, ymax)
        ax.legend(loc='best')
        ax.set_xlabel(f"Time ({time.time_units})")
        ax.set_ylabel("Normalized Flux")
    fname = 'figs'+os.sep+'fig4202_WhiteLC'
    fig.savefig(meta.outputdir+fname+plots.figure_filetype,
                bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)
    return fig, ax


def plot_stellarSpec(meta, ds):
    '''Plot calibrated stellar spectra from
    baseline and in-occultation regions.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    ds : Xarray DataSet
        The DataSet object containing the extracted flux values.
    '''
    fig = plt.figure(4201, figsize=(8, 5))
    plt.clf()
    ax = fig.subplots(1, 1)
    for i in range(len(ds.time)):
        ax.errorbar(ds.wavelength, ds.base_flux[:, i], ds.base_fstd[:, i],
                    fmt='.', ms=2, label=f'Baseline ({ds.time.values[i]})')
        ax.errorbar(ds.wavelength, ds.ecl_flux[:, i], ds.ecl_fstd[:, i],
                    fmt='.', ms=2,
                    label=f'In-Occultation ({ds.time.values[i]})')

    ax.legend(loc='best')
    ax.set_xlabel(r"Wavelength ($\mu$m)")
    ax.set_ylabel(f"Flux ({ds.base_flux.flux_units})")

    fname = 'figs'+os.sep+'fig4201_CalStellarSpec'
    fig.savefig(meta.outputdir+fname+plots.figure_filetype,
                bbox_inches='tight', dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)
    return
