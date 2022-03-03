import numpy as np
import matplotlib.pyplot as plt
import corner
from scipy import stats
from copy import deepcopy

from .likelihood import computeRMS
from .utils import COLORS

def plot_fit(lc, model, meta, fitter, isTitle=True):
    """Plot the fitted model over the data.

    Parameters
    ----------
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The fitted composite model
    meta: MetaClass
        The metadata object
    fitter: str
        The name of the fitter (for plot filename)

    Returns
    -------
    None

    Notes
    -----

    History:
    - December 29, 2021 Taylor Bell
        Moved plotting code to a separate function.
    - January 7-22, 2022 Megan Mansfield
        Adding ability to do a single shared fit across all channels
    """
    if type(fitter)!=str:
        raise ValueError('Expected type str for fitter, instead received a {}'.format(type(fitter)))

    model_sys_full = model.syseval()
    model_phys_full, new_time = model.physeval(interp=meta.interp)
    model_lc = model.eval()
    
    for i, channel in enumerate(lc.fitted_channels):
        flux = np.copy(lc.flux)
        if "unc_fit" in lc.__dict__.keys():
            unc = deepcopy(lc.unc_fit)
        else:
            unc = np.copy(lc.unc)
        model = np.copy(model_lc)
        model_sys = model_sys_full
        model_phys = model_phys_full
        color = lc.colors[i]

        if lc.share:
            flux = flux[channel*len(lc.time):(channel+1)*len(lc.time)]
            unc = unc[channel*len(lc.time):(channel+1)*len(lc.time)]
            model = model[channel*len(lc.time):(channel+1)*len(lc.time)]
            model_sys = model_sys[channel*len(lc.time):(channel+1)*len(lc.time)]
            model_phys = model_phys[channel*len(new_time):(channel+1)*len(new_time)]
        
        residuals = flux - model
        fig = plt.figure(int('51{}'.format(str(channel).zfill(len(str(lc.nchannel))))), figsize=(8, 6))
        plt.clf()
        ax = fig.subplots(3,1)
        ax[0].errorbar(lc.time, flux, yerr=unc, fmt='.', color='w', ecolor=color, mec=color)
        ax[0].plot(lc.time, model, '.', ls='', ms=2, color='0.3', zorder = 10)
        if isTitle:
            ax[0].set_title(f'{meta.eventlabel} - Channel {channel} - {fitter}')
        ax[0].set_ylabel('Normalized Flux', size=14)

        ax[1].errorbar(lc.time, flux/model_sys, yerr=unc, fmt='.', color='w', ecolor=color, mec=color)
        ax[1].plot(new_time, model_phys, color='0.3', zorder = 10)
        ax[1].set_ylabel('Calibrated Flux', size=14)

        ax[2].errorbar(lc.time, residuals*1e6, yerr=unc*1e6, fmt='.', color='w', ecolor=color, mec=color)
        ax[2].plot(lc.time, np.zeros_like(lc.time), color='0.3', zorder=10)
        ax[2].set_ylabel('Residuals (ppm)', size=14)
        ax[2].set_xlabel(str(lc.time_units), size=14)

        fname = 'figs/fig51{}_lc_{}.png'.format(str(channel).zfill(len(str(lc.nchannel))), fitter)
        fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
        if meta.hide_plots:
            plt.close()
        else:
            plt.pause(0.2)
    
    return

def plot_rms(lc, model, meta, fitter):
    """Plot an Allan plot to look for red noise.

    Parameters
    ----------
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The fitted composite model
    meta: MetaClass
        The metadata object
    fitter: str
        The name of the fitter (for plot filename)

    Returns
    -------
    None

    Notes
    -----

    History:
    - December 29, 2021 Taylor Bell
        Moved plotting code to a separate function.
    - January 7-22, 2022 Megan Mansfield
        Adding ability to do a single shared fit across all channels
    - February 18, 2022 Caroline Piaulet
        Using the fitted unc for plotted, if exists
    """
    if type(fitter)!=str:
        raise ValueError('Expected type str for fitter, instead received a {}'.format(type(fitter)))

    time = lc.time
    model_lc = model.eval()

    for channel in lc.fitted_channels:
        flux = np.copy(lc.flux)
        if "unc_fit" in lc.__dict__.keys():
            unc = np.copy(lc.unc_fit)
        else:
            unc = np.copy(lc.unc)
        model = np.copy(model_lc)
        if lc.share:
            flux = flux[channel*len(lc.time):(channel+1)*len(lc.time)]
            unc = unc[channel*len(lc.time):(channel+1)*len(lc.time)]
            model = model[channel*len(lc.time):(channel+1)*len(lc.time)]
        
        residuals = flux - model
        residuals = residuals[np.argsort(time)]

        rms, stderr, binsz = computeRMS(residuals, binstep=1)
        normfactor = 1e-6
        plt.rcParams.update({'legend.fontsize': 11}) # FINDME: this should not be done here but where the rcparams are defined for Eureka
        plt.figure(int('52{}'.format(str(channel).zfill(len(str(lc.nchannel))))), figsize=(8, 6))
        plt.clf()
        plt.suptitle(' Correlated Noise', size=16)
        plt.loglog(binsz, rms / normfactor, color='black', lw=1.5, label='Fit RMS', zorder=3)  # our noise
        plt.loglog(binsz, stderr / normfactor, color='red', ls='-', lw=2, label='Std. Err.', zorder=1)  # expected noise
        plt.xlim(0.95, binsz[-1] * 2)
        plt.ylim(stderr[-1] / normfactor / 2., stderr[0] / normfactor * 2.)
        plt.xlabel("Bin Size", fontsize=14)
        plt.ylabel("RMS (ppm)", fontsize=14)
        plt.xticks(size=12)
        plt.yticks(size=12)
        plt.legend()
        fname = 'figs/fig52{}_'.format(str(channel).zfill(len(str(lc.nchannel))))+'allanplot_'+fitter+'.png'
        plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
        if meta.hide_plots:
            plt.close()
        else:
            plt.pause(0.2)

    return

def plot_corner(samples, lc, meta, freenames, fitter):
    """Plot a corner plot.

    Parameters
    ----------
    samples: ndarray
        The samples produced by the sampling algorithm
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    freenames: iterable
        The names of the fitted parameters
    meta: MetaClass
        The metadata object
    fitter: str
        The name of the fitter (for plot filename)

    Returns
    -------
    None

    Notes
    -----

    History:
    - December 29, 2021 Taylor Bell
        Moved plotting code to a separate function.
    """
    fig = plt.figure(int('53{}'.format(str(lc.channel).zfill(len(str(lc.nchannel))))), figsize=(8, 6))
    fig = corner.corner(samples, fig=fig, show_titles=True,quantiles=[0.16, 0.5, 0.84],title_fmt='.4', labels=freenames)
    fname = 'figs/fig53{}_corner_{}.png'.format(str(lc.channel).zfill(len(str(lc.nchannel))), fitter)
    fig.savefig(meta.outputdir+fname, bbox_inches='tight', pad_inches=0.05, dpi=250)
    if meta.hide_plots:
        plt.close()
    else:
        plt.pause(0.2)

    return

def plot_chain(samples, lc, meta, freenames, fitter='emcee', full=True, nburn=0):
    """Plot the evolution of the chain to look for temporal trends

    Parameters
    ----------
    samples: ndarray
        The samples produced by the sampling algorithm
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    freenames: iterable
        The names of the fitted parameters
    meta: MetaClass
        The metadata object
    fitter: str
        The name of the fitter (for plot filename)
    full:   bool
        Whether or not the samples passed in include any burn-in steps

    Returns
    -------
    None

    Notes
    -----

    History:
    - December 29, 2021 Taylor Bell
        Moved plotting code to a separate function.
    """
    if len(freenames) > 20:
        # Break the plot into many plots to avoid an enormous figure
        ndims = 20*np.ones(int(len(freenames//20)), dtype=int)
        if len(freenames)-np.sum(ndims) > 0:
            ndims = np.append(ndims, int(len(freenames)-np.sum(ndims)))
    else:
        ndims = np.array([len(freenames),])

    for plot_number, ndim in enumerate(ndims):
        fig, axes = plt.subplots(ndim, 1, num=int('55{}'.format(str(lc.channel).zfill(len(str(lc.nchannel))))), sharex=True, figsize=(6, ndim))
        
        print(plot_number)
        print(ndims)
        print(type(ndims[0]), type(ndim), type(plot_number), type(np.sum(ndims[:plot_number])), type(np.sum(ndims[:plot_number])+ndim))

        for i, j in enumerate(range(np.sum(ndims[:plot_number]), np.sum(ndims[:plot_number])+ndim)):
            axes[i].plot(samples[:, :, j], alpha=0.4)
            axes[i].set_ylabel(freenames[j])
            if full and nburn>0:
                axes[i].axvline(nburn)
        fig.tight_layout(h_pad=0.0)
        
        fname = 'figs/fig55{}'.format(str(lc.channel).zfill(len(str(lc.nchannel))))
        if full:
            fname += '_fullchain'
        else:
            fname += '_chain'
        fname += '_{}'.format(fitter)
        if len(ndims)>1:
            fname += '_plot{}'.format(plot_number)
        fname += '.png'
        fig.savefig(meta.outputdir+fname, bbox_inches='tight', pad_inches=0.05, dpi=250)
        if meta.hide_plots:
            plt.close()
        else:
            plt.pause(0.2)

    return

def plot_res_distr(lc, model, meta, fitter):
    """Plot the normalized distribution of residuals + a Gaussian

    Parameters
    ----------
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The fitted composite model
    meta: MetaClass
        The metadata object
    fitter: str
        The name of the fitter (for plot filename)
        
    Returns
    -------
    None

    Notes
    -----

    History:
    - February 18, 2022 Caroline Piaulet
        Created function
    """
    if type(fitter)!=str:
        raise ValueError('Expected type str for fitter, instead received a {}'.format(type(fitter)))

    time = lc.time
    model_lc = model.eval()

    plt.figure(int('54{}'.format(str(lc.channel).zfill(len(str(lc.nchannel))))), figsize=(8, 6))
    

    for channel in lc.fitted_channels:
        flux = np.copy(lc.flux)
        if "unc_fit" in lc.__dict__.keys():
            unc = np.copy(lc.unc_fit)
        else:
            unc = np.copy(lc.unc)
        model = np.copy(model_lc)
        if lc.share:
            flux = flux[channel*len(lc.time):(channel+1)*len(lc.time)]
            unc = unc[channel*len(lc.time):(channel+1)*len(lc.time)]
            model = model[channel*len(lc.time):(channel+1)*len(lc.time)]
        
        residuals = flux - model
        hist_vals = residuals/unc
        hist_vals[~np.isfinite(hist_vals)] = np.nan # Mask out any infinities

        n, bins, patches = plt.hist(hist_vals,alpha=0.5,color='b',edgecolor='b',lw=1)
        x=np.linspace(-4.,4.,200)
        px=stats.norm.pdf(x,loc=0,scale=1)
        plt.plot(x,px*(bins[1]-bins[0])*len(residuals),'k-',lw=2)
        plt.xlabel("Residuals/scatter", fontsize=14)
        fname = 'figs/fig54{}_'.format(str(channel).zfill(len(str(lc.nchannel))))+'res_distri_'+fitter+'.png'
        plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
        if meta.hide_plots:
            plt.close()
        else:
            plt.pause(0.2)

    return
