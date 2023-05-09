import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import corner
from scipy import stats
try:
    import arviz as az
    from arviz.rcparams import rcParams as az_rcParams
except:
    # PyMC3 hasn't been installed
    pass

from .likelihood import computeRMS
from ..lib import plots, util
from ..lib.split_channels import split


def plot_fit(lc, model, meta, fitter, isTitle=True):
    """Plot the fitted model over the data. (Figs 5101)

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The fitted composite model.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    fitter : str
        The name of the fitter (for plot filename).
    isTitle : bool; optional
        Should figure have a title. Defaults to True.

    Notes
    -----
    History:

    - December 29, 2021 Taylor Bell
        Moved plotting code to a separate function.
    - January 7-22, 2022 Megan Mansfield
        Adding ability to do a single shared fit across all channels
    - February 28-March 1, 2022 Caroline Piaulet
        Adding scatter_ppm parameter
    """
    if type(fitter) != str:
        raise ValueError(f'Expected type str for fitter, instead received a '
                         f'{type(fitter)}')

    model_sys_full = model.syseval()
    model_phys_full, new_time, nints_interp = \
        model.physeval(interp=meta.interp)
    model_lc = model.eval()

    for i, channel in enumerate(lc.fitted_channels):
        flux = np.ma.copy(lc.flux)
        unc = np.ma.copy(lc.unc_fit)
        model = np.ma.copy(model_lc)
        model_sys = model_sys_full
        model_phys = model_phys_full
        color = lc.colors[i]

        if lc.share and not meta.multwhite:
            time = lc.time
            new_timet = new_time

            # Split the arrays that have lengths of the original time axis
            flux, unc, model, model_sys = split([flux, unc, model, model_sys],
                                                meta.nints, channel)

            # Split the arrays that have lengths of the new (potentially
            # interpolated) time axis
            model_phys = split([model_phys, ], nints_interp, channel)[0]
        elif meta.multwhite:
            # Split the arrays that have lengths of the original time axis
            time, flux, unc, model, model_sys = \
                split([lc.time, flux, unc, model, model_sys],
                      meta.nints, channel)

            # Split the arrays that have lengths of the new (potentially
            # interpolated) time axis
            model_phys, new_timet = split([model_phys, new_time],
                                          nints_interp, channel)
        else:
            time = lc.time
            new_timet = new_time

        residuals = flux - model

        # Get binned data and times
        if not hasattr(meta, 'nbin_plot') or meta.nbin_plot is None or \
           meta.nbin_plot > len(time):
            nbin_plot = len(time)
        else:
            nbin_plot = meta.nbin_plot
        binned_time = util.binData(time, nbin_plot)
        binned_flux = util.binData(flux, nbin_plot)
        binned_unc = util.binData(unc, nbin_plot, err=True)
        binned_normflux = util.binData(flux/model_sys, nbin_plot)
        binned_res = util.binData(residuals, nbin_plot)

        fig = plt.figure(5101, figsize=(8, 6))
        plt.clf()

        ax = fig.subplots(3, 1)
        ax[0].errorbar(binned_time, binned_flux, yerr=binned_unc, fmt='.',
                       color='w', ecolor=color, mec=color)
        ax[0].plot(time, model, '.', ls='', ms=1, color='0.3', zorder=10)
        if isTitle:
            ax[0].set_title(f'{meta.eventlabel} - Channel {channel} - '
                            f'{fitter}')
        ax[0].set_ylabel('Normalized Flux', size=14)
        ax[0].set_xticks([])

        ax[1].errorbar(binned_time, binned_normflux, yerr=binned_unc, fmt='.',
                       color='w', ecolor=color, mec=color)
        ax[1].plot(new_timet, model_phys, color='0.3', zorder=10)
        ax[1].set_ylabel('Calibrated Flux', size=14)
        ax[1].set_xticks([])

        ax[2].errorbar(binned_time, binned_res*1e6, yerr=binned_unc*1e6,
                       fmt='.', color='w', ecolor=color, mec=color)
        ax[2].plot(time, np.zeros_like(time), color='0.3', zorder=10)
        ax[2].set_ylabel('Residuals (ppm)', size=14)
        ax[2].set_xlabel(str(lc.time_units), size=14)

        fig.subplots_adjust(hspace=0)
        fig.align_ylabels(ax)

        if lc.white:
            fname_tag = 'white'
        else:
            ch_number = str(channel).zfill(len(str(lc.nchannel)))
            fname_tag = f'ch{ch_number}'
        fname = (f'figs{os.sep}fig5101_{fname_tag}_lc_{fitter}'
                 + plots.figure_filetype)
        fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
        if not meta.hide_plots:
            plt.pause(0.2)


def plot_phase_variations(lc, model, meta, fitter, isTitle=True):
    """Plot the fitted model over the data. (Figs 5104 and Figs 5304)

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The fitted composite model.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    fitter : str
        The name of the fitter (for plot filename).
    isTitle : bool; optional
        Should figure have a title. Defaults to True.

    Notes
    -----
    History:

    - September 12, 2022 Taylor Bell
        Initial version.
    """
    if type(fitter) != str:
        raise ValueError(f'Expected type str for fitter, instead received a '
                         f'{type(fitter)}')

    model_sys_full = model.syseval()
    model_phys_full, new_time, nints_interp = \
        model.physeval(interp=meta.interp)

    for i, channel in enumerate(lc.fitted_channels):
        flux = np.ma.copy(lc.flux)
        unc = np.ma.copy(lc.unc_fit)
        model_sys = model_sys_full
        model_phys = model_phys_full
        flux /= model_sys
        color = lc.colors[i]

        if lc.share and not meta.multwhite:
            time = lc.time
            new_timet = new_time

            # Split the arrays that have lengths of the original time axis
            flux, unc, model, model_sys = split([flux, unc, model, model_sys],
                                                meta.nints, channel)

            # Split the arrays that have lengths of the new (potentially
            # interpolated) time axis
            model_phys = split([model_phys, ],
                               nints_interp, channel)[0]
        elif meta.multwhite:
            # Split the arrays that have lengths of the original time axis
            time, flux, unc, model, model_sys = \
                split([lc.time, flux, unc, model, model_sys],
                      meta.nints, channel)

            # Split the arrays that have lengths of the new (potentially
            # interpolated) time axis
            model_phys, new_timet = split([model_phys, new_time],
                                          nints_interp, channel)
        else:
            time = lc.time
            new_timet = new_time

        # Normalize to zero flux at eclipse
        flux -= 1
        model_phys -= 1

        # Convert to ppm
        model_phys *= 1e6
        flux *= 1e6
        unc *= 1e6

        # Get binned data and times
        if not hasattr(meta, 'nbin_plot') or meta.nbin_plot is None:
            nbin_plot = 100
        elif meta.nbin_plot > len(time):
            nbin_plot = len(time)
        else:
            nbin_plot = meta.nbin_plot
        binned_time = util.binData(time, nbin_plot)
        binned_flux = util.binData(flux, nbin_plot)
        binned_unc = util.binData(unc, nbin_plot, err=True)

        # Setup the figure
        fig = plt.figure(5104, figsize=(8, 6))
        plt.clf()
        ax = fig.gca()
        if isTitle:
            ax.set_title(f'{meta.eventlabel} - Channel {channel} - '
                         f'{fitter}')
        ax.set_ylabel('Normalized Flux - 1 (ppm)', size=14)
        ax.set_xlabel(str(lc.time_units), size=14)
        fig.patch.set_facecolor('white')

        # Plot the binned observations
        ax.errorbar(binned_time, binned_flux, yerr=binned_unc, fmt='.',
                    color='w', ecolor=color, mec=color)
        # Plot the model
        ax.plot(new_timet, model_phys, '.', ls='', ms=2, color='0.3',
                zorder=10)

        # Set nice axis limits
        sigma = np.ma.mean(binned_unc)
        max_astro = np.ma.max((model_phys-1))
        ax.set_ylim(-4*sigma, max_astro+6*sigma)
        ax.set_xlim(np.min(time), np.max(time))

        # Save/show the figure
        if lc.white:
            fname_tag = 'white'
        else:
            ch_number = str(channel).zfill(len(str(lc.nchannel)))
            fname_tag = f'ch{ch_number}'
        fname = (f'figs{os.sep}fig5104_{fname_tag}_phaseVariations_{fitter}'
                 + plots.figure_filetype)
        fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
        if not meta.hide_plots:
            plt.pause(0.2)

        if meta.isplots_S5 >= 3:
            # Setup the figure
            fig = plt.figure(5304, figsize=(8, 6))
            plt.clf()
            ax = fig.gca()
            if isTitle:
                ax.set_title(f'{meta.eventlabel} - Channel {channel} - '
                             f'{fitter}')
            ax.set_ylabel('Normalized Flux - 1 (ppm)', size=14)
            ax.set_xlabel(str(lc.time_units), size=14)
            fig.patch.set_facecolor('white')

            # Plot the unbinned data without errorbars
            ax.plot(time, flux, '.', c='k', zorder=0, alpha=0.01)
            # Plot the binned data with errorbars
            ax.errorbar(binned_time, binned_flux, yerr=binned_unc, fmt='.',
                        color=color, zorder=1)
            # Plot the physical model
            ax.plot(new_time, model_phys, '.', ls='', ms=2, color='0.3',
                    zorder=10)

            # Set nice axis limits
            sigma = np.ma.std(flux-model_phys)
            max_astro = np.ma.max(model_phys)
            ax.set_ylim(-3*sigma, max_astro+3*sigma)
            ax.set_xlim(np.min(time), np.max(time))
            # Save/show the figure
            if lc.white:
                fname_tag = 'white'
            else:
                ch_number = str(channel).zfill(len(str(lc.nchannel)))
                fname_tag = f'ch{ch_number}'
            fname = (f'figs{os.sep}fig5304_{fname_tag}_phaseVariations'
                     f'_{fitter}' + plots.figure_filetype)
            fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
            if not meta.hide_plots:
                plt.pause(0.2)


def plot_rms(lc, model, meta, fitter):
    """Plot an Allan plot to look for red noise. (Figs 5301)

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The fitted composite model.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    fitter : str
        The name of the fitter (for plot filename).

    Notes
    -----
    History:

    - December 29, 2021 Taylor Bell
        Moved plotting code to a separate function.
    - January 7-22, 2022 Megan Mansfield
        Adding ability to do a single shared fit across all channels
    """
    if type(fitter) != str:
        raise ValueError(f'Expected type str for fitter, instead received a '
                         f'{type(fitter)}')

    model_lc = model.eval()

    for channel in lc.fitted_channels:
        flux = np.ma.copy(lc.flux)
        model = np.ma.copy(model_lc)

        if lc.share and not meta.multwhite:
            time = lc.time

            # Split the arrays that have lengths of the original time axis
            flux, model = split([flux, model], meta.nints, channel)
        elif meta.multwhite:
            # Split the arrays that have lengths of the original time axis
            time, flux, model = split([lc.time, flux, model],
                                      meta.nints, channel)
        else:
            time = lc.time

        residuals = flux - model
        residuals = residuals[np.argsort(time)]

        rms, stderr, binsz = computeRMS(residuals, binstep=1)
        normfactor = 1e-6
        fig = plt.figure(
            int('52{}'.format(str(0).zfill(len(str(lc.nchannel))))),
            figsize=(8, 6))
        fig.clf()
        ax = fig.gca()
        ax.set_title(' Correlated Noise', size=16, pad=20)
        # our noise
        ax.loglog(binsz, rms / normfactor, color='black', lw=1.5,
                  label='Fit RMS', zorder=3)
        # expected noise
        ax.loglog(binsz, stderr / normfactor, color='red', ls='-', lw=2,
                  label=r'Std. Err. ($1/\sqrt{N}$)', zorder=1)

        # Format the main axes
        ax.set_xlim(0.95, binsz[-1] * 2)
        ax.set_ylim(stderr[-1] / normfactor / 2., stderr[0] / normfactor * 2.)
        ax.set_xlabel("Bin Size (N frames)", fontsize=14)
        ax.set_ylabel("RMS (ppm)", fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend(loc=1)

        # Add second x-axis using time instead of N-binned
        dt = (time[1]-time[0])*24*3600

        def t_N(N):
            return N*dt

        def N_t(t):
            return t/dt

        ax2 = ax.secondary_xaxis('top', functions=(t_N, N_t))
        ax2.set_xlabel('Bin Size (seconds)', fontsize=14)
        ax2.tick_params(axis='both', labelsize=12)

        if lc.white:
            fname_tag = 'white'
        else:
            ch_number = str(channel).zfill(len(str(lc.nchannel)))
            fname_tag = f'ch{ch_number}'
        fname = (f'figs{os.sep}fig5301_{fname_tag}_allanplot_{fitter}'
                 + plots.figure_filetype)
        plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
        if not meta.hide_plots:
            plt.pause(0.2)


def plot_corner(samples, lc, meta, freenames, fitter):
    """Plot a corner plot. (Figs 5501)

    Parameters
    ----------
    samples : ndarray
        The samples produced by the sampling algorithm.
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    freenames : iterable
        The names of the fitted parameters.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    fitter : str
        The name of the fitter (for plot filename).

    Notes
    -----
    History:

    - December 29, 2021 Taylor Bell
        Moved plotting code to a separate function.
    """
    ndim = len(freenames)+1  # One extra for the 1D histogram
    fig = plt.figure(5501, figsize=(ndim*1.4, ndim*1.4))
    fig.clf()

    # Don't allow offsets or scientific notation in tick labels
    old_useOffset = rcParams['axes.formatter.useoffset']
    old_xtick_labelsize = rcParams['xtick.labelsize']
    old_ytick_labelsize = rcParams['ytick.labelsize']
    rcParams['axes.formatter.useoffset'] = False
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    fig = corner.corner(samples, fig=fig, quantiles=[0.16, 0.5, 0.84],
                        max_n_ticks=3, labels=freenames, show_titles=True,
                        title_fmt='.3', title_kwargs={"fontsize": 10},
                        label_kwargs={"fontsize": 10}, fontsize=10,
                        labelpad=0.25)

    if lc.white:
        fname_tag = 'white'
    else:
        ch_number = str(lc.channel).zfill(len(str(lc.nchannel)))
        fname_tag = f'ch{ch_number}'
    fname = (f'figs{os.sep}fig5501_{fname_tag}_corner_{fitter}'
             + plots.figure_filetype)
    fig.savefig(meta.outputdir+fname, bbox_inches='tight', pad_inches=0.05,
                dpi=300)
    if not meta.hide_plots:
        plt.pause(0.2)

    rcParams['axes.formatter.useoffset'] = old_useOffset
    rcParams['xtick.labelsize'] = old_xtick_labelsize
    rcParams['ytick.labelsize'] = old_ytick_labelsize


def plot_chain(samples, lc, meta, freenames, fitter='emcee', burnin=False,
               nburn=0, nrows=3, ncols=4, nthin=1):
    """Plot the evolution of the chain to look for temporal trends. (Figs 5303)

    Parameters
    ----------
    samples : ndarray
        The samples produced by the sampling algorithm.
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    freenames : iterable
        The names of the fitted parameters.
    fitter : str; optional
        The name of the fitter (for plot filename). Defaults to 'emcee'.
    burnin : bool; optional
        Whether or not the samples include the burnin phase. Defaults to False.
    nburn : int; optional
        The number of burn-in steps that are discarded later. Defaults to 0.
    nrows : int; optional
        The number of rows to make per figure. Defaults to 3.
    ncols : int; optional
        The number of columns to make per figure. Defaults to 4.
    nthin : int; optional
        If >1, the plot will use every nthin point to help speed up
        computation and reduce clutter on the plot. Defaults to 1.

    Notes
    -----
    History:

    - December 29, 2021 Taylor Bell
        Moved plotting code to a separate function.
    """
    nsubplots = nrows*ncols
    nplots = int(np.ceil(len(freenames)/nsubplots))

    k = 0
    for plot_number in range(nplots):
        fig = plt.figure(5303, figsize=(6*ncols, 4*nrows))
        fig.clf()
        axes = fig.subplots(nrows, ncols, sharex=True)

        for j in range(ncols):
            for i in range(nrows):
                if k >= samples.shape[2]:
                    axes[i][j].set_axis_off()
                    continue
                vals = samples[::nthin, :, k]
                xvals = np.arange(samples.shape[0])[::nthin]
                n3sig, n2sig, n1sig, med, p1sig, p2sig, p3sig = \
                    np.percentile(vals, [0.15, 2.5, 16, 50, 84, 97.5, 99.85],
                                  axis=1)
                axes[i][j].fill_between(xvals, n3sig, p3sig, alpha=0.2,
                                        label=r'3$\sigma$')
                axes[i][j].fill_between(xvals, n2sig, p2sig, alpha=0.2,
                                        label=r'2$\sigma$')
                axes[i][j].fill_between(xvals, n1sig, p1sig, alpha=0.2,
                                        label=r'1$\sigma$')
                axes[i][j].plot(xvals, med, label='Median')
                axes[i][j].set_ylabel(freenames[k])
                axes[i][j].set_xlim(0, samples.shape[0]-1)
                for arr in [n3sig, n2sig, n1sig, med, p1sig, p2sig, p3sig]:
                    # Add some horizontal lines to make movement in walker
                    # population more obvious
                    axes[i][j].axhline(arr[0], ls='dotted', c='k', lw=1)
                if burnin and nburn > 0:
                    axes[i][j].axvline(nburn, ls='--', c='k',
                                       label='End of Burn-In')
                add_legend = ((j == (ncols-1) and i == (nrows//2)) or
                              (k == samples.shape[2]-1))
                if add_legend:
                    axes[i][j].legend(loc=6, bbox_to_anchor=(1.01, 0.5))
                k += 1
        fig.tight_layout(h_pad=0.0)

        if lc.white:
            fname_tag = 'white'
        else:
            ch_number = str(lc.channel).zfill(len(str(lc.nchannel)))
            fname_tag = f'ch{ch_number}'
        fname = f'figs{os.sep}fig5303_{fname_tag}'
        if burnin:
            fname += '_burninchain'
        else:
            fname += '_chain'
        fname += '_'+fitter
        if nplots > 1:
            fname += f'_plot{plot_number+1}of{nplots}'
        fname += plots.figure_filetype
        fig.savefig(meta.outputdir+fname, bbox_inches='tight',
                    pad_inches=0.05, dpi=300)
        if not meta.hide_plots:
            plt.pause(0.2)


def plot_trace(trace, model, lc, freenames, meta, fitter='nuts', compact=False,
               **kwargs):
    """Plot the evolution of the trace to look for temporal trends. (Figs 5305)

    Parameters
    ----------
    trace : pymc3.backends.base.MultiTrace or arviz.InferenceData
        A ``MultiTrace`` or ArviZ ``InferenceData`` object that contains the
        samples.
    model :

    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    freenames : iterable
        The names of the fitted parameters.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    fitter : str; optional
        The name of the fitter (for plot filename). Defaults to 'nuts'.
    compact: bool; optional
        Plot multidimensional variables in a single plot. Defailts to False.
    **kwargs : dict
        Additional keyword arguments to pass to pm.traceplot.

    Notes
    -----
    History:

    - November 22, 2022 Taylor Bell
        Initial version.
    """

    max_subplots = az_rcParams['plot.max_subplots'] // 2
    nplots = int(np.ceil(len(freenames)/max_subplots))
    npanels = min([len(freenames), max_subplots])

    for i in range(nplots):
        with model.model:
            ax = az.plot_trace(trace,
                               var_names=freenames[i*npanels:(i+1)*npanels],
                               compact=compact, show=False, **kwargs)
        fig = ax[0][0].figure

        if lc.white:
            fname_tag = 'white'
        else:
            ch_number = str(lc.channel).zfill(len(str(lc.nchannel)))
            fname_tag = f'ch{ch_number}'
        fname = f'figs{os.sep}fig5305_{fname_tag}_trace'
        fname += '_'+fitter
        fname += f'figure{i+1}of{nplots}'
        fname += plots.figure_filetype
        fig.savefig(meta.outputdir+fname, bbox_inches='tight',
                    pad_inches=0.05, dpi=300)
        if not meta.hide_plots:
            plt.pause(0.2)
        else:
            plt.close(fig)


def plot_res_distr(lc, model, meta, fitter):
    """Plot the normalized distribution of residuals + a Gaussian. (Fig 5302)

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The fitted composite model.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    fitter : str
        The name of the fitter (for plot filename).

    Notes
    -----
    History:

    - February 18, 2022 Caroline Piaulet
        Created function
    """
    if type(fitter) != str:
        raise ValueError(f'Expected type str for fitter, instead received a '
                         f'{type(fitter)}')

    model_lc = model.eval()

    for channel in lc.fitted_channels:
        plt.figure(5302, figsize=(8, 6))
        plt.clf()

        flux = np.ma.copy(lc.flux)
        unc = np.ma.copy(np.array(lc.unc_fit))
        model = np.ma.copy(model_lc)

        if lc.share or meta.multwhite:
            # Split the arrays that have lengths of the original time axis
            flux, unc, model = split([flux, unc, model],
                                     meta.nints, channel)

        residuals = flux - model
        hist_vals = residuals/unc
        hist_vals[~np.isfinite(hist_vals)] = np.nan  # Mask out any infinities

        n, bins, patches = plt.hist(hist_vals, alpha=0.5, color='b',
                                    edgecolor='b', lw=1)
        x = np.linspace(-4., 4., 200)
        px = stats.norm.pdf(x, loc=0, scale=1)
        plt.plot(x, px*(bins[1]-bins[0])*len(residuals), 'k-', lw=2)
        plt.xlabel("Residuals/Uncertainty", fontsize=14)
        if lc.white:
            fname_tag = 'white'
        else:
            ch_number = str(channel).zfill(len(str(lc.nchannel)))
            fname_tag = f'ch{ch_number}'
        fname = (f'figs{os.sep}fig5302_{fname_tag}_res_distri_{fitter}'
                 + plots.figure_filetype)
        plt.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
        if not meta.hide_plots:
            plt.pause(0.2)


def plot_GP_components(lc, model, meta, fitter, isTitle=True):
    """Plot the lightcurve + GP model + residuals (Figs 5102)

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The fitted composite model.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    fitter : str
        The name of the fitter (for plot filename).
    isTitle : bool; optional
        Should figure have a title. Defaults to True.

    Notes
    -----
    History:

    - February 28, 2022 Eva-Maria Ahrer
        Written function
    - March 9, 2022 Eva-Maria Ahrer
        Adapted with shared parameters
    """
    if type(fitter) != str:
        raise ValueError(f'Expected type str for fitter, instead received a '
                         f'{type(fitter)}')

    model_with_GP = model.eval(incl_GP=True)
    model_sys_full = model.syseval()
    model_lc = model.eval()
    model_GP = model.GPeval(model_lc)

    for i, channel in enumerate(lc.fitted_channels):
        flux = np.ma.copy(lc.flux)
        unc = np.ma.copy(lc.unc_fit)
        model = np.ma.copy(model_with_GP)
        model_sys = model_sys_full
        model_GP_component = model_GP
        color = lc.colors[i]

        if lc.share and not meta.multwhite:
            time = lc.time
            # Split the arrays that have lengths of the original time axis
            flux, unc, model, model_sys, model_GP_component = \
                split([flux, unc, model, model_sys, model_GP_component],
                      meta.nints, channel)
        elif meta.multwhite:
            # Split the arrays that have lengths of the original time axis
            time, flux, unc, model, model_sys, model_GP_component = \
                split([lc.time, flux, unc, model, model_sys,
                       model_GP_component], meta.nints, channel)

        residuals = flux - model
        fig = plt.figure(5102, figsize=(8, 6))
        plt.clf()
        ax = fig.subplots(3, 1)
        ax[0].errorbar(time, flux, yerr=unc, fmt='.', color='w',
                       ecolor=color, mec=color)
        ax[0].plot(time, model, '.', ls='', ms=2, color='0.3',
                   zorder=10)
        if isTitle:
            ax[0].set_title(f'{meta.eventlabel} - Channel {channel} - '
                            f'{fitter}')
        ax[0].set_ylabel('Normalized Flux', size=14)
        ax[1].plot(time, model_GP_component, '.', color=color)
        ax[1].set_ylabel('GP component', size=14)
        ax[1].set_xlabel(str(lc.time_units), size=14)
        ax[2].errorbar(time, residuals*1e6, yerr=unc*1e6, fmt='.',
                       color='w', ecolor=color, mec=color)
        ax[2].plot(time, np.zeros_like(time), color='0.3', zorder=10)
        ax[2].set_ylabel('Residuals (ppm)', size=14)
        ax[2].set_xlabel(str(lc.time_units), size=14)

        if lc.white:
            fname_tag = 'white'
        else:
            ch_number = str(channel).zfill(len(str(lc.nchannel)))
            fname_tag = f'ch{ch_number}'
        fname = (f'figs{os.sep}fig5102_{fname_tag}_lc_GP_{fitter}'
                 + plots.figure_filetype)
        fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
        if not meta.hide_plots:
            plt.pause(0.2)
