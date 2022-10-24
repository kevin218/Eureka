import numpy as np
import copy
import pymc3_ext as pmx
from astropy import table

from .likelihood import computeRedChiSq
from . import plots_s5 as plots
from .fitters import group_variables, load_old_fitparams, save_fit


def exoplanetfitter(lc, model, meta, log, calling_function='exoplanet',
                    **kwargs):
    """Perform sampling using exoplanet.

    Parameters
    ----------
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log: logedit.Logedit
        The open log in which notes from this step can be added.
    calling_function: str, optional
        The fitter that is being run (e.g. may be 'emcee' if running lsqfitter
        to initialize emcee walkers). Defailts to 'exoplanet'.
    **kwargs:
        Arbitrary keyword arguments.

    Returns
    -------
    best_model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model after fitting

    Notes
    -----
    History:

    - April 6, 2022 Taylor Bell
        Initial version.
    """
    # Group the different variable types
    freenames, freepars, prior1, prior2, priortype, indep_vars = \
        group_variables(model)
    if hasattr(meta, 'old_fitparams') and meta.old_fitparams is not None:
        freepars = load_old_fitparams(meta, log, lc.channel, freenames)
    
    model.setup(lc.time, lc.flux, lc.unc)

    start = {}
    for name, val in zip(freenames, freepars):
        start[name] = val

    log.writelog('Running exoplanet optimizer...')
    with model.model:
        map_soln = pmx.optimize(start=start)

    # Get the best fit params
    fit_params = np.array([map_soln[name] for name in freenames])
    model.update(fit_params)

    if "scatter_ppm" in freenames:
        ind = [i for i in np.arange(len(freenames))
               if freenames[i][0:11] == "scatter_ppm"]
        for chan in range(len(ind)):
            lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = \
                fit_params[ind[chan]] * 1e-6
    elif "scatter_mult" in freenames:
        ind = [i for i in np.arange(len(freenames))
               if freenames[i][0:12] == "scatter_mult"]
        if not hasattr(lc, 'unc_fit'):
            lc.unc_fit = copy.deepcopy(lc.unc)
        for chan in range(len(ind)):
            lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = \
                (fit_params[ind[chan]] *
                 lc.unc[chan*lc.time.size:(chan+1)*lc.time.size])

    t_results = table.Table([freenames, fit_params],
                            names=("Parameter", "Mean"))

    # Save the fit ASAP
    save_fit(meta, lc, model, calling_function, t_results, freenames)

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, log, model, meta, freenames)

    log.writelog('\nEXOPLANET RESULTS:')
    for i in range(len(freenames)):
        if 'scatter_mult' in freenames[i]:
            chan = freenames[i].split('_')[-1]
            if chan.isnumeric():
                chan = int(chan)
            else:
                chan = 0
            scatter_ppm = (fit_params[i] *
                           np.ma.median(lc.unc[chan*lc.time.size:
                                               (chan+1)*lc.time.size]) * 1e6)
            log.writelog(f'{freenames[i]}: {fit_params[i]}; {scatter_ppm} ppm')
        else:
            log.writelog(f'{freenames[i]}: {fit_params[i]}')
    log.writelog('')

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter=calling_function)

    # Plot GP fit + components
    # if model.GP and meta.isplots_S5 >= 1:
    #     plots.plot_GP_components(lc, model, meta, fitter=calling_function)

    # Zoom in on phase variations
    # if meta.isplots_S5 >= 1 and 'sinusoid_pc' in meta.run_myfuncs:
    #     plots.plot_phase_variations(lc, model, meta, fitter=calling_function)

    # Plot Allan plot
    if meta.isplots_S5 >= 3 and calling_function == 'exoplanet':
        # This plot is only really useful if you're actually using the
        # exoplanet fitter, otherwise don't make it
        plots.plot_rms(lc, model, meta, fitter=calling_function)

    # Plot residuals distribution
    if meta.isplots_S5 >= 3 and calling_function == 'exoplanet':
        plots.plot_res_distr(lc, model, meta, fitter=calling_function)

    # Make a new model instance
    model.chi2red = chi2red
    model.fit_params = fit_params

    return model


def nutsfitter(lc, model, meta, log, **kwargs):
    """Perform sampling using PyMC3 NUTS sampler.

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    meta : eureka.lib.readECF.MetaClass
        The metadata object
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    **kwargs : dict
        Arbitrary keyword arguments.

    Returns
    -------
    best_model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model after fitting

    Notes
    -----
    History:

    - October 5, 2022 Taylor Bell
        Initial version.
    """
    # Group the different variable types
    freenames, freepars, prior1, prior2, priortype, indep_vars = \
        group_variables(model)
    if hasattr(meta, 'old_fitparams') and meta.old_fitparams is not None:
        freepars = load_old_fitparams(meta, log, lc.channel, freenames)
    ndim = len(freenames)

    model.setup(lc.time, lc.flux, lc.unc)

    start = {}
    for name, val in zip(freenames, freepars):
        start[name] = val

    log.writelog('Running PyMC3 NUTS sampler...')
    with model.model:
        trace = pmx.sample(tune=meta.tune, draws=meta.draws, start=start,
                           target_accept=meta.target_accept,
                           chains=meta.chains, cores=meta.ncpu)
        print()

    samples = np.hstack([trace[name].reshape(-1, 1) for name in freenames])

    # Record median + percentiles
    q = np.percentile(samples, [16, 50, 84], axis=0)
    fit_params = q[1]  # median
    mean_params = np.mean(samples, axis=0)
    errs = np.std(samples, axis=0)

    # Create table of results
    t_results = table.Table([freenames, mean_params, q[0]-q[1], q[2]-q[1],
                             q[0], fit_params, q[2]],
                            names=("Parameter", "Mean", "-1sigma", "+1sigma",
                                   "16th", "50th", "84th"))

    upper_errs = q[2]-q[1]
    lower_errs = q[1]-q[0]

    model.update(fit_params)
    model.errs = dict(zip(freenames, errs))
    if "scatter_ppm" in freenames:
        ind = [i for i in np.arange(len(freenames))
               if freenames[i][0:11] == "scatter_ppm"]
        for chan in range(len(ind)):
            lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = \
                fit_params[ind[chan]] * 1e-6
    elif "scatter_mult" in freenames:
        ind = [i for i in np.arange(len(freenames))
               if freenames[i][0:12] == "scatter_mult"]
        for chan in range(len(ind)):
            lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = \
                fit_params[ind[chan]] * lc.unc[chan*lc.time.size:
                                               (chan+1)*lc.time.size]
    else:
        lc.unc_fit = lc.unc

    # Save the fit ASAP so plotting errors don't make you lose everything
    save_fit(meta, lc, model, 'nuts', t_results, freenames, samples)

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, log, model, meta, freenames)

    log.writelog('PYMC3 NUTS RESULTS:')
    for i in range(ndim):
        if 'scatter_mult' in freenames[i]:
            chan = freenames[i].split('_')[-1]
            if chan.isnumeric():
                chan = int(chan)
            else:
                chan = 0
            scatter_ppm = (1e6 * fit_params[i] *
                           np.ma.median(lc.unc[chan*lc.time.size:
                                               (chan+1)*lc.time.size]))
            scatter_ppm_upper = (1e6 * upper_errs[i] *
                                 np.ma.median(lc.unc[chan*lc.time.size:
                                                     (chan+1)*lc.time.size]))
            scatter_ppm_lower = (1e6 * lower_errs[i] *
                                 np.ma.median(lc.unc[chan*lc.time.size:
                                                     (chan+1)*lc.time.size]))
            log.writelog(f'{freenames[i]}: {fit_params[i]} (+{upper_errs[i]},'
                         f' -{lower_errs[i]}); {scatter_ppm} '
                         f'(+{scatter_ppm_upper}, -{scatter_ppm_lower}) ppm')
        else:
            log.writelog(f'{freenames[i]}: {fit_params[i]} (+{upper_errs[i]},'
                         f' -{lower_errs[i]})')
    log.writelog('')

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter='nuts')

    # Plot GP fit + components
    # if model.GP and meta.isplots_S5 >= 1:
    #     plots.plot_GP_components(lc, model, meta, fitter='nuts')

    # Zoom in on phase variations
    # if meta.isplots_S5 >= 1 and 'sinusoid_pc' in meta.run_myfuncs:
    #     plots.plot_phase_variations(lc, model, meta, fitter='nuts')

    # Plot Allan plot
    if meta.isplots_S5 >= 3:
        plots.plot_rms(lc, model, meta, fitter='nuts')

    # Plot residuals distribution
    if meta.isplots_S5 >= 3:
        plots.plot_res_distr(lc, model, meta, fitter='nuts')

    if meta.isplots_S5 >= 5:
        plots.plot_corner(samples, lc, meta, freenames, fitter='nuts')

    # Make a new model instance
    model.chi2red = chi2red
    model.fit_params = fit_params

    return model
