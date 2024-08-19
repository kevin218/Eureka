import numpy as np
import pandas as pd
import copy
from io import StringIO
import os
import sys
import h5py
import xarray as xr
from astraeus import xarrayIO as xrio
import time as time_pkg

from scipy.optimize import minimize
import lmfit
import emcee

from dynesty import NestedSampler
from dynesty.utils import resample_equal

from .likelihood import (computeRedChiSq, lnprob, ln_like, ptform,
                         update_uncertainty)
from . import plots_s5 as plots
from ..lib import astropytable
from ..lib.split_channels import get_trim

from multiprocessing import Pool

from astropy import table


def lsqfitter(lc, model, meta, log, calling_function='lsq', **kwargs):
    """Perform least-squares fit.

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    calling_function : str
        The fitter that is being run (e.g. may be 'emcee' if running lsqfitter
        to initialize emcee walkers). Defailts to 'lsq'.
    **kwargs : dict
        Arbitrary keyword arguments.

    Returns
    -------
    best_model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model after fitting

    Notes
    -----
    History:

    - December 29-30, 2021 Taylor Bell
        Updated documentation and arguments. Reduced repeated code.
        Also saving covariance matrix for later estimation of sampler
        step size.
    - January 7-22, 2022 Megan Mansfield
        Adding ability to do a single shared fit across all channels
    - February 28-March 1, 2022 Caroline Piaulet
        Adding scatter_ppm parameter
    - Mar 13-Apr 18, 2022 Caroline Piaulet
        Record an astropy table for param values
    """
    # Group the different variable types
    freenames = lc.freenames
    freepars, prior1, prior2, priortype, indep_vars = \
        group_variables(model)
    if meta.old_fitparams is not None:
        freepars = load_old_fitparams(lc, meta, log, freenames,
                                      calling_function)

    start_lnprob = lnprob(freepars, lc, model, prior1, prior2, priortype,
                          freenames)
    log.writelog(f'Starting lnprob: {start_lnprob}', mute=(not meta.verbose))

    # Plot starting point
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta,
                       fitter=calling_function+'StartingPoint')
        # Plot GP starting point
        if model.GP:
            plots.plot_GP_components(lc, model, meta,
                                     fitter=calling_function+'StartingPoint')

    def neg_lnprob(theta, lc, model, prior1, prior2, priortype, freenames):
        return -lnprob(theta, lc, model, prior1, prior2, priortype, freenames)
    global lsq_t0
    lsq_t0 = time_pkg.time()

    def callback_full(theta, lc, model, prior1, prior2, priortype, freenames):
        global lsq_t0
        if (time_pkg.time()-lsq_t0) > 0.5:
            lsq_t0 = time_pkg.time()
            print('Current lnprob = ', lnprob(theta, lc, model, prior1, prior2,
                                              priortype, freenames), end='\r')

    def callback(theta):
        return callback_full(theta, lc, model, prior1, prior2, priortype,
                             freenames)

    results = minimize(neg_lnprob, freepars,
                       args=(lc, model, prior1, prior2, priortype, freenames),
                       method=meta.lsq_method, tol=meta.lsq_tol,
                       options={'maxiter': meta.lsq_maxiter},
                       callback=callback)

    log.writelog("\nVerbose lsq results: {}\n".format(results),
                 mute=(not meta.verbose))
    if not meta.verbose:
        log.writelog("Success?: {}".format(results.success))
        log.writelog(results.message)

    # Get the best fit params
    fit_params = results.x

    # Create table of results
    t_results = table.Table([freenames, fit_params],
                            names=("Parameter", "Mean"))

    model.update(fit_params)
    lc.unc_fit = update_uncertainty(fit_params, lc.nints, lc.unc, freenames,
                                    lc.nchannel_fitted)

    # Save the fit ASAP
    save_fit(meta, lc, model, calling_function, t_results, freenames)

    end_lnprob = lnprob(fit_params, lc, model, prior1, prior2, priortype,
                        freenames)
    log.writelog(f'Ending lnprob: {end_lnprob}', mute=(not meta.verbose))

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, log, model, meta, freenames)

    log.writelog('\nLSQ RESULTS:')
    for i in range(len(freenames)):
        if 'scatter_mult' in freenames[i]:
            chan = freenames[i].split('_')[-1]
            if chan.isnumeric():
                chan = int(chan)
            else:
                chan = 0
            trim1, trim2 = get_trim(meta.nints, chan)
            unc = np.ma.median(lc.unc[trim1:trim2])
            scatter_ppm = 1e6*fit_params[i]*unc
            log.writelog(f'{freenames[i]}: {fit_params[i]}; {scatter_ppm} ppm')
        else:
            log.writelog(f'{freenames[i]}: {fit_params[i]}')
    log.writelog('')

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter=calling_function)

    # Plot GP fit + components
    if model.GP and meta.isplots_S5 >= 1:
        plots.plot_GP_components(lc, model, meta, fitter=calling_function)

    # Zoom in on phase variations
    if meta.isplots_S5 >= 1 and ('sinusoid_pc' in meta.run_myfuncs
                                 or 'poet_pc' in meta.run_myfuncs
                                 or 'quasilambert_pc' in meta.run_myfuncs):
        plots.plot_phase_variations(lc, model, meta, fitter=calling_function)

    # Plot Allan plot
    if meta.isplots_S5 >= 3 and calling_function == 'lsq' and \
            np.size(lc.flux) > 20:
        # This plot is only really useful if you're actually using the
        # lsq fitter, otherwise don't make it
        # Also, mc3.stats.time_avg breaks when testing with a small
        # number of integrations
        plots.plot_rms(lc, model, meta, fitter=calling_function)

    # Plot residuals distribution
    if meta.isplots_S5 >= 3 and calling_function == 'lsq':
        plots.plot_res_distr(lc, model, meta, fitter=calling_function)

    # Make a new model instance
    best_model = copy.deepcopy(model)
    best_model.components[0].update(fit_params)

    # Save the covariance matrix in case it's needed to estimate step size
    # for a sampler
    # FINDME:
    # Commented out for now because op.least_squares() doesn't provide
    # covariance matrix
    # Need to compute using Jacobian matrix instead (hess_inv = (J.T J)^{-1})
    # model_lc = model.eval()
    # if results[1] is not None:
    #     residuals = (lc.flux - model_lc)
    #     cov_mat = results[1]*np.var(residuals)
    # else:
    #     # Sometimes lsq will fail to converge and will return a None
    #     # covariance matrix
    #     cov_mat = None
    cov_mat = None
    best_model.__setattr__('cov_mat', cov_mat)
    best_model.__setattr__('chi2red', chi2red)
    best_model.__setattr__('fit_params', fit_params)

    return best_model


def demcfitter(lc, model, meta, log, **kwargs):
    """Perform sampling using Differential Evolution Markov Chain.

    This is an empty placeholder function to be filled later.

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
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

    - December 29, 2021 Taylor Bell
        Updated documentation and arguments
    """
    best_model = None
    return best_model


def emceefitter(lc, model, meta, log, **kwargs):
    """Perform sampling using emcee.

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

    - December 29, 2021 Taylor Bell
        Updated documentation. Reduced repeated code.
    - January 7-22, 2022 Megan Mansfield
        Adding ability to do a single shared fit across all channels
    - February 23-25, 2022 Megan Mansfield
        Added log-uniform and Gaussian priors.
    - February 28-March 1, 2022 Caroline Piaulet
        Adding scatter_ppm parameter. Added statements to avoid some initial
        state issues.
    - Mar 13-Apr 18, 2022 Caroline Piaulet
        Record an astropy table for mean, median, percentiles,
        +/- 1 sigma, all params
    """
    # Group the different variable types
    freenames = lc.freenames
    freepars, prior1, prior2, priortype, indep_vars = \
        group_variables(model)
    if meta.old_fitparams is not None:
        freepars = load_old_fitparams(lc, meta, log, freenames, 'emcee')
    ndim = len(freenames)

    if meta.old_chain is not None:
        pos, nwalkers = start_from_oldchain_emcee(lc, meta, log, ndim,
                                                  freenames, freepars,
                                                  prior1, prior2,
                                                  priortype)
    else:
        if meta.lsq_first:
            # Only call lsq fitter first if asked
            log.writelog('\nCalling lsqfitter first...')
            # RUN LEAST SQUARES
            lsq_sol = lsqfitter(lc, model, meta, log,
                                calling_function='emcee_lsq', **kwargs)

            freepars = lsq_sol.fit_params
        else:
            lsq_sol = None
        pos, nwalkers = initialize_emcee_walkers(meta, log, ndim, lsq_sol,
                                                 freepars, prior1, prior2,
                                                 priortype)

    start_lnprob = lnprob(np.median(pos, axis=0), lc, model, prior1, prior2,
                          priortype, freenames)
    log.writelog(f'Starting lnprob: {start_lnprob}', mute=(not meta.verbose))

    # Plot starting point
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta,
                       fitter='emceeStartingPoint')
        # Plot GP starting point
        if model.GP:
            plots.plot_GP_components(lc, model, meta,
                                     fitter='emceeStartingPoint')

    # Initialize tread pool
    if meta.ncpu > 1:
        pool = Pool(meta.ncpu)
    else:
        meta.ncpu = 1
        pool = None

    # Run emcee burn-in
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(lc, model, prior1, prior2,
                                          priortype, freenames),
                                    pool=pool)
    log.writelog('Running emcee burn-in and production steps...')
    sampler.run_mcmc(pos, meta.run_nsteps, progress=True)
    # log.writelog('Running emcee burn-in...')
    # state = sampler.run_mcmc(pos, meta.run_nsteps, progress=True)
    # # Log some details about the burn-in phase
    # acceptance_fraction = np.mean(sampler.acceptance_fraction)
    # log.writelog(f"Mean acceptance fraction: {acceptance_fraction:.3f}",
    #              mute=(not meta.verbose))
    # try:
    # autocorr = sampler.get_autocorr_time()
    #     log.writelog(f"Mean autocorrelation time: {autocorr:.3f} steps",
    #                  mute=(not meta.verbose))
    # except:
    #     log.writelog("Error: Unable to estimate the autocorrelation time!",
    #                  mute=(not meta.verbose))
    # mid_lnprob = lnprob(np.median(sampler.get_chain()[-1], axis=0), lc,
    #                     model, prior1, prior2, priortype, freenames)
    # log.writelog(f'Intermediate lnprob: {mid_lnprob}',
    #              mute=(not meta.verbose))
    # if meta.isplots_S5 >= 3:
    #     plots.plot_chain(sampler.get_chain(), lc, meta, freenames,
    #                      fitter='emcee', burnin=True)
    # # Reset the sampler and do the production run
    # log.writelog('Running emcee production run...')
    # sampler.reset()
    # sampler.run_mcmc(state, meta.run_nsteps-meta.run_nburn, progress=True)
    # samples = sampler.get_chain(flat=True)

    samples = sampler.get_chain(flat=True, discard=meta.run_nburn)
    if meta.ncpu > 1:
        # Close the thread pool
        pool.close()
        pool.join()

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
    lc.unc_fit = update_uncertainty(fit_params, lc.nints, lc.unc, freenames,
                                    lc.nchannel_fitted)

    # Save the fit ASAP so plotting errors don't make you lose everything
    save_fit(meta, lc, model, 'emcee', t_results, freenames, samples)

    end_lnprob = lnprob(fit_params, lc, model, prior1, prior2, priortype,
                        freenames)
    log.writelog(f'Ending lnprob: {end_lnprob}', mute=(not meta.verbose))
    acceptance_fraction = np.mean(sampler.acceptance_fraction)
    log.writelog(f"Mean acceptance fraction: {acceptance_fraction:.3f}",
                 mute=(not meta.verbose))
    try:
        autocorr = np.mean(sampler.get_autocorr_time())
        log.writelog(f"Mean autocorrelation time: {autocorr:.3f} steps",
                     mute=(not meta.verbose))
    except:
        # FINDME: Need to only catch the expected exception
        log.writelog("WARNING: Unable to estimate the autocorrelation time!",
                     mute=(not meta.verbose))

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, log, model, meta, freenames)

    log.writelog('\nEMCEE RESULTS:')
    for i in range(ndim):
        if 'scatter_mult' in freenames[i]:
            chan = freenames[i].split('_')[-1]
            if chan.isnumeric():
                chan = int(chan)
            else:
                chan = 0
            trim1, trim2 = get_trim(meta.nints, chan)
            unc = np.ma.median(lc.unc[trim1:trim2])
            scatter_ppm = 1e6*fit_params[i]*unc
            scatter_ppm_upper = 1e6*upper_errs[i]*unc
            scatter_ppm_lower = 1e6*lower_errs[i]*unc
            log.writelog(f'{freenames[i]}: {fit_params[i]} (+{upper_errs[i]},'
                         f' -{lower_errs[i]}); {scatter_ppm} '
                         f'(+{scatter_ppm_upper}, -{scatter_ppm_lower}) ppm')
        else:
            log.writelog(f'{freenames[i]}: {fit_params[i]} (+{upper_errs[i]},'
                         f' -{lower_errs[i]})')
    log.writelog('')

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter='emcee')

    # Plot GP fit + components
    if model.GP and meta.isplots_S5 >= 1:
        plots.plot_GP_components(lc, model, meta, fitter='emcee')

    # Zoom in on phase variations
    if meta.isplots_S5 >= 1 and ('sinusoid_pc' in meta.run_myfuncs
                                 or 'poet_pc' in meta.run_myfuncs
                                 or 'quasilambert_pc' in meta.run_myfuncs):
        plots.plot_phase_variations(lc, model, meta, fitter='emcee')

    # Plot Allan plot
    if meta.isplots_S5 >= 3 and np.size(lc.flux) > 20:
        # mc3.stats.time_avg breaks when testing with a small
        # number of integrations
        plots.plot_rms(lc, model, meta, fitter='emcee')

    # Plot residuals distribution
    if meta.isplots_S5 >= 3:
        plots.plot_res_distr(lc, model, meta, fitter='emcee')

    # Plot chain evolution
    if meta.isplots_S5 >= 3:
        plots.plot_chain(sampler.get_chain(), lc, meta, freenames,
                         fitter='emcee', burnin=True, nburn=meta.run_nburn)
        plots.plot_chain(sampler.get_chain(discard=meta.run_nburn), lc, meta,
                         freenames, fitter='emcee', burnin=False)

    if meta.isplots_S5 >= 5:
        plots.plot_corner(samples, lc, meta, freenames, fitter='emcee')

    # Make a new model instance
    best_model = copy.deepcopy(model)
    best_model.components[0].update(fit_params)
    best_model.__setattr__('chi2red', chi2red)
    best_model.__setattr__('fit_params', fit_params)

    return best_model


def start_from_oldchain_emcee(lc, meta, log, ndim, freenames, freepars,
                              prior1, prior2, priortype):
    """Restart emcee using the ending point of an old chain.

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    meta : eureka.lib.readECF.MetaClass
        The meta data object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    ndim : int
        The number of fitted parameters.
    freenames : list
        The names of the fitted parameters.
    freepars : list
        The starting values of the fitted parameters.
    prior1 : list
        The list of prior1 values.
    prior2 : list
        The list of prior2 values.
    priortype : list
        The types of each prior (to determine meaning of prior1 and prior2).

    Returns
    -------
    pos : ndarray
        The starting positions for all walkers.
    nwalkers : int
        The number of walkers (may differ from the requested number
        if unable to get all walkers within the priors).

    Raises
    ------
    AssertionError
        The old chain is not compatible with the current fit.
    AssertionError
        Unable to get enough walkers within the prior range.
    """
    if lc.white:
        channel_tag = '_white'
    elif lc.share:
        channel_tag = '_shared'
    else:
        ch_number = str(lc.channel).zfill(len(str(lc.nchannel)))
        channel_tag = f'_ch{ch_number}'

    foldername = os.path.join(meta.topdir, *meta.old_chain.split(os.sep))
    fname = f'S5_emcee_fitparams{channel_tag}.csv'
    fitted_values = pd.read_csv(os.path.join(foldername, fname),
                                escapechar='#', skipinitialspace=True)
    full_keys = np.array(fitted_values['Parameter'])

    fname = f'S5_emcee_samples{channel_tag}'
    # Load HDF5 files
    full_fname = os.path.join(foldername, fname)+'.h5'
    ds = xrio.readXR(full_fname, verbose=False)
    if ds is None:
        # Working with an old save file
        with h5py.File(full_fname, 'r') as hf:
            samples = hf['samples'][:]
    else:
        samples = ds.to_array().T.values
    log.writelog(f'Old chain path: {full_fname}')

    if not np.all([key in freenames for key in full_keys]):
        # There were extra free parameters before - just get the relevant ones
        relevant_inds = np.array([key in freenames for key in full_keys])
        removed_inds = full_keys[~relevant_inds]
        full_keys = full_keys[relevant_inds]
        samples = samples[:, relevant_inds]
        message = ('Old chain had extra fitted parameters. '
                   'Removing the previously fitted parameters:\n'
                   f'    {removed_inds}')
        log.writelog(message, mute=(not meta.verbose))

    # Initialize the walkers using samples from the old chain
    nwalkers = meta.run_nwalkers
    pos = samples[-nwalkers:]
    walkers_used = nwalkers

    # Make sure that no walkers are starting in the same place as
    # they would then exactly follow each other
    repeat_pos = np.where([np.any(np.all(pos[i] == np.delete(pos, i, axis=0),
                                         axis=1))
                           for i in range(pos.shape[0])])[0]
    while (len(repeat_pos) > 0 and
           samples.shape[0] > (walkers_used+len(repeat_pos))):
        pos[repeat_pos] = samples[:-walkers_used][-len(repeat_pos):]
        walkers_used += len(repeat_pos)
        repeat_pos = np.where([np.any(np.all(pos[i] ==
                                             np.delete(pos, i, axis=0),
                                             axis=1))
                               for i in range(pos.shape[0])])[0]

    # If unable to initialize all walkers in unique starting locations,
    # use fewer walkers unless there'd be fewer walkers than dimensions
    if len(repeat_pos) > 0 and (nwalkers-len(repeat_pos) > ndim):
        pos = np.delete(pos, repeat_pos, axis=0)
        nwalkers = pos.shape[0]
        log.writelog(f'Warning: Unable to initialize all walkers at different '
                     f'positions using old chain!\nUsing {nwalkers} walkers '
                     f'instead of the initially requested {meta.run_nwalkers} '
                     f'walkers')
    elif len(repeat_pos) > 0:
        message = (f'Error: Unable to initialize all walkers at different '
                   f'positions using old chain!\nUsing {nwalkers} walkers '
                   f'instead of the initially requested {meta.run_nwalkers} '
                   f'walkers is not permitted as there are {ndim} fitted '
                   f'parameters')
        log.writelog(message, mute=True)
        raise AssertionError(message)

    if not np.all([key in full_keys for key in freenames]):
        # There are now extra free parameters
        # Populate them using initialize_emcee_walkers
        missing_freenames = np.array([key for key in freenames
                                      if key not in full_keys])
        message = ('Old chain was missing some fitted parameters. '
                   'Adding the new fitted parameters:\n'
                   f'    {missing_freenames}')
        log.writelog(message, mute=(not meta.verbose))

        meta.run_nwalkers = nwalkers
        temp_pos, nwalkers = initialize_emcee_walkers(
            meta, log, ndim, None, freepars, prior1, prior2, priortype)

        new_pos = np.zeros((nwalkers, len(freenames)))
        for i, key in enumerate(freenames):
            if key not in full_keys:
                # There are now extra free parameters
                # Populate them using initialize_emcee_walkers
                new_pos[:, i] = temp_pos[:, i]
            else:
                # This variable already existed, so just add it
                new_pos[:, i] = pos[:, np.where(full_keys == key)[0][0]]

        pos = new_pos

    return pos, nwalkers


def initialize_emcee_walkers(meta, log, ndim, lsq_sol, freepars, prior1,
                             prior2, priortype):
    """Initialize emcee walker starting positions

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The meta data object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    ndim : int
        The number of fitted parameters.
    lsq_sol : The results from the lsqfitter.
        The results from the lsqfitter.
    freepars : list
        The initial values of the fitted parameters.
    prior1 : list
        The list of prior1 values.
    prior2 : list
        The list of prior2 values.
    priortype : list
        The types of each prior (to determine meaning of prior1 and prior2).

    Returns
    -------
    pos : ndarray
        The starting position of all walkers.
    nwalkers : int
        The number of walkers (may differ from the requested number
        if unable to get all walkers within the priors).

    Raises
    ------
    AssertionError
        Failed to initialize any walkers within the priors
    AssertionError
        Failed to initialize enough walkers within the priors
    """
    u = np.where(priortype == 'U')[0]
    lu = np.where(priortype == 'LU')[0]
    n = np.where(priortype == 'N')[0]
    if lsq_sol is not None and lsq_sol.cov_mat is not None:
        step_size = np.diag(lsq_sol.cov_mat)
        if np.any(step_size == 0.):
            ind_zero_u = np.where(step_size[u] == 0.)[0]
            ind_zero_lu = np.where(step_size[lu] == 0.)[0]
            ind_zero_n = np.where(step_size[n] == 0.)[0]
            step_size[u][ind_zero_u] = (0.001*(prior2[u][ind_zero_u] -
                                               prior1[u][ind_zero_u]))
            step_size[lu][ind_zero_lu] = (0.001 *
                                          (np.exp(prior2[lu][ind_zero_lu]) -
                                           np.exp(prior1[lu][ind_zero_lu])))
            step_size[n][ind_zero_n] = 0.1*prior2[n][ind_zero_n]
    else:
        # Sometimes the lsq fitter won't converge and will give None as
        # the covariance matrix. In that case, we need to establish the
        # step size in another way. Using a fractional step compared to
        # the prior range can work best for precisely known values like
        # t0 and period
        log.writelog('No covariance matrix from LSQ - falling back on a step '
                     'size based on the prior range', mute=(not meta.verbose))
        step_size = np.ones(ndim)
        step_size[u] = 0.001*(prior2[u] - prior1[u])
        step_size[lu] = 0.001*(np.exp(prior2[lu]) - np.exp(prior1[lu]))
        step_size[n] = 0.1*prior2[n]
    nwalkers = meta.run_nwalkers

    # make it robust to lsq hitting the upper or lower bound of the param space
    ind_max = np.where(freepars[u] - prior2[u] == 0.)[0]
    ind_min = np.where(freepars[u] - prior1[u] == 0.)[0]
    ind_max_LU = np.where(np.log(freepars[lu]) - prior2[lu] == 0.)[0]
    ind_min_LU = np.where(np.log(freepars[lu]) - prior1[lu] == 0.)[0]
    pmid = (prior2+prior1)/2.

    if len(ind_max) > 0 or len(ind_max_LU) > 0:
        log.writelog('Warning: >=1 params hit the upper bound in the lsq fit. '
                     'Setting to the middle of the interval.')
        freepars[u][ind_max] = pmid[u][ind_max]
        freepars[lu][ind_max_LU] = (np.exp(prior2[lu][ind_max_LU]) +
                                    np.exp(prior1[lu][ind_max_LU]))/2.
    if len(ind_min) > 0 or len(ind_min_LU) > 0:
        log.writelog('Warning: >=1 params hit the lower bound in the lsq fit. '
                     'Setting to the middle of the interval.')
        freepars[u][ind_min] = pmid[u][ind_min]
        freepars[lu][ind_min_LU] = (np.exp(prior2[lu][ind_min_LU]) +
                                    np.exp(prior1[lu][ind_min_LU]))/2.

    # Generate the walker positions
    pos = np.array([freepars + step_size*np.random.randn(ndim)
                    for i in range(nwalkers)])

    # Make sure the walker positions obey the priors
    in_range = np.array([((prior1[u] <= ii).all() and (ii <= prior2[u]).all())
                         for ii in pos[:, u]])
    in_range2 = np.array([((prior1[lu] <= np.log(ii)).all() and
                           (np.log(ii) <= prior2[lu]).all())
                          for ii in pos[:, lu]])
    if not np.all(in_range) or not np.all(in_range2):
        log.writelog('Not all walkers were initialized within the priors, '
                     'using a smaller proposal distribution')
        pos = pos[in_range]
        # Make sure the step size is well within the limits
        uniform_step = np.abs(np.append((prior2-freepars).reshape(-1, 1)/10,
                                        (freepars-prior1).reshape(-1, 1)/10,
                                        axis=1))
        step_size_options = np.append(step_size.reshape(-1, 1), uniform_step,
                                      axis=1)
        if len(lu) != 0:
            step_size_options[lu, 1] = np.abs((np.exp(prior2[lu]) -
                                               freepars[lu]).reshape(-1, 1)/10)
            step_size_options[lu, 2] = np.abs((np.exp(prior1[lu]) -
                                               freepars[lu]).reshape(-1, 1)/10)
        if len(n) != 0:
            step_size_options[n, 1:] = step_size_options[n, 0].reshape(-1, 1)
        step_size = np.min(step_size_options, axis=1)
        if pos.shape[0] == 0:
            remove_zeroth = True
            new_nwalkers = nwalkers-len(pos)
            pos = np.zeros((1, ndim))
        else:
            remove_zeroth = False
            new_nwalkers = nwalkers-len(pos)
        pos = np.append(pos, np.array([freepars +
                                       step_size*np.random.randn(ndim)
                                       for i in range(new_nwalkers)
                                       ]).reshape(-1, ndim), axis=0)
        if remove_zeroth:
            pos = pos[1:]
        in_range = np.array([((prior1[u] <= ii).all() and
                              (ii <= prior2[u]).all())
                             for ii in pos[:, u]])
        in_range2 = np.array([((prior1[lu] <= np.log(ii)).all() and
                               (np.log(ii) <= prior2[lu]).all())
                              for ii in pos[:, lu]])
    if not np.any(in_range) and not np.any(in_range2):
        raise AssertionError('Failed to initialize any walkers within the set '
                             'bounds for all parameters!\n'
                             'Check your stating position, decrease your step '
                             'size, or increase the bounds on your parameters')
    elif not np.all(in_range) or not np.all(in_range2):
        old_nwalkers = nwalkers
        pos = pos[in_range+in_range2]
        nwalkers = pos.shape[0]
        if nwalkers > ndim:
            log.writelog(f'Warning: Failed to initialize all walkers within '
                         f'the set bounds for all parameters!\nUsing '
                         f'{nwalkers} walkers instead of the initially '
                         f'requested {old_nwalkers} walkers')
        else:
            message = (f'Error: Failed to initialize all walkers within the '
                       f'set bounds for all parameters!\nUsing '
                       f'{nwalkers} walkers instead of the initially requested'
                       f' {old_nwalkers} walkers is not permitted as there are'
                       f' {ndim} fitted parameters')
            log.writelog(message, mute=True)
            raise AssertionError(message)
    return pos, nwalkers


def dynestyfitter(lc, model, meta, log, **kwargs):
    """Perform sampling using dynesty.

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
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

    - December 29, 2021 Taylor Bell
        Updated documentation. Reduced repeated code.
    - January 7-22, 2022 Megan Mansfield
        Adding ability to do a single shared fit across all channels
    - February 23-25, 2022 Megan Mansfield
        Added log-uniform and Gaussian priors.
    - February 28-March 1, 2022 Caroline Piaulet
        Adding scatter_ppm parameter.
    - Mar 13-Apr 18, 2022 Caroline Piaulet
        Record an astropy table for mean, median, percentiles,
        +/- 1 sigma, all params
    """
    # Group the different variable types
    freenames = lc.freenames
    freepars, prior1, prior2, priortype, indep_vars = \
        group_variables(model)
    if meta.old_fitparams is not None:
        freepars = load_old_fitparams(lc, meta, log, freenames, 'dynesty')

    # DYNESTY
    nlive = meta.run_nlive  # number of live points
    bound = meta.run_bound  # use MutliNest algorithm for bounds
    ndims = len(freepars)  # two parameters
    sample = meta.run_sample  # uniform sampling
    tol = meta.run_tol  # the stopping criterion

    start_lnprob = lnprob(freepars, lc, model, prior1, prior2, priortype,
                          freenames)
    log.writelog(f'Starting lnprob: {start_lnprob}', mute=(not meta.verbose))

    # Plot starting point
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta,
                       fitter='dynestyStartingPoint')
        # Plot GP starting point
        if model.GP:
            plots.plot_GP_components(lc, model, meta,
                                     fitter='dynestyStartingPoint')

    # START DYNESTY
    l_args = [lc, model, freenames]

    log.writelog('Running dynesty...')

    min_nlive = int(np.ceil(ndims*(ndims+1)//2))
    if nlive == 'min':
        nlive = min_nlive
        log.writelog(f'Using {nlive} live points...')
    elif nlive < min_nlive:
        log.writelog(f'**** WARNING: You should set run_nlive to at least '
                     f'{min_nlive} ****')

    if meta.ncpu > 1:
        pool = Pool(meta.ncpu)
        queue_size = meta.ncpu
    else:
        meta.ncpu = 1
        pool = None
        queue_size = None
    sampler = NestedSampler(ln_like, ptform, ndims, pool=pool,
                            queue_size=queue_size, bound=bound, sample=sample,
                            nlive=nlive, logl_args=l_args,
                            ptform_args=[prior1, prior2, priortype])
    sampler.run_nested(dlogz=tol, print_progress=True)  # output progress bar
    res = sampler.results  # get results dictionary from sampler
    if meta.ncpu > 1:
        pool.close()
        pool.join()

    log.writelog('', mute=(not meta.verbose))
    # Need to temporarily redirect output since res.summar() prints rather
    # than returns a string
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    res.summary()
    sys.stdout = old_stdout
    log.writelog(mystdout.getvalue(), mute=(not meta.verbose))

    # get function that resamples from the nested samples to give sampler
    # with equal weight
    # draw posterior samples
    weights = np.exp(res['logwt'] - res['logz'][-1])
    samples = resample_equal(res.samples, weights)
    log.writelog('Number of posterior samples is {}'.format(len(samples)),
                 mute=(not meta.verbose))

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
    lc.unc_fit = update_uncertainty(fit_params, lc.nints, lc.unc, freenames,
                                    lc.nchannel_fitted)

    # Save the fit ASAP so plotting errors don't make you lose everything
    save_fit(meta, lc, model, 'dynesty', t_results, freenames, samples)

    end_lnprob = lnprob(fit_params, lc, model, prior1, prior2, priortype,
                        freenames)
    log.writelog(f'Ending lnprob: {end_lnprob}', mute=(not meta.verbose))

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, log, model, meta, freenames)

    log.writelog('\nDYNESTY RESULTS:')
    for i in range(ndims):
        if 'scatter_mult' in freenames[i]:
            chan = freenames[i].split('_')[-1]
            if chan.isnumeric():
                chan = int(chan)
            else:
                chan = 0
            trim1, trim2 = get_trim(meta.nints, chan)
            unc = np.ma.median(lc.unc[trim1:trim2])
            scatter_ppm = 1e6*fit_params[i]*unc
            scatter_ppm_upper = 1e6*upper_errs[i]*unc
            scatter_ppm_lower = 1e6*lower_errs[i]*unc
            log.writelog(f'{freenames[i]}: {fit_params[i]} (+{upper_errs[i]},'
                         f' -{lower_errs[i]}); {scatter_ppm} '
                         f'(+{scatter_ppm_upper}, -{scatter_ppm_lower}) ppm')
        else:
            log.writelog(f'{freenames[i]}: {fit_params[i]} (+{upper_errs[i]},'
                         f' -{lower_errs[i]})')
    log.writelog('')

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter='dynesty')

    # Plot GP fit + components
    if model.GP and meta.isplots_S5 >= 1:
        plots.plot_GP_components(lc, model, meta, fitter='dynesty')

    # Zoom in on phase variations
    if meta.isplots_S5 >= 1 and ('sinusoid_pc' in meta.run_myfuncs
                                 or 'poet_pc' in meta.run_myfuncs
                                 or 'quasilambert_pc' in meta.run_myfuncs):
        plots.plot_phase_variations(lc, model, meta, fitter='dynesty')

    # Plot Allan plot
    if meta.isplots_S5 >= 3 and np.size(lc.flux) > 20:
        # mc3.stats.time_avg breaks when testing with a small
        # number of integrations
        plots.plot_rms(lc, model, meta, fitter='dynesty')

    # Plot residuals distribution
    if meta.isplots_S5 >= 3:
        plots.plot_res_distr(lc, model, meta, fitter='dynesty')

    # plot using corner.py
    if meta.isplots_S5 >= 5:
        plots.plot_corner(samples, lc, meta, freenames, fitter='dynesty')

    # Make a new model instance
    best_model = copy.deepcopy(model)
    best_model.components[0].update(fit_params)
    best_model.__setattr__('chi2red', chi2red)
    best_model.__setattr__('fit_params', fit_params)

    return best_model


def lmfitter(lc, model, meta, log, **kwargs):
    """Perform a fit using lmfit.

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
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

    - December 29, 2021 Taylor Bell
        Updated documentation. Reduced repeated code.
    - February 28-March 1, 2022 Caroline Piaulet
        Adding scatter_ppm parameter.
    - Mar 13-Apr 18, 2022 Caroline Piaulet
         Record an astropy table for parameter values
    """
    # TODO: Do something so that duplicate param names can all be handled
    # (e.g. two Polynomail models with c0). Perhaps append something to the
    # parameter name like c0_1 and c0_2?)
    freenames = lc.freenames

    # Group the different variable types
    param_list, indep_vars = group_variables_lmfit(model)

    # Add the time as an independent variable
    indep_vars['time'] = lc.time

    # Initialize lmfit Params object
    initialParams = lmfit.Parameters()
    # Insert parameters
    initialParams.add_many(*param_list)

    # Create the lmfit lightcurve model
    lcmodel = lmfit.Model(model.eval)
    lcmodel.independent_vars = indep_vars.keys()

    # Fit light curve model to the simulated data
    result = lcmodel.fit(lc.flux, weights=1/lc.unc, params=initialParams,
                         **indep_vars, **kwargs)

    # Get the best fit params
    fit_params = result.__dict__['params']
    # new_params = [(fit_params.get(i).name, fit_params.get(i).value,
    #                fit_params.get(i).vary, fit_params.get(i).min,
    #                fit_params.get(i).max) for i in fit_params]

    # Create table of results
    t_results = table.Table([freenames, fit_params],
                            names=("Parameter", "Mean"))

    model.update(fit_params)
    lc.unc_fit = update_uncertainty(fit_params, lc.nints, lc.unc, freenames,
                                    lc.nchannel_fitted)

    # Save the fit ASAP
    save_fit(meta, lc, model, 'lmfitter', t_results, freenames)

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, log, model, meta, freenames)

    # Log results
    log.writelog(result.fit_report(), mute=(not meta.verbose))

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter='lmfitter')

    # Plot GP fit + components
    if model.GP and meta.isplots_S5 >= 1:
        plots.plot_GP_components(lc, model, meta, fitter='lmfitter')

    # Zoom in on phase variations
    if meta.isplots_S5 >= 1 and ('sinusoid_pc' in meta.run_myfuncs
                                 or 'poet_pc' in meta.run_myfuncs
                                 or 'quasilambert_pc' in meta.run_myfuncs):
        plots.plot_phase_variations(lc, model, meta, fitter='lmfitter')

    # Plot Allan plot
    if meta.isplots_S5 >= 3 and np.size(lc.flux) > 20:
        # mc3.stats.time_avg breaks when testing with a small
        # number of integrations
        plots.plot_rms(lc, model, meta, fitter='lmfitter')

    # Plot residuals distribution
    if meta.isplots_S5 >= 3:
        plots.plot_res_distr(lc, model, meta, fitter='lmfitter')

    # Create new model with best fit parameters
    best_model = copy.deepcopy(model)
    best_model.components[0].update(fit_params)
    best_model.__setattr__('chi2red', chi2red)
    best_model.__setattr__('fit_params', fit_params)

    return best_model


def group_variables(model):
    """Group variables into fitted and frozen.

    Parameters
    ----------
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit

    Returns
    -------
    freepars : np.array
        The fitted variables.
    prior1 : np.array
        The lower bound for constrained variables with uniform/log uniform
        priors, or mean for constrained variables with Gaussian priors.
    prior2 : np.array
        The upper bound for constrained variables with uniform/log uniform
        priors, or mean for constrained variables with Gaussian priors.
    priortype : np.array
        Keywords indicating the type of prior for each free parameter.
    indep_vars : dict
        The frozen variables.

    Notes
    -----
    History:

    - December 29, 2021 Taylor Bell
        Moved code to separate function to reduce repeated code.
    - January 11, 2022 Megan Mansfield
        Added ability to have shared parameters
    - February 23-25, 2022 Megan Mansfield
        Added log-uniform and Gaussian priors.
    """
    parameters_dict = model.components[0].parameters.dict
    freenames = model.components[0].freenames

    # Group the different variable types
    freepars = []
    prior1 = []
    prior2 = []
    priortype = []
    for ii, name in enumerate(freenames):
        param = parameters_dict[name]
        # param = list(param)
        if ((param[1] == 'free') or (param[1] == 'shared')
                or ('white' in param[1])):
            freepars.append(param[0])
            if len(param) == 5:  # If prior is specified.
                prior1.append(param[2])
                prior2.append(param[3])
                priortype.append(param[4])
            elif (len(param) > 3) & (len(param) < 5):
                # If prior bounds are specified but not the prior type
                raise IndexError("If you want to specify prior parameters, you"
                                 " must also specify the prior type: 'U', 'LU'"
                                 ", or 'N'.")
            else:
                # If no prior is specified,
                # assume uniform prior with infinite bounds.
                prior1.append(-np.inf)
                prior2.append(np.inf)
                priortype.append('U')
    freepars = np.array(freepars)
    prior1 = np.array(prior1)
    prior2 = np.array(prior2)
    priortype = np.array(priortype)
    indep_vars = dict([[key, parameters_dict[key][0]]
                       for key in parameters_dict.keys()
                       if key not in freenames])

    return freepars, prior1, prior2, priortype, indep_vars


def group_variables_lmfit(model):
    """Group variables into fitted and frozen for lmfit fitter.

    Parameters
    ----------
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit

    Returns
    -------
    paramlist : list
        The fitted variables.
    indep_vars : dict
        The frozen variables.

    Notes
    -----
    History:

    - December 29, 2021 Taylor Bell
        Moved code to separate function to look similar to other fitters.
    """
    all_params = [i for j in [model.components[n].parameters.dict.items()
                  for n in range(len(model.components))] for i in j]

    # Group the different variable types
    param_list = []
    indep_vars = {}
    for param in all_params:
        param = list(param)
        if param[1][1] == 'free':
            param[1][1] = True
            param_list.append(tuple(param))
        elif param[1][1] == 'fixed':
            param[1][1] = False
            param_list.append(tuple(param))
        else:
            indep_vars[param[0]] = param[1]

    return param_list, indep_vars


def load_old_fitparams(lc, meta, log, freenames, fitter):
    """Load in the best-fit values from a previous fit.

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    freenames : list
        The names of the fitted parameters.
    fitter : str
        The name of the fitter, to figure out the old fitparams filename name.

    Returns
    -------
    fitted_values : np.array
        The best-fit values from a previous fit

    Raises
    ------
    AssertionError
        The old fit is incompatible with the current fit.
    """
    if lc.white:
        channel_tag = '_white'
    elif lc.share:
        channel_tag = '_shared'
    else:
        ch_number = str(lc.channel).zfill(len(str(lc.nchannel)))
        channel_tag = f'_ch{ch_number}'

    foldername = os.path.join(meta.topdir, *meta.old_fitparams.split(os.sep))
    fname = f'S5_{fitter}_fitparams{channel_tag}.csv'
    fitted_values = pd.read_csv(os.path.join(foldername, fname),
                                escapechar='#', skipinitialspace=True)
    full_keys = np.array(fitted_values['Parameter'])

    if np.all(full_keys != freenames):
        log.writelog('Old fit does not have the same fitted parameters and '
                     'cannot be used to initialize the new fit.\n'
                     'The old fit included:\n['+','.join(full_keys)+']\n'
                     'The new fit included:\n['+','.join(freenames)+']',
                     mute=True)
        raise AssertionError('Old fit does not have the same fitted parameters'
                             ' and cannot be used to initialize the new fit.\n'
                             'The old fit included:\n['+','.join(full_keys) +
                             ']\nThe new fit included:\n['+','.join(freenames)
                             + ']')

    if '50th' in fitted_values.keys():
        # A sampler was used, so use the (more reliable) median
        oldfitparam = fitted_values['50th'].to_numpy()
    else:
        # An optimizer was used, so only the Mean column will be populated
        oldfitparam = fitted_values['Mean'].to_numpy()

    return oldfitparam


def save_fit(meta, lc, model, fitter, results_table, freenames, samples=[]):
    """Save a fit as a txt file as well as the entire chain if provided.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit.
    fitter : str
        The current fitter being used.
    fit_params : np.array
        The best-fit values from the current fit.
    freenames : list
        The list of fitted parameter names.
    samples : ndarray; optional
        The full chain from a sampling method, by default [].

    Notes
    -----
    History:

    - Mar 13-Apr 18, 2022 Caroline Piaulet
        Record an astropy table for mean, median, percentiles,
        +/- 1 sigma, all params
    """
    if lc.white:
        channel_tag = '_white'
    elif lc.share:
        channel_tag = '_shared'
    else:
        ch_number = str(lc.channel).zfill(len(str(lc.nchannel)))
        channel_tag = f'_ch{ch_number}'

    # Save the fitted parameters and their uncertainties (if possible)
    fname = f'S5_{fitter}_fitparams{channel_tag}'
    results_table.write(meta.outputdir+fname+'.csv', format='csv',
                        overwrite=False)

    # Save the chain from the sampler using Astraeus (if a chain was provided)
    if len(samples) != 0:
        fname = meta.outputdir+f'S5_{fitter}_samples{channel_tag}.h5'
        ds = dict([(freenames[i], xr.DataArray(samples[:, i], dims=['sample'],
                                               name=freenames[i]))
                   for i in range(len(freenames))])
        ds = xrio.makeDataset(ds)
        xrio.writeXR(fname, ds)

    # Directory structure should not use expanded HW values
    spec_hw_val = meta.spec_hw
    bg_hw_val = meta.bg_hw
    spec_hw_val //= meta.expand
    if not isinstance(bg_hw_val, str):
        # Only divide if value is not a string (spectroscopic modes)
        bg_hw_val //= meta.expand
    # Save the S5 outputs in a human readable ecsv file
    if not isinstance(meta.bg_hw, str):
        # Only divide if value is not a string (spectroscopic modes)
        bg_hw = meta.bg_hw//meta.expand
    else:
        bg_hw = meta.bg_hw
    event_ap_bg = meta.eventlabel+"_ap"+str(meta.spec_hw//meta.expand) + \
        '_bg'+str(bg_hw)
    meta.tab_filename_s5 = (meta.outputdir+'S5_'+event_ap_bg+"_Table_Save" +
                            channel_tag+'.txt')
    wavelengths = np.mean(np.append(meta.wave_low.reshape(1, -1),
                                    meta.wave_hi.reshape(1, -1), axis=0),
                          axis=0)
    wave_errs = (meta.wave_hi-meta.wave_low)/2
    # Evaluate each individual model for easier access outside of Eureka!
    individual_models = []
    for comp in model.components:
        if comp.name != 'GP':
            individual_models.append([comp.name, comp.eval()])
        else:
            fit = model.eval(incl_GP=False)
            individual_models.append([comp.name, comp.eval(fit)])
    individual_models = np.array(individual_models, dtype=object)

    model_lc = model.eval()
    residuals = lc.flux-model_lc
    astropytable.savetable_S5(meta.tab_filename_s5, meta, lc.time,
                              wavelengths[lc.fitted_channels],
                              wave_errs[lc.fitted_channels],
                              lc.flux, lc.unc_fit, individual_models, model_lc,
                              residuals)

    return
