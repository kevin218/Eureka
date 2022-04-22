import numpy as np
import pandas as pd
import copy
from io import StringIO
import os, sys
import h5py
import time as time_pkg

from scipy.optimize import minimize
import lmfit
import emcee

from dynesty import NestedSampler
from dynesty.utils import resample_equal

from .likelihood import computeRedChiSq, lnprob, ln_like, ptform
from . import plots_s5 as plots
from ..lib import astropytable

from multiprocessing import Pool

def lsqfitter(lc, model, meta, log, calling_function='lsq', **kwargs):
    """Perform least-squares fit.

    Parameters
    ----------
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    meta: MetaClass
        The metadata object
    log: logedit.Logedit
        The open log in which notes from this step can be added.
    **kwargs:
        Arbitrary keyword arguments.

    Returns
    -------
    best_model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model after fitting

    Notes
    -----
    History:

    - December 29-30, 2021 Taylor Bell
        Updated documentation and arguments. Reduced repeated code.
        Also saving covariance matrix for later estimation of sampler step size.
    - January 7-22, 2022 Megan Mansfield
        Adding ability to do a single shared fit across all channels
    - February 28-March 1, 2022 Caroline Piaulet
        Adding scatter_ppm parameter
    """
    # Group the different variable types
    freenames, freepars, prior1, prior2, priortype, indep_vars = group_variables(model)
    if hasattr(meta, 'old_fitparams') and meta.old_fitparams is not None:
        freepars = load_old_fitparams(meta, log, lc.channel, freenames)

    start_lnprob = lnprob(freepars, lc, model, prior1, prior2, priortype, freenames)
    log.writelog(f'Starting lnprob: {start_lnprob}', mute=(not meta.verbose))

    neg_lnprob = lambda theta, lc, model, prior1, prior2, priortype, freenames: -lnprob(theta, lc, model, prior1, prior2, priortype, freenames)
    global lsq_t0
    lsq_t0 = time_pkg.time()
    def callback_full(theta, lc, model, prior1, prior2, priortype, freenames):
        global lsq_t0
        if (time_pkg.time()-lsq_t0)>0.5:
            lsq_t0 = time_pkg.time()
            print('Current lnprob = ', lnprob(theta, lc, model, prior1, prior2, priortype, freenames), end='\r')
    callback = lambda theta: callback_full(theta, lc, model, prior1, prior2, priortype, freenames)

    if not hasattr(meta, 'lsq_method'):
        log.writelog('No lsq optimization method specified - using Nelder-Mead by default.')
        meta.lsq_method = 'Nelder-Mead'
    if not hasattr(meta, 'lsq_tol'):
        log.writelog('No lsq tolerance specified - using 1e-6 by default.')
        meta.lsq_tol = 1e-6
    if not hasattr(meta, 'lsq_maxiter'):
        meta.lsq_maxiter = None
    results = minimize(neg_lnprob, freepars, args=(lc, model, prior1, prior2, priortype, freenames), method=meta.lsq_method, tol=meta.lsq_tol, options={'maxiter':meta.lsq_maxiter}, callback=callback)

    log.writelog("\nVerbose lsq results: {}\n".format(results), mute=(not meta.verbose))
    if not meta.verbose:
        log.writelog("Success?: {}".format(results.success))
        log.writelog(results.message)

    # Get the best fit params
    fit_params = results.x

    model.update(fit_params, freenames)
    if "scatter_ppm" in freenames:
        ind = [i for i in np.arange(len(freenames)) if freenames[i][0:11] == "scatter_ppm"]
        for chan in range(len(ind)):
            lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = fit_params[ind[chan]] * 1e-6
    elif "scatter_mult" in freenames:
        ind = [i for i in np.arange(len(freenames)) if freenames[i][0:12] == "scatter_mult"]
        if not hasattr(lc, 'unc_fit'):
            lc.unc_fit = copy.copy(lc.unc)
        for chan in range(len(ind)):
            lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = fit_params[ind[chan]] * lc.unc[chan*lc.time.size:(chan+1)*lc.time.size]

    # Save the fit ASAP
    save_fit(meta, lc, model, calling_function, fit_params, freenames)

    end_lnprob = lnprob(fit_params, lc, model, prior1, prior2, priortype, freenames)
    log.writelog(f'Ending lnprob: {end_lnprob}', mute=(not meta.verbose))

    # Make a new model instance
    best_model = copy.copy(model)
    best_model.components[0].update(fit_params, freenames)

    # Save the covariance matrix in case it's needed to estimate step size for a sampler
    # FINDME
    # Commented out for now because op.least_squares() doesn't provide covariance matrix
    # Need to compute using Jacobian matrix instead (hess_inv = (J.T J)^{-1})
    # model_lc = model.eval()
    # if results[1] is not None:
    #     residuals = (lc.flux - model_lc)
    #     cov_mat = results[1]*np.var(residuals)
    # else:
    #     # Sometimes lsq will fail to converge and will return a None covariance matrix
    #     cov_mat = None
    cov_mat = None
    best_model.__setattr__('cov_mat',cov_mat)

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter=calling_function)

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, log, model, meta, freenames)
    
    print('\nLSQ RESULTS:')
    for i in range(len(freenames)):
        if 'scatter_mult' in freenames[i]:
            chan = freenames[i].split('_')[-1]
            if chan.isnumeric():
                chan = int(chan)
            else:
                chan = 0
            scatter_ppm = fit_params[i] * np.ma.median(lc.unc[chan*lc.time.size:(chan+1)*lc.time.size]) * 1e6
            log.writelog('{0}: {1}; {2} ppm'.format(freenames[i], fit_params[i], scatter_ppm))
        else:
            log.writelog('{0}: {1}'.format(freenames[i], fit_params[i]))
    log.writelog('')

    # Plot Allan plot
    if meta.isplots_S5 >= 3 and calling_function=='lsq':
        # This plot is only really useful if you're actually using the lsq fitter, otherwise don't make it
        plots.plot_rms(lc, model, meta, fitter=calling_function)

    # Plot residuals distribution
    if meta.isplots_S5 >= 3 and calling_function=='lsq':
        plots.plot_res_distr(lc, model, meta, fitter=calling_function)

    best_model.__setattr__('chi2red',chi2red)
    best_model.__setattr__('fit_params',fit_params)

    return best_model

def demcfitter(lc, model, meta, log, **kwargs):
    """Perform sampling using Differential Evolution Markov Chain.

    This is an empty placeholder function to be filled later.

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    meta : MetaClass
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
        Updated documentation and arguments
    """
    best_model = None
    return best_model

def emceefitter(lc, model, meta, log, **kwargs):
    """Perform sampling using emcee.

    Parameters
    ----------
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    meta: MetaClass
        The metadata object
    log: logedit.Logedit
        The open log in which notes from this step can be added.
    **kwargs:
        Arbitrary keyword arguments.

    Returns
    -------
    best_model: eureka.S5_lightcurve_fitting.models.CompositeModel
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

    """
    # Group the different variable types
    freenames, freepars, prior1, prior2, priortype, indep_vars = group_variables(model)
    if hasattr(meta, 'old_fitparams') and meta.old_fitparams is not None:
        freepars = load_old_fitparams(meta, log, lc.channel, freenames)
    ndim = len(freenames)

    if hasattr(meta, 'old_chain') and meta.old_chain is not None:
        pos, nwalkers = start_from_oldchain_emcee(meta, log, ndim, lc.channel, freenames)
    else:
        if not hasattr(meta, 'lsq_first') or meta.lsq_first:
            # Only call lsq fitter first if asked or lsq_first option wasn't passed (allowing backwards compatibility)
            log.writelog('\nCalling lsqfitter first...')
            # RUN LEAST SQUARES
            lsq_sol = lsqfitter(lc, model, meta, log, calling_function='emcee_lsq', **kwargs)

            freepars = lsq_sol.fit_params

            # SCALE UNCERTAINTIES WITH REDUCED CHI2
            if meta.rescale_err:
                lc.unc *= np.sqrt(lsq_sol.chi2red)
        else:
            lsq_sol = None
        pos, nwalkers = initialize_emcee_walkers(meta, log, ndim, lsq_sol, freepars, prior1, prior2, priortype)

    start_lnprob = lnprob(np.median(pos, axis=0), lc, model, prior1, prior2, priortype, freenames)
    log.writelog(f'Starting lnprob: {start_lnprob}', mute=(not meta.verbose))

    # Initialize tread pool
    if hasattr(meta, 'ncpu') and meta.ncpu > 1:
        pool = Pool(meta.ncpu)
    else:
        meta.ncpu = 1
        pool = None
    
    # Run emcee burn-in
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(lc, model, prior1, prior2, priortype, freenames),
                                    pool=pool)
    log.writelog('Running emcee burn-in...')
    state = sampler.run_mcmc(pos, meta.run_nsteps, progress=True)
    # # Log some details about the burn-in phase
    # log.writelog("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)), mute=(not meta.verbose))
    # try:
    #     log.writelog("Mean autocorrelation time: {0:.3f} steps".format(sampler.get_autocorr_time()), mute=(not meta.verbose))
    # except:
    #     log.writelog("Error: Unable to estimate the autocorrelation time!", mute=(not meta.verbose))
    # mid_lnprob = lnprob(np.median(sampler.get_chain()[-1], axis=0), lc, model, prior1, prior2, priortype, freenames)
    # log.writelog(f'Intermediate lnprob: {mid_lnprob}', mute=(not meta.verbose))
    # if meta.isplots_S5 >= 3:
    #     plots.plot_chain(sampler.get_chain(), lc, meta, freenames, fitter='emcee', burnin=True)
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

    # Compute the medians and uncertainties
    fit_params = []
    upper_errs = []
    lower_errs = []
    for i in range(ndim):
        q = np.percentile(samples[:, i], [16, 50, 84])
        lower_errs.append(q[0])
        fit_params.append(q[1])
        upper_errs.append(q[2])
    fit_params = np.array(fit_params)
    upper_errs = np.array(upper_errs)-fit_params
    lower_errs = fit_params-np.array(lower_errs)

    model.update(fit_params, freenames)
    if "scatter_ppm" in freenames:
        ind = [i for i in np.arange(len(freenames)) if freenames[i][0:11] == "scatter_ppm"]
        for chan in range(len(ind)):
            lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = fit_params[ind[chan]] * 1e-6
    elif "scatter_mult" in freenames:
        ind = [i for i in np.arange(len(freenames)) if freenames[i][0:12] == "scatter_mult"]
        for chan in range(len(ind)):
            lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = fit_params[ind[chan]] * lc.unc[chan*lc.time.size:(chan+1)*lc.time.size]
    else:
        lc.unc_fit = lc.unc

    # Save the fit ASAP so plotting errors don't make you lose everything
    save_fit(meta, lc, model, 'emcee', fit_params, freenames, samples, upper_errs=upper_errs, lower_errs=lower_errs)

    end_lnprob = lnprob(fit_params, lc, model, prior1, prior2, priortype, freenames)
    log.writelog(f'Ending lnprob: {end_lnprob}', mute=(not meta.verbose))
    log.writelog("Mean acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)), mute=(not meta.verbose))
    try:
        log.writelog("Mean autocorrelation time: {0:.3f} steps".format(sampler.get_autocorr_time()), mute=(not meta.verbose))
    except:
        log.writelog("WARNING: Unable to estimate the autocorrelation time!", mute=(not meta.verbose))

    if meta.isplots_S5 >= 3:
        plots.plot_chain(sampler.get_chain(), lc, meta, freenames, fitter='emcee', burnin=True, nburn=meta.run_nburn)
        plots.plot_chain(sampler.get_chain(discard=meta.run_nburn), lc, meta, freenames, fitter='emcee', burnin=False)

    # Make a new model instance
    best_model = copy.copy(model)
    best_model.components[0].update(fit_params, freenames)

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter='emcee')

    #Plot GP fit + components
    if model.GP and meta.isplots_S5 >= 1:
        plots.plot_GP_components(lc, model, meta, fitter='emcee')

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
            scatter_ppm = fit_params[i] * np.ma.median(lc.unc[chan*lc.time.size:(chan+1)*lc.time.size]) * 1e6
            scatter_ppm_upper = upper_errs[i] * np.ma.median(lc.unc[chan*lc.time.size:(chan+1)*lc.time.size]) * 1e6
            scatter_ppm_lower = lower_errs[i] * np.ma.median(lc.unc[chan*lc.time.size:(chan+1)*lc.time.size]) * 1e6
            log.writelog('{0}: {1} (+{2}, -{3}); {4} (+{5}, -{6}) ppm'.format(freenames[i], fit_params[i], upper_errs[i], lower_errs[i], scatter_ppm, scatter_ppm_upper, scatter_ppm_lower))
        else:
            log.writelog('{0}: {1} (+{2}, -{3})'.format(freenames[i], fit_params[i], upper_errs[i], lower_errs[i]))
    log.writelog('')
    
    # Plot Allan plot
    if meta.isplots_S5 >= 3:
        plots.plot_rms(lc, model, meta, fitter='emcee')
        
    # Plot residuals distribution
    if meta.isplots_S5 >= 3:
        plots.plot_res_distr(lc, model, meta, fitter='emcee')

    if meta.isplots_S5 >= 5:
        plots.plot_corner(samples, lc, meta, freenames, fitter='emcee')

    best_model.__setattr__('chi2red',chi2red)
    best_model.__setattr__('fit_params',fit_params)

    return best_model

def start_from_oldchain_emcee(meta, log, ndim, channel, freenames):
    if meta.sharedp:
        channel_key = 'shared'
    else:
        channel_key = f'ch{channel}'
    
    foldername = os.path.join(meta.topdir, *meta.old_chain.split(os.sep))
    fname = f'S5_emcee_fitparams_{channel_key}.csv'
    fitted_values = pd.read_csv(os.path.join(foldername,fname), escapechar='#', skipinitialspace=True)
    full_keys = np.array(fitted_values.keys())
    full_keys[0] = full_keys[0][1:] # Remove the " " from the start of the first key

    if np.all(full_keys!=freenames):
        log.writelog('Old chain does not have the same fitted parameters and cannot be used to initialize the new fit.\n'+
                     'The old chain included:\n['+','.join(full_keys)+']\n'+
                     'The new chain included:\n['+','.join(freenames)+']', mute=True)
        raise AssertionError('Old chain does not have the same fitted parameters and cannot be used to initialize the new fit.\n'+
                             'The old chain included:\n['+','.join(full_keys)+']\n'+
                             'The new chain included:\n['+','.join(freenames)+']')
    
    fname = f'S5_emcee_samples_{channel_key}'
    if os.path.isfile(os.path.join(foldername,fname)+'.h5'):
        # New code to load HDF5 files
        full_fname = os.path.join(foldername,fname)+'.h5'
        with h5py.File(full_fname, 'r') as hf:
            samples = hf['samples'][:]
    else:
        # Keep this old code for the time being to allow backwards compatibility
        full_fname = os.path.join(foldername,fname)+'.csv'
        samples = np.loadtxt(full_fname, delimiter=',')
    log.writelog(f'Old chain path: {full_fname}')

    # Initialize the walkers using samples from the old chain
    nwalkers = meta.run_nwalkers
    pos = samples[-nwalkers:]
    walkers_used = nwalkers
    
    # Make sure that no walkers are starting in the same place as they would then exactly follow each other
    repeat_pos = np.where([np.any(np.all(pos[i] == np.delete(pos, i, axis=0), axis=1)) for i in range(pos.shape[0])])[0]
    while len(repeat_pos) > 0 and samples.shape[0]>(walkers_used+len(repeat_pos)):
        pos[repeat_pos] = samples[:-walkers_used][-len(repeat_pos):]
        walkers_used += len(repeat_pos)
        repeat_pos = np.where([np.any(np.all(pos[i] == np.delete(pos, i, axis=0), axis=1)) for i in range(pos.shape[0])])[0]

    # If unable to initialize all walkers in unique starting locations, use fewer walkers unless there'd be fewer walkers than dimensions
    if len(repeat_pos)>0 and (nwalkers-len(repeat_pos) > ndim):
        pos = np.delete(pos, repeat_pos, axis=0)
        nwalkers = pos.shape[0]
        log.writelog('Warning: Unable to initialize all walkers at different positions using old chain!\n'+
                     'Using {} walkers instead of the initially requested {} walkers'.format(nwalkers, meta.run_nwalkers))
    elif len(repeat_pos)>0:
        log.writelog('Error: Unable to initialize all walkers at different positions using old chain!\n'+
                     'Using {} walkers instead of the initially requested {} walkers is not permitted as there are {} fitted parameters'.format(nwalkers, meta.run_nwalkers, ndim), mute=True)
        raise AssertionError('Error: Unable to initialize all walkers at different positions using old chain!\n'+
                             'Using {} walkers instead of the initially requested {} walkers is not permitted as there are {} fitted parameters'.format(nwalkers, meta.run_nwalkers, ndim))

    return pos, nwalkers

def initialize_emcee_walkers(meta, log, ndim, lsq_sol, freepars, prior1, prior2, priortype):
    if lsq_sol is not None and lsq_sol.cov_mat is not None:
        step_size = np.diag(lsq_sol.cov_mat)
        ind_zero = np.where(step_size==0.)[0]
        if len(ind_zero):
            step_size[ind_zero][priortype[ind_zero]=='U'] = 0.001*(prior2[ind_zero][priortype[ind_zero]=='U'] - prior1[ind_zero][priortype[ind_zero]=='U'])
            step_size[ind_zero][priortype[ind_zero]=='LU'] = 0.001*(np.exp(prior2[ind_zero][priortype[ind_zero]=='LU']) - np.exp(prior1[ind_zero][priortype[ind_zero]=='LU']))
            step_size[ind_zero][priortype[ind_zero]=='N'] = 0.1*prior2[ind_zero][priortype[ind_zero]=='N']
    else:
        # Sometimes the lsq fitter won't converge and will give None as the covariance matrix
        # In that case, we need to establish the step size in another way. Using a fractional step
        # compared to the prior range can work best for precisely known values like t0 and period
        log.writelog('No covariance matrix from LSQ - falling back on a step size based on the prior range')
        step_size = np.ones(ndim)
        step_size[priortype=='U'] = 0.001*(prior2[priortype=='U'] - prior1[priortype=='U'])
        step_size[priortype=='LU'] = 0.001*(np.exp(prior2[priortype=='LU']) - np.exp(prior1[priortype=='LU']))
        step_size[priortype=='N'] = 0.1*prior2[priortype=='N']
    nwalkers = meta.run_nwalkers

    # make it robust to lsq hitting the upper or lower bound of the param space
    ind_max = np.where(freepars[priortype=='U'] - prior2[priortype=='U'] == 0.)[0]
    ind_min = np.where(freepars[priortype=='U'] - prior1[priortype=='U'] == 0.)[0]
    ind_max_LU = np.where(np.log(freepars[priortype=='LU']) - prior2[priortype=='LU'] == 0.)[0]
    ind_min_LU = np.where(np.log(freepars[priortype=='LU']) - prior1[priortype=='LU'] == 0.)[0]
    pmid = (prior2+prior1)/2.
    if len(ind_max)>0 or len(ind_max_LU)>0:
        log.writelog('Warning: >=1 params hit the upper bound in the lsq fit. Setting to the middle of the interval.')
        freepars[priortype=='U'][ind_max] = pmid[priortype=='U'][ind_max]
        freepars[priortype=='LU'][ind_max_LU] = (np.exp(prior2[priortype=='LU'][ind_max_LU])+np.exp(prior1[priortype=='LU'][ind_max_LU]))/2.
    if len(ind_min)>0 or len(ind_min_LU)>0:
        log.writelog('Warning: >=1 params hit the lower bound in the lsq fit. Setting to the middle of the interval.')
        freepars[priortype=='U'][ind_min] = pmid[priortype=='U'][ind_min]
        freepars[priortype=='LU'][ind_min_LU] = (np.exp(prior2[priortype=='LU'][ind_min_LU])+np.exp(prior1[priortype=='LU'][ind_min_LU]))/2.
    
    # Generate the walker positions
    pos = np.array([freepars + np.array(step_size)*np.random.randn(ndim) for i in range(nwalkers)])

    # Make sure the walker positions obey the priors
    uniformprior = np.where(priortype=='U')[0]
    loguniformprior = np.where(priortype=='LU')[0]
    normalprior = np.where(priortype=='N')[0]
    in_range = np.array([((prior1[uniformprior] <= ii) & (ii <= prior2[uniformprior])).all() for ii in pos[:,uniformprior]])
    in_range2 = np.array([((prior1[loguniformprior] <= np.log(ii)) & (np.log(ii) <= prior2[loguniformprior])).all() for ii in pos[:,loguniformprior]])
    if not (np.all(in_range))&(np.all(in_range2)):
        log.writelog('Not all walkers were initialized within the priors, using a smaller proposal distribution')
        pos = pos[in_range]
        # Make sure the step size is well within the limits
        step_size_options = np.append(step_size.reshape(-1,1), np.abs(np.append((prior2-freepars).reshape(-1,1)/10, (freepars-prior1).reshape(-1,1)/10, axis=1)), axis=1)
        if len(loguniformprior)!=0:
            step_size_options[loguniformprior,1] = np.abs((np.exp(prior2[loguniformprior])-freepars[loguniformprior]).reshape(-1,1)/10)
            step_size_options[loguniformprior,2] = np.abs((np.exp(prior1[loguniformprior])-freepars[loguniformprior]).reshape(-1,1)/10)
        if len(normalprior)!=0:
            step_size_options[normalprior,1:] = step_size_options[normalprior,0].reshape(-1,1)
        step_size = np.min(step_size_options, axis=1)
        if pos.shape[0]==0:
            remove_zeroth = True
            new_nwalkers = nwalkers-len(pos)
            pos = np.zeros((1,ndim))
        else:
            remove_zeroth = False
            new_nwalkers = nwalkers-len(pos)
        pos = np.append(pos, np.array([freepars + np.array(step_size)*np.random.randn(ndim) for i in range(new_nwalkers)]).reshape(-1,ndim), axis=0)
        if remove_zeroth:
            pos = pos[1:]
        in_range = np.array([((prior1[uniformprior] <= ii) & (ii <= prior2[uniformprior])).all() for ii in pos[:,uniformprior]])
        in_range2 = np.array([((prior1[loguniformprior] <= np.log(ii)) & (np.log(ii) <= prior2[loguniformprior])).all() for ii in pos[:,loguniformprior]])
    if not (np.any(in_range))&(np.any(in_range2)):
        raise AssertionError('Failed to initialize any walkers within the set bounds for all parameters!\n'+
                             'Check your stating position, decrease your step size, or increase the bounds on your parameters')
    elif not (np.all(in_range))&(np.all(in_range2)):
        old_nwalkers = nwalkers
        pos = pos[in_range+in_range2]
        nwalkers = pos.shape[0]
        if nwalkers > ndim:
            log.writelog('Warning: Failed to initialize all walkers within the set bounds for all parameters!\n'+
                         'Using {} walkers instead of the initially requested {} walkers'.format(nwalkers, old_nwalkers))
        else:
            log.writelog('Error: Failed to initialize all walkers within the set bounds for all parameters!\n'+
                         'Using {} walkers instead of the initially requested {} walkers is not permitted as there are {} fitted parameters'.format(nwalkers, old_nwalkers, ndim), mute=True)
            raise AssertionError('Error: Failed to initialize all walkers within the set bounds for all parameters!\n'+
                                 'Using {} walkers instead of the initially requested {} walkers is not permitted as there are {} fitted parameters'.format(nwalkers, old_nwalkers, ndim))
    return pos, nwalkers

def dynestyfitter(lc, model, meta, log, **kwargs):
    """Perform sampling using dynesty.

    Parameters
    ----------
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    meta: MetaClass
        The metadata object
    log: logedit.Logedit
        The open log in which notes from this step can be added.
    **kwargs:
        Arbitrary keyword arguments.

    Returns
    -------
    best_model: eureka.S5_lightcurve_fitting.models.CompositeModel
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
    """
    # Group the different variable types
    freenames, freepars, prior1, prior2, priortype, indep_vars = group_variables(model)
    if hasattr(meta, 'old_fitparams') and meta.old_fitparams is not None:
        freepars = load_old_fitparams(meta, log, lc.channel, freenames)

    # DYNESTY
    nlive = meta.run_nlive # number of live points
    bound = meta.run_bound  # use MutliNest algorithm for bounds
    ndims = len(freepars)  # two parameters
    sample = meta.run_sample  # uniform sampling
    tol = meta.run_tol  # the stopping criterion

    start_lnprob = lnprob(freepars, lc, model, prior1, prior2, priortype, freenames)
    log.writelog(f'Starting lnprob: {start_lnprob}', mute=(not meta.verbose))

    # START DYNESTY
    l_args = [lc, model, freenames]

    log.writelog('Running dynesty...')

    min_nlive = int(np.ceil(ndims*(ndims+1)//2))
    if nlive < min_nlive:
        log.writelog(f'**** WARNING: You should set run_nlive to at least {min_nlive} ****')

    if hasattr(meta, 'ncpu') and meta.ncpu > 1:
        pool = Pool(meta.ncpu)
        queue_size = meta.ncpu
    else:
        meta.ncpu = 1
        pool = None
        queue_size = None
    sampler = NestedSampler(ln_like, ptform, ndims, pool=pool, queue_size=queue_size,
                            bound=bound, sample=sample, nlive=nlive, logl_args = l_args,
                            ptform_args=[prior1, prior2, priortype])
    sampler.run_nested(dlogz=tol, print_progress=True)  # output progress bar
    res = sampler.results  # get results dictionary from sampler
    if meta.ncpu > 1:
        pool.close()
        pool.join()

    logZdynesty = res.logz[-1]  # value of logZ
    logZerrdynesty = res.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

    log.writelog('', mute=(not meta.verbose))
    # Need to temporarily redirect output since res.summar() prints rather than returns a string
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    res.summary()
    sys.stdout = old_stdout
    log.writelog(mystdout.getvalue(), mute=(not meta.verbose))

    # get function that resamples from the nested samples to give sampler with equal weight
    # draw posterior samples
    weights = np.exp(res['logwt'] - res['logz'][-1])
    samples = resample_equal(res.samples, weights)
    log.writelog('Number of posterior samples is {}'.format(len(samples)), mute=(not meta.verbose))

    # Compute the medians and uncertainties
    fit_params = []
    upper_errs = []
    lower_errs = []
    for i in range(ndims):
        q = np.percentile(samples[:, i], [16, 50, 84])
        lower_errs.append(q[0])
        fit_params.append(q[1])
        upper_errs.append(q[2])
    fit_params = np.array(fit_params)
    upper_errs = np.array(upper_errs)-fit_params
    lower_errs = fit_params-np.array(lower_errs)

    model.update(fit_params, freenames)
    if "scatter_ppm" in freenames:
        ind = [i for i in np.arange(len(freenames)) if freenames[i][0:11] == "scatter_ppm"]
        for chan in range(len(ind)):
            lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = fit_params[ind[chan]] * 1e-6
    elif "scatter_mult" in freenames:
        ind = [i for i in np.arange(len(freenames)) if freenames[i][0:12] == "scatter_mult"]
        for chan in range(len(ind)):
            lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = fit_params[ind[chan]] * lc.unc[chan*lc.time.size:(chan+1)*lc.time.size]
    else:
        lc.unc_fit = lc.unc

    # Save the fit ASAP so plotting errors don't make you lose everything
    save_fit(meta, lc, model, 'dynesty', fit_params, freenames, samples, upper_errs=upper_errs, lower_errs=lower_errs)

    end_lnprob = lnprob(fit_params, lc, model, prior1, prior2, priortype, freenames)
    log.writelog(f'Ending lnprob: {end_lnprob}', mute=(not meta.verbose))

    # plot using corner.py
    if meta.isplots_S5 >= 5:
        plots.plot_corner(samples, lc, meta, freenames, fitter='dynesty')

    # Make a new model instance
    best_model = copy.copy(model)
    best_model.components[0].update(fit_params, freenames)

    #Plot GP fit + components
    if model.GP and meta.isplots_S5 >= 1:
        plots.plot_GP_components(lc, model, meta, fitter='dynesty')

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter='dynesty')

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
            scatter_ppm = fit_params[i] * np.ma.median(lc.unc[chan*lc.time.size:(chan+1)*lc.time.size]) * 1e6
            scatter_ppm_upper = upper_errs[i] * np.ma.median(lc.unc[chan*lc.time.size:(chan+1)*lc.time.size]) * 1e6
            scatter_ppm_lower = lower_errs[i] * np.ma.median(lc.unc[chan*lc.time.size:(chan+1)*lc.time.size]) * 1e6
            log.writelog('{0}: {1} (+{2}, -{3}); {4} (+{5}, -{6}) ppm'.format(freenames[i], fit_params[i], upper_errs[i], lower_errs[i], scatter_ppm, scatter_ppm_upper, scatter_ppm_lower))
        else:
            log.writelog('{0}: {1} (+{2}, -{3})'.format(freenames[i], fit_params[i], upper_errs[i], lower_errs[i]))
    log.writelog('')

    # Plot Allan plot
    if meta.isplots_S5 >= 3:
        plots.plot_rms(lc, model, meta, fitter='dynesty')

    # Plot residuals distribution
    if meta.isplots_S5 >= 3:
        plots.plot_res_distr(lc, model, meta, fitter='dynesty')

    best_model.__setattr__('chi2red',chi2red)
    best_model.__setattr__('fit_params',fit_params)

    return best_model

def lmfitter(lc, model, meta, log, **kwargs):
    """Perform a fit using lmfit.

    Parameters
    ----------
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    meta: MetaClass
        The metadata object
    log: logedit.Logedit
        The open log in which notes from this step can be added.
    **kwargs:
        Arbitrary keyword arguments.

    Returns
    -------
    best_model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model after fitting

    Notes
    -----
    History:

    - December 29, 2021 Taylor Bell
        Updated documentation. Reduced repeated code.
    - February 28-March 1, 2022 Caroline Piaulet
        Adding scatter_ppm parameter. 
    """
        #TODO: Do something so that duplicate param names can all be handled (e.g. two Polynomail models with c0). Perhaps append something to the parameter name like c0_1 and c0_2?)

    # Group the different variable types
    param_list, freenames, indep_vars = group_variables_lmfit(model)

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

    log.writelog(result.fit_report(), mute=(not meta.verbose))

    # Get the best fit params
    fit_params = result.__dict__['params']
    new_params = [(fit_params.get(i).name, fit_params.get(i).value,
                   fit_params.get(i).vary, fit_params.get(i).min,
                   fit_params.get(i).max) for i in fit_params]

    model.update(fit_params, freenames)
    if "scatter_ppm" in freenames:
        ind = [i for i in np.arange(len(freenames)) if freenames[i][0:11] == "scatter_ppm"]
        for chan in range(len(ind)):
            lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = fit_params[ind[chan]] * 1e-6
    elif "scatter_mult" in freenames:
        ind = [i for i in np.arange(len(freenames)) if freenames[i][0:12] == "scatter_mult"]
        for chan in range(len(ind)):
            lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = fit_params[ind[chan]] * lc.unc[chan*lc.time.size:(chan+1)*lc.time.size]
    else:
        lc.unc_fit = lc.unc

    # Save the fit ASAP
    save_fit(meta, lc, model, 'lmfitter', fit_params, freenames)

    # Create new model with best fit parameters
    best_model = copy.copy(model)
    best_model.components[0].update(fit_params, freenames)

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter='lmfitter')

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, log, model, meta, freenames)

    # Plot Allan plot
    if meta.isplots_S5 >= 3:
        plots.plot_rms(lc, model, meta, fitter='lmfitter')

    best_model.__setattr__('chi2red',chi2red)
    best_model.__setattr__('fit_params',fit_params)

    return best_model

def group_variables(model):
    """Group variables into fitted and frozen.

    Parameters
    ----------
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit

    Returns
    -------
    freenames: np.array
        The names of fitted variables.
    freepars: np.array
        The fitted variables.
    prior1: np.array
        The lower bound for constrained variables with uniform/log uniform priors, or mean for constrained variables with Gaussian priors.
    prior2: np.array
        The upper bound for constrained variables with uniform/log uniform priors, or mean for constrained variables with Gaussian priors.
    priortype: np.array
        Keywords indicating the type of prior for each free parameter.
    indep_vars: dict
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
    all_params = []
    alreadylist = []
    for chan in np.arange(model.components[0].nchan):
        temp=model.components[0].longparamlist[chan]
        for par in list(model.components[0].parameters.dict.items()):
            if par[0] in temp:
                if not all_params:
                    all_params.append(par)
                    alreadylist.append(par[0])
                if par[0] not in alreadylist:
                    all_params.append(par)
                    alreadylist.append(par[0])

    # Group the different variable types
    freenames = []
    freepars = []
    prior1 = []
    prior2 = []
    priortype = []
    indep_vars = {}
    for ii, item in enumerate(all_params):
        name, param = item
        #param = list(param)
        if (param[1] == 'free') or (param[1] == 'shared'):
            freenames.append(name)
            freepars.append(param[0])
            if len(param) == 5: #If prior is specified.
                prior1.append(param[2])
                prior2.append(param[3])
                priortype.append(param[4])
            elif (len(param) > 3) & (len(param) < 5): #If prior bounds are specified but not the prior type
                raise IndexError("If you want to specify prior parameters, you must also specify the prior type: 'U', 'LU', or 'N'.")
            else: #If no prior is specified, assume uniform prior with infinite bounds.
                prior1.append(-np.inf)
                prior2.append(np.inf)
                priortype.append('U')
        elif param[1] == 'independent':
            indep_vars[name] = param[0]
    freenames = np.array(freenames)
    freepars = np.array(freepars)
    prior1 = np.array(prior1)
    prior2 = np.array(prior2)
    priortype = np.array(priortype)

    return freenames, freepars, prior1, prior2, priortype, indep_vars

def group_variables_lmfit(model):
    """Group variables into fitted and frozen for lmfit fitter.

    Parameters
    ----------
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit

    Returns
    -------
    paramlist: list
        The fitted variables.
    freenames: np.array
        The names of fitted variables.
    indep_vars: dict
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
    freenames = []
    indep_vars = {}
    for param in all_params:
        param = list(param)
        if param[1][1] == 'free':
            freenames.append(param[0])
            param[1][1] = True
            param_list.append(tuple(param))
        elif param[1][1] == 'fixed':
            param[1][1] = False
            param_list.append(tuple(param))
        else:
            indep_vars[param[0]] = param[1]
    freenames = np.array(freenames)

    return param_list, freenames, indep_vars

def load_old_fitparams(meta, log, channel, freenames):
    if meta.sharedp:
        channel_key = 'shared'
    else:
        channel_key = f'ch{channel}'

    fname = os.path.join(meta.topdir, *meta.old_fitparams.split(os.sep))
    fitted_values = pd.read_csv(fname, escapechar='#', skipinitialspace=True)
    full_keys = np.array(fitted_values.keys())
    full_keys[0] = full_keys[0][1:] # Remove the " " from the start of the first key

    if np.all(full_keys!=freenames):
        log.writelog('Old fit does not have the same fitted parameters and cannot be used to initialize the new fit.\n'+
                     'The old fit included:\n['+','.join(full_keys)+']\n'+
                     'The new fit included:\n['+','.join(freenames)+']', mute=True)
        raise AssertionError('Old fit does not have the same fitted parameters and cannot be used to initialize the new fit.\n'+
                             'The old fit included:\n['+','.join(full_keys)+']\n'+
                             'The new fit included:\n['+','.join(freenames)+']')

    return np.array(fitted_values)[0]

def save_fit(meta, lc, model, fitter, fit_params, freenames, samples=[], upper_errs=[], lower_errs=[]):
    # Save the fitted parameters and their uncertainties (if possible)
    if lc.share:
        fname = f'S5_{fitter}_fitparams_shared'
    else:
        fname = f'S5_{fitter}_fitparams_ch{str(lc.channel).zfill(len(str(lc.nchannel)))}'
    data = fit_params.reshape(1,-1)
    if len(upper_errs)!=0 and len(lower_errs)!=0:
        data = np.append(data, -lower_errs.reshape(1,-1), axis=0)
        data = np.append(data, upper_errs.reshape(1,-1), axis=0)
    np.savetxt(meta.outputdir+fname+'.csv', data, header=','.join(freenames), delimiter=',')

    # Save the chain from the sampler (if a chain was provided)
    if len(samples)!=0:
        if lc.share:
            fname = f'S5_{fitter}_samples_shared'
        else:
            fname = f'S5_{fitter}_samples_ch{str(lc.channel).zfill(len(str(lc.nchannel)))}'
        with h5py.File(meta.outputdir+fname+'.h5', 'w') as hf:
            hf.create_dataset("samples",  data=samples)

    # Save the S5 outputs in a human readable ecsv file
    event_ap_bg = meta.eventlabel + "_ap" + str(meta.spec_hw) + '_bg' + str(meta.bg_hw)
    if lc.share:
        channel_tag = '_shared'
    else:
        channel_tag = f'_ch{str(lc.channel).zfill(len(str(lc.nchannel)))}'
    meta.tab_filename_s5 = meta.outputdir + 'S5_' + event_ap_bg + "_Table_Save"+channel_tag+".txt"
    wavelengths = np.mean(np.append(meta.wave_low.reshape(1,-1), meta.wave_hi.reshape(1,-1), axis=0), axis=0)
    wave_errs = (meta.wave_hi-meta.wave_low)/2
    model_lc = model.eval()
    residuals = lc.flux-model_lc
    astropytable.savetable_S5(meta.tab_filename_s5, meta.time, wavelengths[lc.fitted_channels], wave_errs[lc.fitted_channels], lc.flux, lc.unc_fit, model_lc, residuals)

    return
