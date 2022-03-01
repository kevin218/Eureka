import numpy as np
import copy
from io import StringIO
import sys

import lmfit
import emcee

from dynesty import NestedSampler
from dynesty.utils import resample_equal

from ..lib import lsq
from .parameters import Parameters
from .likelihood import computeRedChiSq, lnprob, ln_like, ptform
from . import plots_s5 as plots

#FINDME: Keep reload statements for easy testing
from importlib import reload
reload(lsq)
reload(plots)

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
    freenames, freepars, pmin, pmax, indep_vars = group_variables(model)
    
    results = lsq.minimize(lc, model, freepars, pmin, pmax, freenames, indep_vars)
    
    if meta.run_verbose:
        log.writelog("\nVerbose lsq results: {}\n".format(results))
    else:
        log.writelog("Success?: {}".format(results.success))
        log.writelog(results.message)

    # Get the best fit params
    fit_params = results.x
    
    # Make a new model instance
    best_model = copy.copy(model)
    best_model.components[0].update(fit_params, freenames)

    model.update(fit_params, freenames)
    if "scatter_ppm" in freenames:
        ind = [i for i in np.arange(len(freenames)) if freenames[i][0:11] == "scatter_ppm"]
        lc.unc_fit = np.ones_like(lc.flux) * fit_params[ind[0]] * 1e-6        
        if len(ind)>1:
            for chan in np.arange(lc.flux.size//lc.time.size):
                lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = fit_params[ind[chan]] * 1e-6
    
    # Save the covariance matrix in case it's needed to estimate step size for a sampler
    model_lc = model.eval()

    residuals = (lc.flux - model_lc)
    # FINDME
    # Commented out for now because op.least_squares() doesn't provide covariance matrix
    # Need to compute using Jacobian matrix instead (hess_inv = (J.T J)^{-1})
    # if results[1] is not None:
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
    chi2red = computeRedChiSq(lc, model, meta, freenames)
    
    print('\nLSQ RESULTS:')
    for freenames_i, fit_params_i in zip(freenames, fit_params):
        log.writelog('{0}: {1}'.format(freenames_i, fit_params_i))
    log.writelog('')

    # Plot Allan plot
    if meta.isplots_S5 >= 3 and calling_function=='lsq':
        # This plot is only really useful if you're actually using the lsq fitter, otherwise don't make it
        plots.plot_rms(lc, model, meta, fitter=calling_function)

    best_model.__setattr__('chi2red',chi2red)
    best_model.__setattr__('fit_params',fit_params)

    save_fit(meta, lc, calling_function, fit_params, freenames)

    return best_model

def demcfitter(lc, model, meta, log, **kwargs):
    """Perform sampling using Differential Evolution Markov Chain.

    This is an empty placeholder function to be filled later.

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
    - February 28-March 1, 2022 Caroline Piaulet
        Adding scatter_ppm parameter. Made robust statements to avoid initial 
        state issues.
    """
    if not hasattr(meta, 'lsq_first') or meta.lsq_first:
        # Only call lsq fitter first if asked or lsq_first option wasn't passed (allowing backwards compatibility)
        log.writelog('\nCalling lsqfitter first...')
        # RUN LEAST SQUARES
        lsq_sol = lsqfitter(lc, model, meta, log, calling_function='emcee_lsq', **kwargs)

        # SCALE UNCERTAINTIES WITH REDUCED CHI2
        if meta.rescale_err:
            lc.unc *= np.sqrt(lsq_sol.chi2red)
    else:
        lsq_sol = None
    
    # Group the different variable types
    freenames, freepars, pmin, pmax, indep_vars = group_variables(model)
    
    if lsq_sol is not None and lsq_sol.cov_mat is not None:
        step_size = np.diag(lsq_sol.cov_mat)
        ind_zero = np.where(step_size==0.)[0]
        if len(ind_zero):
            step_size[ind_zero] = 0.001*np.abs(freepars[ind_zero])
    else:
        # Sometimes the lsq fitter won't converge and will give None as the covariance matrix
        # In that case, we need to establish the step size in another way. A fractional step compared
        # to the value can work okay, but it may fail if the step size is larger than the bounds
        # which is not uncommon for precisely known values like t0 and period
        log.writelog('No covariance matrix from LSQ - falling back on a 0.1% step size')
        step_size = 0.001*np.abs(freepars)
    ndim = len(step_size)
    nwalkers = meta.run_nwalkers
    run_nsteps = meta.run_nsteps
    burn_in = meta.run_nburn

    # make it robust to lsq hitting the upper or lower bound of the param space
    ind_max = np.where(freepars - pmax == 0.)
    ind_min = np.where(freepars - pmin == 0.)
    pmid = (pmax+pmin)/2.
    if len(ind_max[0]):
        log.writelog('Warning: >=1 params hit the upper bound in the lsq fit. Setting to the middle of the interval.')
        freepars[ind_max] = pmid[ind_max]
    if len(ind_min[0]):
        log.writelog('Warning: >=1 params hit the lower bound in the lsq fit. Setting to the middle of the interval.')
        freepars[ind_min] = pmid[ind_min]
    
    ind_zero_step = np.where(step_size==0.)
    if len(ind_zero_step[0]):
        log.writelog('Warning: >=1 params would have a zero step. changing to 0.001 * prior range')
        step_size[ind_zero_step] = 0.001*(pmax[ind_zero_step] - pmin[ind_zero_step])
        
    
    pos = np.array([freepars + np.array(step_size)*np.random.randn(ndim) for i in range(nwalkers)])
    in_range = np.array([all((pmin <= ii) & (ii <= pmax)) for ii in pos])
    if not np.all(in_range):
        log.writelog('Not all walkers were initialized within the priors, using a smaller proposal distribution')
        pos = pos[in_range]
        # Make sure the step size is well within the limits
        step_size_options = np.append(step_size.reshape(-1,1), np.abs(np.append((pmax-freepars).reshape(-1,1)/10, (freepars-pmin).reshape(-1,1)/10, axis=1)), axis=1)
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
        in_range = np.array([all((pmin <= ii) & (ii <= pmax)) for ii in pos])
    if not np.any(in_range):
        raise AssertionError('Failed to initialize any walkers within the set bounds for all parameters!\n'+
                             'Check your stating position, decrease your step size, or increase the bounds on your parameters')
    elif not np.all(in_range):
        log.writelog('Warning: Failed to initialize all walkers within the set bounds for all parameters!')
        log.writelog('Using {} walkers instead of the initially requested {} walkers'.format(np.sum(in_range), nwalkers))
        pos = pos[in_range]
        nwalkers = pos.shape[0]

    log.writelog('Running emcee...')
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(lc, model, pmin, pmax, freenames))
    sampler.run_mcmc(pos, run_nsteps, progress=True)
    samples = sampler.chain[:, burn_in::1, :].reshape((-1, ndim))
    if meta.isplots_S5 >= 5:
        plots.plot_corner(samples, lc, meta, freenames, fitter='emcee')

    medians = []
    for i in range(len(step_size)):
            q = np.percentile(samples[:, i], [16, 50, 84])
            medians.append(q[1])
    fit_params = np.array(medians)

    # Make a new model instance
    best_model = copy.copy(model)
    best_model.components[0].update(fit_params, freenames)

    model.update(fit_params, freenames)
    if "scatter_ppm" in freenames:
        ind = [i for i in np.arange(len(freenames)) if freenames[i][0:11] == "scatter_ppm"]
        lc.unc_fit = np.ones_like(lc.flux) * fit_params[ind[0]] * 1e-6        
        if len(ind)>1:
            for chan in np.arange(lc.flux.size//lc.time.size):
                lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = fit_params[ind[chan]] * 1e-6

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter='emcee')

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, model, meta, freenames)

    log.writelog('\nEMCEE RESULTS:')
    for freenames_i, fit_params_i in zip(freenames, fit_params):
        log.writelog('{0}: {1}'.format(freenames_i, fit_params_i))
    log.writelog('')
    
    # Plot Allan plot
    if meta.isplots_S5 >= 3:
        plots.plot_rms(lc, model, meta, fitter='emcee')
        
    # Plot residuals distribution
    if meta.isplots_S5 >= 3:
        plots.plot_res_distr(lc, model, meta, fitter='emcee')

    best_model.__setattr__('chi2red',chi2red)
    best_model.__setattr__('fit_params',fit_params)
    
    save_fit(meta, lc, 'emcee', fit_params, freenames, samples)

    return best_model

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
    - February 28-March 1, 2022 Caroline Piaulet
        Adding scatter_ppm parameter. 
    """
    # Group the different variable types
    freenames, freepars, pmin, pmax, indep_vars = group_variables(model)

    # DYNESTY
    nlive = meta.run_nlive # number of live points
    bound = meta.run_bound  # use MutliNest algorithm for bounds
    ndims = len(freepars)  # two parameters
    sample = meta.run_sample  # uniform sampling
    tol = meta.run_tol  # the stopping criterion

    # START DYNESTY
    l_args = [lc, model, pmin, pmax, freenames]

    log.writelog('Running dynesty...')

    min_nlive = int(np.ceil(ndims*(ndims+1)//2))
    if nlive < min_nlive:
        log.writelog(f'**** WARNING: You should set run_nlive to at least {min_nlive} ****')

    sampler = NestedSampler(ln_like, ptform, ndims,
                            bound=bound, sample=sample, nlive=nlive, logl_args = l_args,
                            ptform_args=[pmin, pmax])
    sampler.run_nested(dlogz=tol, print_progress=True)  # output progress bar
    res = sampler.results  # get results dictionary from sampler

    logZdynesty = res.logz[-1]  # value of logZ
    logZerrdynesty = res.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

    if meta.run_verbose:
        log.writelog('')
        # Need to temporarily redirect output since res.summar() prints rather than returns a string
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        res.summary()
        sys.stdout = old_stdout
        log.writelog(mystdout.getvalue())

    # get function that resamples from the nested samples to give sampler with equal weight
    # draw posterior samples
    weights = np.exp(res['logwt'] - res['logz'][-1])
    samples = resample_equal(res.samples, weights)
    if meta.run_verbose:
        log.writelog('Number of posterior samples is {}'.format(len(samples)))

    # plot using corner.py
    if meta.isplots_S5 >= 5:
        plots.plot_corner(samples, lc, meta, freenames, fitter='dynesty')

    medians = []
    for i in range(len(freenames)):
            q = np.percentile(samples[:, i], [16, 50, 84])
            medians.append(q[1])
    fit_params = np.array(medians)

    # Make a new model instance
    best_model = copy.copy(model)
    best_model.components[0].update(fit_params, freenames)

    model.update(fit_params, freenames)
    if "scatter_ppm" in freenames:
        ind = [i for i in np.arange(len(freenames)) if freenames[i][0:11] == "scatter_ppm"]
        lc.unc_fit = np.ones_like(lc.flux) * fit_params[ind[0]] * 1e-6        
        if len(ind)>1:
            for chan in np.arange(lc.flux.size//lc.time.size):
                lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = fit_params[ind[chan]] * 1e-6


    model_lc = model.eval()
    residuals = (lc.flux - model_lc) #/ lc.unc

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter='dynesty')

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, model, meta, freenames)

    log.writelog('\nDYNESTY RESULTS:')
    for freenames_i, fit_params_i in zip(freenames, fit_params):
        log.writelog('{0}: {1}'.format(freenames_i, fit_params_i))
    log.writelog('')

    # Plot Allan plot
    if meta.isplots_S5 >= 3:
        plots.plot_rms(lc, model, meta, fitter='dynesty')

    best_model.__setattr__('chi2red',chi2red)
    best_model.__setattr__('fit_params',fit_params)

    save_fit(meta, lc, 'dynesty', fit_params, freenames, samples)

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

    if meta.run_verbose:
        log.writelog(result.fit_report())

    # Get the best fit params
    fit_params = result.__dict__['params']
    new_params = [(fit_params.get(i).name, fit_params.get(i).value,
                   fit_params.get(i).vary, fit_params.get(i).min,
                   fit_params.get(i).max) for i in fit_params]

    # Create new model with best fit parameters
    params = Parameters()
    # Store each as an attribute
    for param in new_params:
        setattr(params, param[0], param[1:])

    # Make a new model instance
    best_model = copy.copy(model)
    best_model.components[0].update(fit_params, freenames)
    # best_model.parameters = params
    # best_model.name = ', '.join(['{}:{}'.format(k, round(v[0], 2)) for k, v in params.dict.items()])

    model.update(fit_params, freenames)
    if "scatter_ppm" in freenames:
        ind = [i for i in np.arange(len(freenames)) if freenames[i][0:11] == "scatter_ppm"]
        lc.unc_fit = np.ones_like(lc.flux) * fit_params[ind[0]] * 1e-6        
        if len(ind)>1:
            for chan in np.arange(lc.flux.size//lc.time.size):
                lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = fit_params[ind[chan]] * 1e-6

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter='lmfitter')

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, model, meta, freenames)

    # Plot Allan plot
    if meta.isplots_S5 >= 3:
        plots.plot_rms(lc, model, meta, fitter='lmfitter')

    best_model.__setattr__('chi2red',chi2red)
    best_model.__setattr__('fit_params',fit_params)

    save_fit(meta, lc, 'lmfitter', fit_params, freenames)

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
    pmin: np.array
        The lower bound for constrained variables.
    pmax: np.array
        The upper bound for constrained variables.
    indep_vars: dict
        The frozen variables.

    Notes
    -----
    History:

    - December 29, 2021 Taylor Bell
        Moved code to separate function to reduce repeated code.
    - January 11, 2022 Megan Mansfield
        Added ability to have shared parameters
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
    pmin = []
    pmax = []
    indep_vars = {}
    for ii, item in enumerate(all_params):
        name, param = item
        #param = list(param)
        if (param[1] == 'free') or (param[1] == 'shared'):
            freenames.append(name)
            freepars.append(param[0])
            if len(param) > 3:
                pmin.append(param[2])
                pmax.append(param[3])
            else:
                pmin.append(-np.inf)
                pmax.append(np.inf)
        elif param[1] == 'independent':
            indep_vars[name] = param[0]
    freenames = np.array(freenames)
    freepars = np.array(freepars)
    pmin = np.array(pmin)
    pmax = np.array(pmax)

    return freenames, freepars, pmin, pmax, indep_vars

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

def save_fit(meta, lc, fitter, fit_params, freenames, samples=[]):
    if lc.share:
        fname = f'S5_{fitter}_fitparams_shared.csv'
    else:
        fname = f'S5_{fitter}_fitparams_ch{lc.channel}.csv'
    np.savetxt(meta.outputdir+fname, fit_params.reshape(1,-1), header=','.join(freenames), delimiter=',')

    if len(samples)!=0:
        if lc.share:
            fname = f'S5_{fitter}_samples_shared.csv'
        else:
            fname = f'S5_{fitter}_samples_ch{lc.channel}.csv'
        np.savetxt(meta.outputdir+fname, samples, header=','.join(freenames), delimiter=',')
    
    return