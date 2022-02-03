import numpy as np
import copy

import lmfit
import emcee

from dynesty import NestedSampler
from dynesty.utils import resample_equal

from ..lib import lsq
from .parameters import Parameters
from .likelihood import computeRedChiSq, lnprob, ptform
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

    log.writelog('\nLSQ RESULTS:')
    for freenames_i, fit_params_i in zip(freenames, fit_params):
        log.writelog('{0}: {1}'.format(freenames_i, fit_params_i))
    log.writelog('')

    # Plot Allan plot
    if meta.isplots_S5 >= 3:
        plots.plot_rms(lc, model, meta, fitter='lsq')

    best_model.__setattr__('chi2red',chi2red)
    best_model.__setattr__('fit_params',fit_params)

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
    """
    log.writelog('\nCalling lsqfitter first...')
    lsq_sol = lsqfitter(lc, model, meta, log, calling_function='emcee_lsq', **kwargs)

    # SCALE UNCERTAINTIES WITH REDUCED CHI2
    if meta.rescale_err:
        lc.unc *= np.sqrt(lsq_sol.chi2red)

    # Group the different variable types
    freenames, freepars, pmin, pmax, indep_vars = group_variables(model)

    if lsq_sol.cov_mat is not None:
        step_size = np.diag(lsq_sol.cov_mat)
    else:
        # Sometimes the lsq fitter won't converge and will give None as the covariance matrix
        # In that case, we need to establish the step size in another way. A fractional step compared
        # to the value can work okay, but it may fail if the step size is larger than the bounds
        # which is not uncommon for precisely known values like t0 and period
        log.writelog('No covariance matrix from LSQ - falling back on a 0.1% step size')
        step_size = 0.001*freepars
    ndim = len(step_size)
    nwalkers = meta.run_nwalkers
    run_nsteps = meta.run_nsteps
    burn_in = meta.run_nburn

    pos = np.array([freepars + np.array(step_size)*np.random.randn(ndim) for i in range(nwalkers)])
    in_range = np.array([all((pmin <= ii) & (ii <= pmax)) for ii in pos])
    n_loops = 0
    while not np.all(in_range) and n_loops<meta.max_pos_iters:
        n_loops += 1
        pos = pos[in_range]
        step_size /= 2 # Make the proposal size a bit smaller to reduce odds of rejection
        pos = np.append(pos, np.array([freepars + np.array(step_size)*np.random.randn(ndim) for i in range(nwalkers-len(pos))]))
        in_range = np.array([all((pmin <= ii) & (ii <= pmax)) for ii in pos])
    if not np.any(in_range):
        raise AssertionError('Failed to initialize any walkers within the set bounds for all parameters!\n'+
                             'Check your stating position, decrease your step size, or increase the bounds on your parameters')
    elif not np.all(in_range):
        log.writelog('Warning: Failed to initialize all walkers within the set bounds for all parameters!')
        log.writelog('Using {} walkers instead of the initially requested {} walkers'.format(np.sum(in_range), nwalkers))
        nwalkers = np.sum(in_range)

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

    best_model.__setattr__('chi2red',chi2red)
    best_model.__setattr__('fit_params',fit_params)

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
    """
    log.writelog('\nCalling lsqfitter first...')
    # RUN LEAST SQUARES
    lsq_sol = lsqfitter(lc, model, meta, log, calling_function='dynesty_lsq', **kwargs)

    # SCALE UNCERTAINTIES WITH REDUCED CHI2
    if meta.rescale_err:
        lc.unc *= np.sqrt(lsq_sol.chi2red)

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

    # the prior_transform function for dynesty requires there only be one argument
    ptform_lambda = lambda theta: ptform(theta, pmin, pmax)

    log.writelog('Running dynesty...')
    sampler = NestedSampler(lnprob, ptform_lambda, ndims,
                            bound=bound, sample=sample, nlive=nlive, logl_args = l_args)
    sampler.run_nested(dlogz=tol, print_progress=True)  # output progress bar
    res = sampler.results  # get results dictionary from sampler

    logZdynesty = res.logz[-1]  # value of logZ
    logZerrdynesty = res.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

    if meta.run_verbose:
        log.writelog('')
        log.writelog(res.summary())

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
    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter='dynesty')

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, model, meta, freenames)

    # Plot Allan plot
    if meta.isplots_S5 >= 3:
        plots.plot_rms(lc, model, meta, fitter='dynesty')

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
    """
    # all_params = [i for j in [model.components[n].parameters.dict.items()
    #               for n in range(len(model.components))] for i in j]
    all_params = list(model.components[0].parameters.dict.items())

    # Group the different variable types
    freenames = []
    freepars = []
    pmin = []
    pmax = []
    indep_vars = {}
    for ii, item in enumerate(all_params):
        name, param = item
        #param = list(param)
        if param[1] == 'free':
            freenames.append(name)
            freepars.append(param[0])
            if len(param) > 3:
                pmin.append(param[2])
                pmax.append(param[3])
            else:
                pmin.append(-np.inf)
                pmax.append(np.inf)
        # elif param[1] == 'fixed':
        #     pinitial.append(param[0])
        #     pmin.append(param[0])
        #     pmax.append(param[0])
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
