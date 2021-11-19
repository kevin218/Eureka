"""Functions used to fit models to light curve data

Author: Joe Filippazzo
Email: jfilippazzo@stsci.edu
"""
import numpy as np
import lmfit
import copy
from importlib import reload
from ..lib import lsq
reload(lsq)

from .parameters import Parameters

import matplotlib.pyplot as plt
import scipy.optimize as op
import emcee
import corner

import dynesty
from scipy.stats import norm
import matplotlib as mpl
from dynesty import NestedSampler
from dynesty.utils import resample_equal

def lsqfitter(lc, model, **kwargs):
    """Perform least-squares fit

    Parameters
    ----------
    data: sequence
        The observational data
    model: ExoCTK.lightcurve_fitting.models.Model
        The model to fit
    uncertainty: np.ndarray (optional)
        The uncertainty on the (same shape) data
    method: str
        The name of the method to use
    name: str
        A name for the best fit model
    verbose: bool
        Print some stuff

    Returns
    -------
    lsq.Model.fit.fit_report
        The results of the fit
    """
    # Concatenate the lists of parameters
    #all_keys   = [i for j in [model.components[n].parameters.dict.keys()
    #              for n in range(len(model.components))] for i in j]
    all_params = [i for j in [model.components[n].parameters.dict.items()
                  for n in range(len(model.components))] for i in j]
    #print(all_params)
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
    freepars = np.array(freepars)
    pmin = np.array(pmin)
    pmax = np.array(pmax)




    # Set the uncertainty
    if lc.unc is None:
        lc.unc = np.sqrt(lc.flux)

    #lc.etc = {}
    #lc.etc['time'] = lc.time

    results = lsq.minimize(lc, model, freepars, pmin, pmax, freenames, indep_vars)

    if kwargs['run_verbose'][0]:
        print(results)

    # Get the best fit params
    fit_params = results[0]
    # new_params = [(fit_params.get(i).name, fit_params.get(i).value,
    #                fit_params.get(i).vary, fit_params.get(i).min,
    #                fit_params.get(i).max) for i in fit_params]

    # Create new model with best fit parameters
    # params = Parameters()

    # Try to store each as an attribute
    # for param in new_params:
    #     setattr(params, param[0], param[1:])

    # Make a new model instance
    best_model = copy.copy(model)
    best_model.components[0].update(fit_params, freenames)


    model.update(fit_params, freenames)
    model_lc = model.eval()
    residuals = (lc.flux - model_lc) #/ lc.unc
    if kwargs['run_show_plot'][0]:
        model.plot(time=lc.time, draw=True)
        print()

        fig, ax = plt.subplots(2,1)
        ax[0].errorbar(lc.time, lc.flux, yerr=lc.unc, fmt='.')
        ax[0].plot(lc.time, model_lc, zorder = 10)

        
        ax[1].errorbar(lc.time, residuals, yerr=lc.unc, fmt='.')
        plt.savefig('lc_lsq.png', dpi=300)
        plt.show()

    chi2 = np.sum((residuals / lc.unc) ** 2)
    chi2red = chi2 / (len(lc.unc) - len(freenames))

    if kwargs['run_verbose'][0]:
        print('red. Chi2: ', chi2red)
    # best_model.parameters = params
    # best_model.name = ', '.join(['{}:{}'.format(k, round(v[0], 2)) for k, v in params.dict.items()])

        print('\nLSQ RESULTS:\n')
        for freenames_i, fit_params_i in zip(freenames, fit_params):
            print('{0}: {1}'.format(freenames_i, fit_params_i))
        print('\n')
    if kwargs['run_show_plot'][0]:
        rmsplot(lc, model_lc, figname='allanplot_lsq.png')

    return best_model#, chi2red, fit_params

def demcfitter(time, data, model, uncertainty=None, **kwargs):
    """Use Differential Evolution Markov Chain

    Parameters
    ----------
    data: sequence
        The observational data
    model: ExoCTK.lightcurve_fitting.models.Model
        The model to fit
    uncertainty: np.ndarray (optional)
        The uncertainty on the (same shape) data
    method: str
        The name of the method to use
    name: str
        A name for the best fit model
    verbose: bool
        Print some stuff

    Returns
    -------
    demc.Model.fit.fit_report
        The results of the fit
    """
    best_model = None
    return best_model



def emceefitter(lc, model, **kwargs):
    """Perform sampling using emcee

    Parameters
    ----------
    data: sequence
        The observational data
    model: ExoCTK.lightcurve_fitting.models.Model
        The model to fit
    uncertainty: np.ndarray (optional)
        The uncertainty on the (same shape) data
    method: str
        The name of the method to use
    name: str
        A name for the best fit model
    verbose: bool
        Print some stuff

    Returns
    -------
    lsq.Model.fit.fit_report
        The results of the fit
    """
    # Concatenate the lists of parameters
    #all_keys   = [i for j in [model.components[n].parameters.dict.keys()
    #              for n in range(len(model.components))] for i in j]

    lsq_sol = lsqfitter(lc, model, **kwargs)

    print(lsq_sol)

    lc.unc *= np.sqrt(lsq_sol[1]) #Getting an error here: 'CompositeModel' object is not subscriptable

    all_params = [i for j in [model.components[n].parameters.dict.items()
                  for n in range(len(model.components))] for i in j]

    #print(all_params)
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
    freepars = np.array(freepars)
    pmin = np.array(pmin)
    pmax = np.array(pmax)
    print('before update: ', freepars)

    model.update(lsq_sol[2], freenames)
    print('after update: ', freepars)

    # Set the uncertainty
    if lc.unc is None:
        lc.unc = np.sqrt(lc.flux)

    #lc.etc = {}
    #lc.etc['time'] = lc.time
    import time

    def ln_like(theta, lc, model, pmin, pmax):
        # params[ifreepars] = freepars
        ilow = np.where(theta < pmin)
        ihi = np.where(theta > pmax)
        theta[ilow] = pmin[ilow]
        theta[ihi] = pmax[ihi]
        model.update(theta, freenames)
        # model.time = time
        # model.components[0].time = time
        model_lc = model.eval()
        #model.plot(lc.time,draw=True)
        residuals = (lc.flux - model_lc) #/ lc.unc
        ln_like_val = (-0.5 * (np.sum((residuals / lc.unc) ** 2+ np.log(2.0 * np.pi * (lc.unc) ** 2))))
        if len(ilow[0]) + len(ihi[0]) > 0: ln_like_val = -np.inf
        return ln_like_val

    def lnprior(theta, pmin, pmax):
        lnprior_prob = 0.
        n = len(theta)
        for i in range(n):
            if np.logical_or(theta[i] < pmin[i],
                                 theta[i] > pmax[i]): lnprior_prob += - np.inf
        return lnprior_prob

    def lnprob(theta, lc, model, pmin, pmax):
        ln_like_val = ln_like(theta, lc, model, pmin, pmax)
        lp = lnprior(theta, pmin, pmax)
        return ln_like_val + lp

    step_size = np.array([5e-3, 5e-3, 1e-1, 5e-3, 1e-2, 1e-2])
    ndim = len(step_size)
    nwalkers = kwargs['run_nwalkers'][0]
    run_nsteps = kwargs['run_nsteps'][0]
    burn_in = kwargs['run_nburn'][0]

    pos = np.array([freepars + np.array(step_size)*np.random.randn(ndim) for i in range(nwalkers)])

    out_of_range = np.array([all((pmin <= ii) & (ii <= pmax)) for ii in pos])
    print(out_of_range)
    pos = pos[out_of_range]
    print(sum(out_of_range))
    nwalkers = sum(out_of_range)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(lc, model, pmin, pmax))

    sampler.run_mcmc(pos, run_nsteps, progress=True)
    samples = sampler.chain[:, burn_in::1, :].reshape((-1, ndim))
    if kwargs['run_show_plot'][0]:
        fig = corner.corner(samples, show_titles=True,quantiles=[0.16, 0.5, 0.84],title_fmt='.4', labels=freenames)
        figname = "corner_emcee.png"
        fig.savefig(figname, bbox_inches='tight', pad_inches=0.05, dpi=250)


    def quantile(x, q):
        return np.percentile(x, [100. * qi for qi in q])

    medians = []
    for i in range(len(step_size)):
            q = quantile(samples[:, i], [0.16, 0.5, 0.84])
            medians.append(q[1])
    fit_params = np.array(medians)

    # new_params = [(fit_params.get(i).name, fit_params.get(i).value,
    #                fit_params.get(i).vary, fit_params.get(i).min,
    #                fit_params.get(i).max) for i in fit_params]

    # Create new model with best fit parameters
    # params = Parameters()

    # Try to store each as an attribute
    # for param in new_params:
    #     setattr(params, param[0], param[1:])

    # Make a new model instance
    best_model = copy.copy(model)
    best_model.components[0].update(fit_params, freenames)
    # best_model.parameters = params
    # best_model.name = ', '.join(['{}:{}'.format(k, round(v[0], 2)) for k, v in params.dict.items()])

    import matplotlib.pyplot as plt

    model.update(fit_params, freenames)
    model_lc = model.eval()
    residuals = (lc.flux - model_lc) #/ lc.unc

    if kwargs['run_show_plot'][0]:
        model.plot(time=lc.time, draw=True)

        fig, ax = plt.subplots(2,1)
        ax[0].errorbar(lc.time, lc.flux, yerr=lc.unc, fmt='.')
        ax[0].plot(lc.time, model_lc, zorder = 10)


        ax[1].errorbar(lc.time, residuals, yerr=lc.unc, fmt='.')
        plt.savefig('lc_emcee.png', dpi=300)
        plt.show()

    ln_like_val = (-0.5 * (np.sum((residuals / lc.unc) ** 2 + np.log(2.0 * np.pi * (lc.unc) ** 2))))
    chi2 = np.sum((residuals / lc.unc) ** 2)
    chi2red = chi2 / (len(lc.unc))
    print(len(lc.unc))
    print(chi2)
    print(chi2red)
    # best_model.parameters = params
    # best_model.name = ', '.join(['{}:{}'.format(k, round(v[0], 2)) for k, v in params.dict.items()])

    if kwargs['run_verbose'][0]:
        for freenames_i, fit_params_i in zip(freenames, fit_params):
            print('{0}: {1}'.format(freenames_i, fit_params_i))

    if kwargs['run_show_plot'][0]:
        rmsplot(lc, model_lc, figname='allanplot_emcee.png')

    return best_model


def dynestyfitter(lc, model, **kwargs):
    """Perform sampling using emcee

    Parameters
    ----------
    data: sequence
        The observational data
    model: ExoCTK.lightcurve_fitting.models.Model
        The model to fit
    uncertainty: np.ndarray (optional)
        The uncertainty on the (same shape) data
    method: str
        The name of the method to use
    name: str
        A name for the best fit model
    verbose: bool
        Print some stuff

    Returns
    -------
    lsq.Model.fit.fit_report
        The results of the fit
    """
    # Concatenate the lists of parameters
    #all_keys   = [i for j in [model.components[n].parameters.dict.keys()
    #              for n in range(len(model.components))] for i in j]

    # RUN LEAST SQUARES
    lsq_sol = lsqfitter(lc, model, **kwargs)
    print(lsq_sol)
    # SCALE UNCERTAINTIES WITH REDUCED CHI2 TODO: put a flag for that into config
    lc.unc *= np.sqrt(lsq_sol[1])

    all_params = [i for j in [model.components[n].parameters.dict.items()
                  for n in range(len(model.components))] for i in j]

    #print(all_params)
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
    freepars = np.array(freepars)
    pmin = np.array(pmin)
    pmax = np.array(pmax)
    print('before update: ', freepars)

    # UPDATE MODEL PARAMETERS WITH LSQ BEST FIT
    model.update(lsq_sol[2], freenames)
    print('after update: ', freepars)

    # Set the uncertainty
    if lc.unc is None:
        lc.unc = np.sqrt(lc.flux)

    #lc.etc = {}
    #lc.etc['time'] = lc.time
    import time


    # DYNESTY

    #PRIOR TRANSFORMATION TODO: ADD GAUSSIAN PRIORS
    def transform_uniform(x, a, b):
        return a + (b - a) * x

    def transform_normal(x, mu, sigma):
        return norm.ppf(x, loc=mu, scale=sigma)

    def ptform(theta):
        p = np.zeros_like(theta)
        n = len(theta)
        for i in range(n):
            p[i] = transform_uniform(theta[i], pmin[i], pmax[i])
        return p


    def ln_like(theta, lc, model, pmin, pmax):
        # params[ifreepars] = freepars
        ilow = np.where(theta < pmin)
        ihi = np.where(theta > pmax)
        theta[ilow] = pmin[ilow]
        theta[ihi] = pmax[ihi]
        model.update(theta, freenames)
        # model.time = time
        # model.components[0].time = time
        model_lc = model.eval()
        #model.plot(lc.time,draw=True)
        residuals = (lc.flux - model_lc) #/ lc.unc
        ln_like_val = (-0.5 * (np.sum((residuals / lc.unc) ** 2+ np.log(2.0 * np.pi * (lc.unc) ** 2))))
        if len(ilow[0]) + len(ihi[0]) > 0: ln_like_val = -np.inf
        return ln_like_val

    def lnprior(theta, pmin, pmax):
        lnprior_prob = 0.
        n = len(theta)
        for i in range(n):
            if np.logical_or(theta[i] < pmin[i],
                                 theta[i] > pmax[i]): lnprior_prob += - np.inf
        return lnprior_prob

    def lnprob(theta, lc, model, pmin, pmax):
        ln_like_val = ln_like(theta, lc, model, pmin, pmax)
        lp = lnprior(theta, pmin, pmax)
        return ln_like_val + lp



    nlive = kwargs['run_nlive'][0] # number of live points
    bound = kwargs['run_bound'][0]  # use MutliNest algorithm for bounds
    ndims = kwargs['run_ndims'][0]  # two parameters
    sample = kwargs['run_sample'][0]  # uniform sampling
    tol = kwargs['run_tol'][0]  # the stopping criterion

    # START DYNESTY
    l_args = [lc, model, pmin, pmax]
    sampler = NestedSampler(lnprob, ptform, ndims,
                            bound=bound, sample=sample, nlive=nlive, logl_args = l_args)
    sampler.run_nested(dlogz=tol, print_progress=True)  # output progress bar
    res = sampler.results  # get results dictionary from sampler

    logZdynesty = res.logz[-1]  # value of logZ
    logZerrdynesty = res.logzerr[-1]  # estimate of the statistcal uncertainty on logZ

    if kwargs['run_verbose'][0]:
        print("log(Z) = {} ± {}".format(logZdynesty, logZerrdynesty))
        print(res.summary())

    # get function that resamples from the nested samples to give sampler with equal weight
    # draw posterior samples
    weights = np.exp(res['logwt'] - res['logz'][-1])
    samples_dynesty = resample_equal(res.samples, weights)
    if kwargs['run_verbose'][0]:
        print('Number of posterior samples is {}'.format(len(samples_dynesty)))

    # plot using corner.py
    if kwargs['run_show_plot'][0]:
        fig = corner.corner(samples_dynesty, labels=freenames, show_titles=True, quantiles=[0.16, 0.5, 0.84],title_fmt='.4')
        figname = "corner_dynesty.png"
        if kwargs['run_output'][0]:
            fig.savefig(figname, bbox_inches='tight', pad_inches=0.05, dpi=250)


    # PLOT MEDIAN OF THE SAMPLES
    def quantile(x, q):
        return np.percentile(x, [100. * qi for qi in q])

    medians = []
    for i in range(len(freenames)):
            q = quantile(samples_dynesty[:, i], [0.16, 0.5, 0.84])
            medians.append(q[1])
    fit_params = np.array(medians)

    # new_params = [(fit_params.get(i).name, fit_params.get(i).value,
    #                fit_params.get(i).vary, fit_params.get(i).min,
    #                fit_params.get(i).max) for i in fit_params]

    # Create new model with best fit parameters
    # params = Parameters()

    # Try to store each as an attribute
    # for param in new_params:
    #     setattr(params, param[0], param[1:])

    # Make a new model instance
    best_model = copy.copy(model)
    best_model.components[0].update(fit_params, freenames)
    # best_model.parameters = params
    # best_model.name = ', '.join(['{}:{}'.format(k, round(v[0], 2)) for k, v in params.dict.items()])

    model.update(fit_params, freenames)
    model_lc = model.eval()
    residuals = (lc.flux - model_lc) #/ lc.unc

    if kwargs['run_show_plot'][0]:
        model.plot(time=lc.time, draw=True)

        fig, ax = plt.subplots(2,1)
        ax[0].errorbar(lc.time, lc.flux, yerr=lc.unc, fmt='.')
        ax[0].plot(lc.time, model_lc, zorder = 10)


        ax[1].errorbar(lc.time, residuals, yerr=lc.unc, fmt='.')
        plt.savefig('lc_dynesty.png', dpi=300)
        plt.show()

    ln_like_val = (-0.5 * (np.sum((residuals / lc.unc) ** 2 + np.log(2.0 * np.pi * (lc.unc) ** 2))))
    chi2 = np.sum((residuals / lc.unc) ** 2)
    chi2red = chi2 / (len(lc.unc))
    if kwargs['run_verbose'][0]:
        print(len(lc.unc))
        print(chi2)
        print(chi2red)
    # best_model.parameters = params
    # best_model.name = ', '.join(['{}:{}'.format(k, round(v[0], 2)) for k, v in params.dict.items()])
    # Plot RMS vs. bin size looking for time-correlated noise

    print('\nDYNESTY RESULTS:\n')
    for freenames_i, fit_params_i in zip(freenames, fit_params):
        print('{0}: {1}'.format(freenames_i, fit_params_i))

    # PLOT ALLAN PLOT
    if kwargs['run_show_plot'][0]:
        rmsplot(lc,model_lc, figname='allanplot_dynesty.png')

    return best_model




def lmfitter(time, data, model, uncertainty=None, **kwargs):
    """Use lmfit

    Parameters
    ----------
    data: sequence
        The observational data
    model: ExoCTK.lightcurve_fitting.models.Model
        The model to fit
    uncertainty: np.ndarray (optional)
        The uncertainty on the (same shape) data
    method: str
        The name of the method to use
    name: str
        A name for the best fit model
    verbose: bool
        Print some stuff

    Returns
    -------
    lmfit.Model.fit.fit_report
        The results of the fit
    """
    # Initialize lmfit Params object
    initialParams = lmfit.Parameters()

    #TODO: Do something so that duplicate param names can all be handled (e.g. two Polynomail models with c0). Perhaps append something to the parameter name like c0_1 and c0_2?)

    # Concatenate the lists of parameters
    all_params = [i for j in [model.components[n].parameters.list
                  for n in range(len(model.components))] for i in j]

    # Group the different variable types
    param_list = []
    indep_vars = {}
    for param in all_params:
        param = list(param)
        if param[2] == 'free':
            param[2] = True
            param_list.append(tuple(param))
        elif param[2] == 'fixed':
            param[2] = False
            param_list.append(tuple(param))
        else:
            indep_vars[param[0]] = param[1]

    # Add the time as an independent variable
    indep_vars['time'] = time

    # Get values from input parameters.Parameters instances
    initialParams.add_many(*param_list)

    # Create the lightcurve model
    lcmodel = lmfit.Model(model.eval)
    lcmodel.independent_vars = indep_vars.keys()

    # Set the uncertainty
    if uncertainty is None:
        uncertainty = np.ones(len(data))

    # Fit light curve model to the simulated data
    result = lcmodel.fit(data, weights=1/uncertainty, params=initialParams,
                         **indep_vars, **kwargs)

    if kwargs['run_verbose'][0]:
        print(result.fit_report())

    # Get the best fit params
    fit_params = result.__dict__['params']
    new_params = [(fit_params.get(i).name, fit_params.get(i).value,
                   fit_params.get(i).vary, fit_params.get(i).min,
                   fit_params.get(i).max) for i in fit_params]

    # Create new model with best fit parameters
    params = Parameters()

    # Try to store each as an attribute
    for param in new_params:
        setattr(params, param[0], param[1:])

    # Make a new model instance
    best_model = copy.copy(model)
    best_model.parameters = params
    best_model.name = ', '.join(['{}:{}'.format(k, round(v[0], 2)) for k, v in params.dict.items()])

    return best_model




# COMPUTE ROOT-MEAN-SQUARE AND STANDARD ERROR OF DATA FOR VARIOUS BIN SIZES
def computeRMS(data, maxnbins=None, binstep=1, isrmserr=False):
    # data    = fit.normresiduals
    # maxnbin = maximum # of bins
    # binstep = Bin step size

    # bin data into multiple bin sizes
    npts = data.size
    if maxnbins is None:
        maxnbins = npts / 10.
    binsz = np.arange(1, maxnbins + binstep, step=binstep, dtype=int)
    nbins = np.zeros(binsz.size, dtype=int)
    rms = np.zeros(binsz.size)
    rmserr = np.zeros(binsz.size)
    for i in range(binsz.size):
        nbins[i] = int(np.floor(data.size / binsz[i]))
        bindata = np.zeros(nbins[i], dtype=float)
        # bin data
        # ADDED INTEGER CONVERSION, mh 01/21/12
        for j in range(nbins[i]):
            bindata[j] = data[j * binsz[i]:(j + 1) * binsz[i]].mean()
        # get rms
        rms[i] = np.sqrt(np.mean(bindata ** 2))
        rmserr[i] = rms[i] / np.sqrt(2. * nbins[i])
    # expected for white noise (WINN 2008, PONT 2006)
    stderr = (data.std() / np.sqrt(binsz)) * np.sqrt(nbins / (nbins - 1.))
    if isrmserr is True:
        return rms, stderr, binsz, rmserr
    else:
        return rms, stderr, binsz

def rmsplot(lc,model_lc, figname='allanplot.png'):
    time = lc.time
    residuals = lc.flux - model_lc
    residuals = residuals[np.argsort(time)]

    rms, stderr, binsz = computeRMS(residuals, binstep=1)
    normfactor = 1e-6
    plt.rcParams.update({'legend.fontsize': 11})
    plt.figure(1111, figsize=(8, 6))
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
    # if savefile != None:
    plt.savefig(figname)
    plt.close()
