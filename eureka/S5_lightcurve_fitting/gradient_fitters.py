import numpy as np
import pandas as pd
import copy
from io import StringIO
import os, sys
import h5py

import pymc3 as pm
import exoplanet
import pymc3_ext as pmx

from ..lib.readEPF import Parameters
from .likelihood import computeRedChiSq, lnprob, ln_like, ptform
from . import plots_s5 as plots
from ..lib import astropytable
from .fitters import group_variables, load_old_fitparams, save_fit

from multiprocessing import Pool

def exoplanetfitter(lc, model, meta, log, calling_function='exoplanet', **kwargs):
    """Perform sampling using exoplanet.

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
    calling_function: str, optional
        The name of the fitter/sampler currently being run (aka emcee if running emcee with lsq_first=True)
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
    freenames, freepars, prior1, prior2, priortype, indep_vars = group_variables(model)
    if hasattr(meta, 'old_fitparams') and meta.old_fitparams is not None:
        freepars = load_old_fitparams(meta, log, lc.channel, freenames)
    
    model.setup(lc.time, lc.flux, lc.unc)

    start = {}
    for name, val in zip(freenames, freepars):
        start[name] = val

    with model:
        map_soln = pmx.optimize(start=start)

    # Get the best fit params
    fit_params = np.array([map_soln[name] for name in freenames])
    model.fit_dict = {freenames[i]: fit_params[i] for i in range(len(freenames))}

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

    print('\nEXOPLANET RESULTS:')
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

    # Compute reduced chi-squared
    chi2red = computeRedChiSq(lc, log, model, meta, freenames)

    # Plot fit
    if meta.isplots_S5 >= 1:
        plots.plot_fit(lc, model, meta, fitter=calling_function)

    # Plot Allan plot
    if meta.isplots_S5 >= 3 and calling_function=='exoplanet':
        # This plot is only really useful if you're actually using the lsq fitter, otherwise don't make it
        plots.plot_rms(lc, model, meta, fitter=calling_function)

    # Plot residuals distribution
    if meta.isplots_S5 >= 3 and calling_function=='exoplanet':
        plots.plot_res_distr(lc, model, meta, fitter=calling_function)

    # Make a new model instance
    best_model = copy.copy(model)
    best_model.__setattr__('chi2red',chi2red)
    best_model.__setattr__('fit_params',fit_params)

    return best_model
