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


def lsqfitter(lc, model, verbose=True, **kwargs):
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

    if verbose:
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
    # best_model.parameters = params
    # best_model.name = ', '.join(['{}:{}'.format(k, round(v[0], 2)) for k, v in params.dict.items()])

    return best_model

def demcfitter(time, data, model, uncertainty=None, verbose=True, **kwargs):
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

def lmfitter(time, data, model, uncertainty=None, verbose=True, **kwargs):
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

    if verbose:
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
