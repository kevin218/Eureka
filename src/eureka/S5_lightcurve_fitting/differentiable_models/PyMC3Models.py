import os
import copy
import numpy as np
import matplotlib.pyplot as plt

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

import pymc3 as pm
BoundedNormal_0 = pm.Bound(pm.Normal, lower=0.0)
BoundedNormal_90 = pm.Bound(pm.Normal, upper=90.)

from ...lib.readEPF import Parameters
from ..utils import COLORS


class fit_class:
    def __init__(self):
        pass


class PyMC3Model:
    def __init__(self, **kwargs):
        """Create a model instance.

        Parameters
        ----------
        **kwargs : dict
            Parameters to set in the PyMC3Model object.
            Any parameter named log will not be loaded into the
            PyMC3Model object as Logedit objects cannot be pickled
            which is required for multiprocessing.
        """
        # Set up default model attributes
        self.name = 'New PyMC3Model'
        self.fitter = None
        self._time = None
        self.time_units = 'BMJD_TDB'
        self._flux = None
        self.freenames = None
        self._parameters = Parameters()
        self.longparamlist = None
        self.paramtitles = None
        self.components = None
        self.modeltype = None
        self.fmt = None
        self.nchan = 1

        # Store the arguments as attributes
        for arg, val in kwargs.items():
            if arg != 'log':
                setattr(self, arg, val)

        # Initialize fit with all parameters (including fixed and independent)
        # which won't get changed throughout the fit
        self.fit = fit_class()
        for key in self.parameters.dict.keys():
            setattr(self.fit, key, getattr(self.parameters, key).value)

    def __mul__(self, other):
        """Multiply model components to make a combined model.

        Parameters
        ----------
        other : eureka.S5_lightcurve_fitting.models.Model
            The model to multiply.

        Returns
        -------
        eureka.S5_lightcurve_fitting.models.CompositeModel
            The combined model.
        """
        # Make sure it is the right type
        attrs = ['flux', 'time']
        if not all([hasattr(other, attr) for attr in attrs]):
            raise TypeError('Only another Model instance may be multiplied.')

        # Combine the model parameters too
        parameters = self.parameters + other.parameters
        paramtitles = self.paramtitles.append(other.paramtitles)

        return CompositePyMC3Model([copy.copy(self), other],
                                   parameters=parameters,
                                   paramtitles=paramtitles)

    @property
    def flux(self):
        """A getter for the flux"""
        return self._flux

    @flux.setter
    def flux(self, flux_array):
        """A setter for the flux

        Parameters
        ----------
        flux_array : sequence
            The flux array
        """
        # Check the type
        if not isinstance(flux_array, (np.ndarray, tuple, list)):
            raise TypeError("flux axis must be a tuple, list, or numpy array.")

        # Set the array
        self._flux = np.ma.masked_array(flux_array)

    @property
    def time(self):
        """A getter for the time"""
        return self._time

    @time.setter
    def time(self, time_array, time_units='BMJD'):
        """A setter for the time

        Parameters
        ----------
        time_array: sequence, astropy.units.quantity.Quantity
            The time array
        time_units: str
            The units of the input time_array, e.g. ['MJD', 'BMJD', 'phase']
        """
        # Check the type
        if not isinstance(time_array, (np.ndarray, tuple, list)):
            raise TypeError("Time axis must be a tuple, list, or numpy array.")

        # Set the units
        self.time_units = time_units

        # Set the array
        # self._time = np.array(time_array)
        self._time = np.ma.masked_array(time_array)

    @property
    def parameters(self):
        """A getter for the parameters"""
        return self._parameters

    @parameters.setter
    def parameters(self, params):
        """A setter for the parameters"""
        # Process if it is a parameters file
        if isinstance(params, str) and os.path.isfile(params):
            params = Parameters(params)

        # Or a Parameters instance
        if (params is not None) and (type(params).__name__ !=
                                     Parameters.__name__):
            raise TypeError("'params' argument must be a JSON file, "
                            "ascii file, or Parameters instance.")

        # Set the parameters attribute
        self._parameters = params

    def update(self, newparams, **kwargs):
        """Update the model with new parameter values.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        **kwargs : dict
            Unused by the base
            eureka.S5_lightcurve_fitting.diferentiable_models.PyMC3Model class.
        """
        for val, key in zip(newparams, self.freenames):
            setattr(self.fit, key, val)

    def setup(self, **kwargs):
        """A placeholder function to do any additional setup.
        """
        return

    def plot(self, time, components=False, ax=None, draw=False, color='blue',
             zorder=np.inf, share=False, chan=0, **kwargs):
        """Plot the model

        Parameters
        ----------
        time: array-like
            The time axis to use
        components: bool
            Plot all model components
        ax: Matplotlib Axes
            The figure axes to plot on

        Returns
        -------
        bokeh.plotting.figure
            The figure
        """
        # Make the figure
        if ax is None:
            fig = plt.figure(5103, figsize=(8, 6))
            ax = fig.gca()

        # Set the time
        self.time = time

        # Plot the model
        label = self.fitter
        if self.name != 'New Model':
            label += ': '+self.name
        
        if not share:
            channel = 0
        else:
            channel = chan
        model = self.eval(channel=channel, **kwargs)

        ax.plot(self.time, model, '.', ls='', ms=2, label=label,
                color=color, zorder=zorder)

        if components and self.components is not None:
            for component in self.components:
                component.plot(self.time, ax=ax, draw=False,
                               color=next(COLORS), zorder=zorder, share=share,
                               chan=chan, **kwargs)

        # Format axes
        ax.set_xlabel(str(self.time_units))
        ax.set_ylabel('Flux')

        if draw:
            fig.show()
        else:
            return


class CompositePyMC3Model(PyMC3Model):
    """A class to create composite models."""
    def __init__(self, components, **kwargs):
        """Initialize the composite model.

        Parameters
        ----------
        models : sequence
            The list of models.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
        # Inherit from PyMC3Model class
        super().__init__(**kwargs)

        # Setup PyMC3 model
        self.model = pm.Model()

        # Store the components
        self.components = components
        for component in self.components:
            # Add the PyMC3 model to each component
            component.model = self.model

        # Setup PyMC3 model parameters
        with self.model:
            for parname in self.paramtitles:
                param = getattr(self.parameters, parname)
                if param.ptype in ['independent', 'fixed']:
                    setattr(self.model, parname, param.value)
                elif param.ptype not in ['free', 'shared', 'white_free',
                                         'white_fixed']:
                    message = (f'ptype {param.ptype} for parameter '
                               f'{param.name} is not recognized.')
                    raise ValueError(message)
                else:
                    for c in range(self.nchan):
                        if c != 0:
                            parname_temp = parname+'_'+str(c)
                        else:
                            parname_temp = parname

                        if param.ptype == 'free' or c == 0:
                            if param.prior == 'U':
                                setattr(self.model, parname_temp,
                                        pm.Uniform(parname_temp,
                                                   lower=param.priorpar1,
                                                   upper=param.priorpar2,
                                                   testval=param.value))
                            elif param.prior == 'N':
                                if parname in ['rp', 'per', 'ecc',
                                               'scatter_mult', 'scatter_ppm',
                                               'c0']:
                                    setattr(self.model, parname_temp,
                                            BoundedNormal_0(
                                                parname_temp,
                                                mu=param.priorpar1,
                                                sigma=param.priorpar2,
                                                testval=param.value))
                                elif parname in ['inc']:
                                    setattr(self.model, parname_temp,
                                            BoundedNormal_90(
                                                parname_temp,
                                                mu=param.priorpar1,
                                                sigma=param.priorpar2,
                                                testval=param.value))
                                else:
                                    setattr(self.model, parname_temp,
                                            pm.Normal(parname_temp,
                                                      mu=param.priorpar1,
                                                      sigma=param.priorpar2,
                                                      testval=param.value))
                            elif param.prior == 'LU':
                                setattr(self.model, parname_temp,
                                        tt.exp(pm.Uniform(
                                            parname_temp, 
                                            lower=param.priorpar1,
                                            upper=param.priorpar2,
                                            testval=param.value)))
                        else:
                            # If a parameter is shared, make it equal to the
                            # 0th parameter value
                            setattr(self.model, parname_temp,
                                    getattr(self.model, parname))

    def setup(self, time, flux, lc_unc):
        self.time = time
        self.flux = flux
        self.lc_unc = lc_unc

        with self.model:
            if hasattr(self.model, 'scatter_ppm'):
                self.scatter_array = (self.model.scatter_ppm
                                      * tt.ones(len(self.time)))
                for c in range(1, self.nchan):
                    parname_temp = 'scatter_ppm_'+str(c)
                    self.scatter_array = tt.concatenate(
                        [self.scatter_array,
                         getattr(self.model, parname_temp)
                         * tt.ones(len(self.time))])
                self.scatter_array /= 1e6
            if hasattr(self.model, 'scatter_mult'):
                # Fitting the noise level as a multiplier
                self.scatter_array = (self.model.scatter_mult *
                                      self.lc_unc[:len(self.time)])
                for c in range(1, self.nchan):
                    if self.parameters.scatter_mult.ptype == 'fixed':
                        parname_temp = 'scatter_mult'
                    else:
                        parname_temp = 'scatter_mult_'+str(c)
                    self.scatter_array = tt.concatenate(
                        [self.scatter_array,
                         (getattr(self.model, parname_temp) *
                          self.lc_unc[c*len(self.time):(c+1)*len(self.time)])])
            if not hasattr(self, 'scatter_array'):
                # Not fitting the noise level
                self.scatter_array = self.lc_unc

            for component in self.components:
                # Do any one-time setup needed after model initialization and
                # before evaluating the model
                component.setup(full_model=self)

            # This is how we tell pymc3 about our observations;
            # we are assuming they are normally distributed about
            # the true model. This line effectively defines our
            # likelihood function.
            pm.Normal("obs", mu=self.eval(eval=False), sd=self.scatter_array,
                      observed=self.flux)

    def eval(self, eval=True, channel=None, **kwargs):
        """Evaluate the model components.

        Parameters
        ----------
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        flux : ndarray
            The evaluated model predictions at the times self.time.
        """
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        flux = np.ones(len(self.time)*self.nchan)

        # Evaluate flux of each component
        for component in self.components:
            if component.time is None:
                component.time = self.time
            if component.modeltype != 'GP':
                flux *= component.eval(eval=eval, channel=channel, **kwargs)

        return flux

    def syseval(self, eval=True, channel=None, **kwargs):
        """Evaluate the systematic model components only.

        Parameters
        ----------
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        flux : ndarray
            The evaluated systematics model predictions at the times self.time.
        """
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        flux = np.ones(len(self.time)*self.nchan)

        # Evaluate flux at each model
        for component in self.components:
            if component.modeltype == 'systematic':
                if component.time is None:
                    component.time = self.time
                flux *= component.eval(eval=eval, channel=channel, **kwargs)

        return flux

    def physeval(self, eval=True, channel=None, **kwargs):
        """Evaluate the physical model components only.

        Parameters
        ----------
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        flux : ndarray
            The evaluated physical model predictions at the times self.time
        """
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        flux = np.ones(len(self.time)*self.nchan)

        # Evaluate flux at each model
        for component in self.components:
            if component.modeltype == 'physical':
                if component.time is None:
                    component.time = self.time
                flux *= component.eval(eval=eval, channel=channel, **kwargs)

        return flux, self.time

    def compute_fp(self, theta=0):
        # Evaluate flux at each model
        for component in self.components:
            if component.name == 'starry':
                return component.compute_fp(theta=theta)

    def update(self, newparams, **kwargs):
        """Update parameters in the model components.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.update().
        """
        for component in self.components:
            component.update(newparams, **kwargs)
