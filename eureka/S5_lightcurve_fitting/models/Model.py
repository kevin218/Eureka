"""Base and child classes to handle models
used to fit light curves

Author: Joe Filippazzo
Email: jfilippazzo@stsci.edu
"""
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

import astropy.units as q

from ..parameters import Parameters
from ..utils import COLORS

class Model:
    def __init__(self, **kwargs):
        """
        Create a model instance
        """
        # Set up model attributes
        self.name = 'New Model'
        self.fitter = None
        self._time = None
        self._flux = None
        self._units = q.day
        self._parameters = Parameters()
        self.components = None
        self.fmt = None

        # Store the arguments as attributes
        for arg, val in kwargs.items():
            setattr(self, arg, val)

    def __mul__(self, other):
        """Multiply model components to make a combined model

        Parameters
        ----------
        other: ExoCTK.lightcurve_fitting.models.Model
            The model to multiply

        Returns
        -------
        ExoCTK.lightcurve_fitting.lightcurve.Model
            The combined model
        """
        # Make sure it is the right type
        attrs = ['units', 'flux', 'time']
        if not all([hasattr(other, attr) for attr in attrs]):
            raise TypeError('Only another Model instance may be multiplied.')

        # Combine the model parameters too
        params = self.parameters + other.parameters

        return CompositeModel([copy.copy(self), other], parameters=params)

    @property
    def flux(self):
        """A getter for the flux"""
        return self._flux

    @flux.setter
    def flux(self, flux_array):
        """A setter for the flux

        Parameters
        ----------
        flux_array: sequence
            The flux array
        """
        # Check the type
        if not isinstance(flux_array, (np.ndarray, tuple, list)):
            raise TypeError("flux axis must be a tuple, list, or numpy array.")

        # Set the array
        self._flux = np.array(flux_array)

    def interp(self, new_time):
        """Interpolate the flux to a new time axis

        Parameters
        ----------
        new_time: sequence, astropy.units.quantity.Quantity
            The time array
        """
        # Check the type
        if not isinstance(new_time, (np.ndarray, tuple, list)):
            raise TypeError("Time axis must be a tuple, list, or numpy array")

        # Calculate the new flux
        self.flux = np.interp(new_time, self.time, self.flux)

        # Set the new time axis
        self.time = new_time

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
        if (params is not None) and (type(params).__name__ != type(Parameters).__name):
            raise TypeError("'params' argument must be a JSON file, ascii\
                             file, or parameters.Parameters instance.")

        # Set the parameters attribute
        self._parameters = params

    def plot(self, time, components=False, ax=None, draw=False, ls='-', color='blue', zorder=np.inf, **kwargs):
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
            fig = plt.figure(figsize=(8,6))
            ax = fig.gca()

        # Set the time
        self.time = time

        # Plot the model
        label = self.fitter
        if self.name!='New Model':
            label += ': '+self.name
        ax.plot(self.time, self.eval(**kwargs), ls=ls, label=label, color=color, zorder=zorder)

        if components and self.components is not None:
            for comp in self.components:
                comp.plot(self.time, ax=ax, draw=False, color=next(COLORS), zorder=zorder, label=comp.fitter+': '+comp.name, **kwargs)

        # Format axes
        ax.set_xlabel(str(self.time_units))
        ax.set_ylabel('Flux')

        if draw:
            fig.show()
        else:
            return

    @property
    def time(self):
        """A getter for the time"""
        return self._time

    @time.setter
    def time(self, time_array, time_units='BJD'):
        """A setter for the time

        Parameters
        ----------
        time_array: sequence, astropy.units.quantity.Quantity
            The time array
        time_units: str
            The units of the input time_array, ['MJD', 'BJD', 'phase']
        """
        # Check the type
        if not isinstance(time_array, (np.ndarray, tuple, list)):
            raise TypeError("Time axis must be a tuple, list, or numpy array.")

        # Set the units
        self.time_units = time_units

        # Set the array
        self._time = time_array

    @property
    def units(self):
        """A getter for the units"""
        return self._units

    @units.setter
    def units(self, units):
        """A setter for the units

        Parameters
        ----------
        units: str
            The time units ['BJD', 'MJD', 'phase']
        """
        # Check the type
        if units not in ['BJD', 'MJD', 'phase']:
            raise TypeError("units axis must be 'BJD', 'MJD', or 'phase'.")

        self._units = units

class CompositeModel(Model):
    """A class to create composite models"""
    def __init__(self, models, **kwargs):
        """Initialize the composite model

        Parameters
        ----------
        models: sequence
            The list of models
        """
        # Inherit from Model calss
        super().__init__(**kwargs)

        # Store the models
        self.components = models

    def eval(self, **kwargs):
        """Evaluate the model components"""
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Empty flux
        flux = 1.

        # Evaluate flux at each model
        for model in self.components:
            if model.time is None:
                model.time = self.time
            flux *= model.eval(**kwargs)

        return flux

    def syseval(self, **kwargs):
        """Evaluate the systematic model components only"""
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Empty flux
        flux = 1.

        # Evaluate flux at each model
        for model in self.components:
            if model.modeltype == 'systematic':
                if model.time is None:
                    model.time = self.time
                flux *= model.eval(**kwargs)

        return flux

    def physeval(self, **kwargs):
        """Evaluate the physical model components only"""
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Empty flux
        flux = 1.

        # Evaluate flux at each model
        for model in self.components:
            if model.modeltype == 'physical':
                if model.time is None:
                    model.time = self.time
                flux *= model.eval(**kwargs)

        return flux

    def update(self, newparams, names, **kwargs):
        """Update parameters in the model components"""
        # Evaluate flux at each model
        for model in self.components:
            model.update(newparams, names, **kwargs)

        return
