import numpy as np
import matplotlib.pyplot as plt
import copy
import os

import astropy.units as q

from ...lib.readEPF import Parameters
from ..utils import COLORS


class Model:
    def __init__(self, **kwargs):
        """Create a model instance.

        Parameters
        ----------
        **kwargs : dict
            Parameters to set in the Model object.
            Any parameter named log will not be loaded into the
            Model object as Logedit objects cannot be pickled
            which is required for multiprocessing.
        """
        # Set up model attributes
        self.name = 'New Model'
        self.fitter = None
        self._time = None
        self._flux = None
        self._units = q.day
        self._parameters = Parameters()
        self.components = None
        self.modeltype = None
        self.fmt = None
        if hasattr(kwargs, 'nchan'):
            self.nchan = kwargs.get('nchan')
        else:
            self.nchan = 1

        # Store the arguments as attributes
        for arg, val in kwargs.items():
            if arg != 'log':
                setattr(self, arg, val)

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
        attrs = ['units', 'flux', 'time']
        if not all([hasattr(other, attr) for attr in attrs]):
            raise TypeError('Only another Model instance may be multiplied.')

        # Combine the model parameters too
        params = self.parameters + other.parameters

        return CompositeModel([copy.copy(self), other], parameters=params)

    @property
    def flux(self):
        """A getter for the flux."""
        return self._flux

    @flux.setter
    def flux(self, flux_array):
        """A setter for the flux."""
        # Check the type
        if not isinstance(flux_array, (np.ndarray, tuple, list)):
            raise TypeError("flux axis must be a tuple, list, or numpy array.")

        # Set the array
        self._flux = np.ma.masked_array(flux_array)

    def interp(self, new_time, **kwargs):
        """Evaluate the model over a different time array.

        Parameters
        ----------
        new_time : sequence
            The time array.
        **kwargs : dict
            Additional parameters to pass to self.eval().
        """
        # Check the type
        if not isinstance(new_time, (np.ndarray, tuple, list)):
            raise TypeError("Time axis must be a tuple, list, or numpy array")

        # Save the current time array
        old_time = copy.deepcopy(self.time)

        # Evaluate the model on the new time array
        self.time = new_time
        interp_flux = self.eval(**kwargs)

        # Reset the time array
        self.time = old_time

        return interp_flux

    def update(self, newparams, names, **kwargs):
        """Update the model with new parameter values.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        names : list
            Parameter names.
        **kwargs : dict
            Unused by the base
            eureka.S5_lightcurve_fitting.models.Model class.
        """
        for ii, arg in enumerate(names):
            if hasattr(self.parameters, arg):
                val = getattr(self.parameters, arg).values[1:]
                val[0] = newparams[ii]
                setattr(self.parameters, arg, val)
        self._parse_coeffs()
        return

    def _parse_coeffs(self):
        """A placeholder function to do any additional processing when
        calling update.
        """
        return

    @property
    def parameters(self):
        """A getter for the parameters."""
        return self._parameters

    @parameters.setter
    def parameters(self, params):
        """A setter for the parameters."""
        # Process if it is a parameters file
        if isinstance(params, str) and os.path.isfile(params):
            params = Parameters(params)

        # Or a Parameters instance
        if (params is not None) and (type(params).__name__ !=
                                     Parameters.__name__):
            raise TypeError("'params' argument must be a JSON file, "
                            "ascii file, or parameters.Parameters instance.")

        # Set the parameters attribute
        self._parameters = params

    def plot(self, time, components=False, ax=None, draw=False, color='blue',
             zorder=np.inf, share=False, chan=0, **kwargs):
        """Plot the model.

        Parameters
        ----------
        time : array-like
            The time axis to use.
        components : bool; optional
            Plot all model components.
        ax : Matplotlib Axes; optional
            The figure axes to plot on.
        draw : bool; optional
            Whether or not to display the plot. Defaults to False.
        color : str; optional
            The color to use for the plot. Defaults to 'blue'.
        zorder : numeric; optional
            The zorder for the plot. Defaults to np.inf.
        share : bool; optional
            Whether or not this model is a shared model. Defaults to False.
        chan : int; optional
            The current channel number. Detaults to 0.
        **kwargs : dict
            Additional parameters to pass to plot and self.eval().

        Returns
        -------
        figure
            The figure.
        """
        # Make the figure
        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.gca()

        # Set the time
        self.time = time

        # Plot the model
        label = self.fitter
        if self.name != 'New Model':
            label += ': '+self.name

        flux = self.eval(**kwargs)
        if share:
            flux = flux[chan*len(self.time):(chan+1)*len(self.time)]

        ax.plot(self.time, flux, '.', ls='', ms=2, label=label, color=color,
                zorder=zorder)

        if components and self.components is not None:
            for comp in self.components:
                comp.plot(self.time, ax=ax, draw=False, color=next(COLORS),
                          zorder=zorder, share=share, chan=chan, **kwargs)

        # Format axes
        ax.set_xlabel(str(self.time_units))
        ax.set_ylabel('Flux')

        if draw:
            fig.show()
        else:
            return

    @property
    def time(self):
        """A getter for the time."""
        return self._time

    @time.setter
    def time(self, time_array, time_units='BJD_TDB'):
        """A setter for the time.

        Parameters
        ----------
        time_array : sequence, astropy.units.quantity.Quantity
            The time array.
        time_units : str; optional
            The units of the input time_array, ['MJD','BJD','BJD_TDB','phase'].
            Defaults to 'BJD_TDB'.
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
        """A getter for the unit.s"""
        return self._units

    @units.setter
    def units(self, units):
        """A setter for the units.

        Parameters
        ----------
        units : str
            The time units ['BJD', 'BJD_TDB', 'MJD', 'phase'].
        """
        # Check the type
        if units not in ['BJD', 'BJD_TDB', 'MJD', 'phase']:
            raise TypeError("units axis must be 'BJD', 'BJD_TDB, "
                            "'MJD', or 'phase'.")

        self._units = units


class CompositeModel(Model):
    """A class to create composite models."""
    def __init__(self, models, **kwargs):
        """Initialize the composite model.

        Parameters
        ----------
        models : sequence
            The list of models.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
        # Inherit from Model calss
        super().__init__(**kwargs)

        # Store the models
        self.components = models

        self.GP = False
        for model in self.components:
            if model.modeltype == 'GP':
                self.GP = True

    def eval(self, incl_GP=False, **kwargs):
        """Evaluate the model components.

        Parameters
        ----------
        incl_GP : bool; optional
            Whether or not to include the GP's predictions in the
            evaluated model predictions.
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

        # Evaluate flux at each model
        for model in self.components:
            if model.time is None:
                model.time = self.time
            if model.modeltype != 'GP':
                flux *= model.eval(**kwargs)

        if incl_GP:
            flux += self.GPeval(flux)

        return flux

    def syseval(self, **kwargs):
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
        for model in self.components:
            if model.modeltype == 'systematic':
                if model.time is None:
                    model.time = self.time
                flux *= model.eval(**kwargs)

        return flux

    def GPeval(self, fit, **kwargs):
        """Evaluate the GP model components only.

        Parameters
        ----------
        fit : ndarray
            The model predictions (excluding the GP).
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        flux : ndarray
            The evaluated GP model predictions at the times self.time.
        """
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Set the default value
        flux = np.zeros(len(self.time)*self.nchan)

        # Evaluate flux
        for model in self.components:
            if model.modeltype == 'GP':
                flux = model.eval(fit, **kwargs)
        return flux

    def physeval(self, interp=False, **kwargs):
        """Evaluate the physical model components only.

        Parameters
        ----------
        interp : bool; optional
            Whether to uniformly sample in time or just use
            the self.time time points. Defaults to False.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        flux : ndarray
            The evaluated physical model predictions at the times self.time
            if interp==False, else at evenly spaced times between self.time[0]
            and self.time[-1] with spacing self.time[1]-self.time[0].
        """
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        if interp:
            dt = self.time[1]-self.time[0]
            steps = int(np.round((self.time[-1]-self.time[0])/dt+1))
            new_time = np.linspace(self.time[0], self.time[-1], steps,
                                   endpoint=True)
        else:
            new_time = self.time

        flux = np.ones(len(self.time)*self.nchan)

        # Evaluate flux at each model
        for model in self.components:
            if model.modeltype == 'physical':
                if model.time is None:
                    model.time = self.time
                if interp:
                    flux *= model.interp(new_time, **kwargs)
                else:
                    flux *= model.eval(**kwargs)
        return flux, new_time

    def update(self, newparams, names, **kwargs):
        """Update parameters in the model components.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        names : list
            Parameter names.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.update().
        """
        # Evaluate flux at each model
        for model in self.components:
            model.update(newparams, names, **kwargs)
        return
