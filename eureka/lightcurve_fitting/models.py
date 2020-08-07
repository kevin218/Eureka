"""Base and child classes to handle models
used to fit light curves

Author: Joe Filippazzo
Email: jfilippazzo@stsci.edu
"""
import numpy as np
import copy
import inspect
import os

import astropy.units as q
try:
    import batman
except ImportError:
    print("Could not import batman. Functionality may be limited.")
from bokeh.plotting import figure, show

from .parameters import Parameters
from .utils import COLORS
from .limb_darkening_fit import ld_profile


class Model:
    def __init__(self, **kwargs):
        """
        Create a model instance
        """
        # Set up model attributes
        self.name = 'New Model'
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
        if isinstance(params, str) and os.file.exists(params):
            params = Parameters(params)

        # Or a Parameters instance
        if not isinstance(params, (Parameters, type(None))):
            raise TypeError("'params' argument must be a JSON file, ascii\
                             file, or parameters.Parameters instance.")

        # Set the parameters attribute
        self._parameters = params

    def plot(self, time, components=False, fig=None, draw=False, color='blue', **kwargs):
        """Plot the model

        Parameters
        ----------
        time: array-like
            The time axis to use
        components: bool
            Plot all model components
        fig: bokeh.plotting.figure (optional)
            The figure to plot on

        Returns
        -------
        bokeh.plotting.figure
            The figure
        """
        # Make the figure
        if fig is None:
            fig = figure(width=800, height=400)

        # Set the time
        self.time = time

        # Plot the model
        fig.line(self.time, self.eval(**kwargs), legend=self.name, color=color)

        if components and self.components is not None:
            for comp in self.components:
                fig = comp.plot(self.time, fig=fig, draw=False, color=next(COLORS), **kwargs)

        # Format axes
        fig.xaxis.axis_label = str(self.units)
        fig.yaxis.axis_label = 'Flux'

        if draw:
            show(fig)
        else:
            return fig

    @property
    def time(self):
        """A getter for the time"""
        return self._time

    @time.setter
    def time(self, time_array, units='MJD'):
        """A setter for the time

        Parameters
        ----------
        time_array: sequence, astropy.units.quantity.Quantity
            The time array
        units: str
            The units of the input time_array, ['MJD', 'BJD', 'phase']
        """
        # Check the type
        if not isinstance(time_array, (np.ndarray, tuple, list)):
            raise TypeError("Time axis must be a tuple, list, or numpy array.")

        # Set the units
        self.units = units

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

    def update(self, newparams, names, **kwargs):
        """Update parameters in the model components"""
        # Evaluate flux at each model
        for model in self.components:
            model.update(newparams, names, **kwargs)

        return


class PolynomialModel(Model):
    """Polynomial Model"""
    def __init__(self, **kwargs):
        """Initialize the polynomial model
        """
        # Inherit from Model class
        super().__init__(**kwargs)

        # Check for Parameters instance
        self.parameters = kwargs.get('parameters')

        # Generate parameters from kwargs if necessary
        if self.parameters is None:
            self._parse_coeffs(kwargs)

    def _parse_coeffs(self, coeff_dict):
        """Convert dict of 'c#' coefficients into a list
        of coefficients in decreasing order, i.e. ['c2','c1','c0']

        Parameters
        ----------
        coeff_dict: dict
            The dictionary of coefficients

        Returns
        -------
        np.ndarray
            The sequence of coefficient values
        """
        params = {cN: coeff for cN, coeff in coeff_dict.items()
                  if cN.startswith('c') and cN[1:].isdigit()}
        self.parameters = Parameters(**params)

        # Parse 'c#' keyword arguments as coefficients
        coeffs = np.zeros(100)
        for k, v in self.parameters.dict.items():
            if k.lower().startswith('c') and k[1:].isdigit():
                coeffs[int(k[1:])] = v[0]

        # Trim zeros and reverse
        self.coeffs = np.trim_zeros(coeffs)[::-1]

    def eval(self, **kwargs):
        """Evaluate the function with the given values"""
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Create the polynomial from the coeffs
        poly = np.poly1d(self.coeffs)

        # Convert to local time
        time_local = self.time - self.time.mean()

        # Evaluate the polynomial
        return np.polyval(poly, time_local)

    def update(self, newparams, names, **kwargs):
        """Update parameter values"""
        for ii,arg in enumerate(names):
            val = getattr(self.parameters,arg).values[1:]
            val[0] = newparams[ii]
            setattr(self.parameters, arg, val)
        return

class TransitModel(Model):
    """Transit Model"""
    def __init__(self, **kwargs):
        """Initialize the transit model
        """
        # Inherit from Model calss
        super().__init__(**kwargs)

        # Check for Parameters instance
        self.parameters = kwargs.get('parameters')

        # Generate parameters from kwargs if necessary
        if self.parameters is None:
            self.parameters = Parameters(**kwargs)

        # Store the ld_profile
        self.ld_func = ld_profile(self.parameters.limb_dark.value)
        len_params = len(inspect.signature(self.ld_func).parameters)
        self.coeffs = ['u{}'.format(n) for n in range(len_params)[1:]]

    def eval(self, **kwargs):
        """Evaluate the function with the given values"""
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Generate with batman
        bm_params = batman.TransitParams()

        # Set all parameters
        for arg, val in self.parameters.dict.items():
            setattr(bm_params, arg, val[0])
        #for p in self.parameters.list:
        #    setattr(bm_params, p[0], p[1])

        # Combine limb darkening coeffs
        bm_params.u = [getattr(self.parameters, u).value for u in self.coeffs]

        # Use batman ld_profile name
        if self.parameters.limb_dark.value == '4-parameter':
            bm_params.limb_dark = 'nonlinear'

        # Make the eclipse
        tt = self.parameters.transittype.value
        m_eclipse = batman.TransitModel(bm_params, self.time, transittype=tt)

        # Evaluate the light curve
        return m_eclipse.light_curve(bm_params)

    def update(self, newparams, names, **kwargs):
        """Update parameter values"""
        for ii,arg in enumerate(names):
            val = getattr(self.parameters,arg).values[1:]
            val[0] = newparams[ii]
            setattr(self.parameters, arg, val)
        # ii = 0
        # for arg, val in self.parameters.dict.items():
        #     val[0] = newparams[ii]
        #     setattr(self.parameters, arg, val)
        #     ii += 1
        return
