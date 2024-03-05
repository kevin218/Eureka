import numpy as np
import matplotlib.pyplot as plt
import copy
import os

from ..utils import COLORS
from ...lib.readEPF import Parameters
from ...lib.split_channels import split


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
        # Set up default model attributes
        self.name = kwargs.get('name', 'New Model')
        self.nchannel = kwargs.get('nchannel', 1)
        self.nchannel_fitted = kwargs.get('nchannel_fitted', 1)
        self.fitted_channels = kwargs.get('fitted_channels', [0, ])
        self.multwhite = kwargs.get('multwhite')
        self.nints = kwargs.get('nints')
        self.fitter = kwargs.get('fitter', None)
        self.time = kwargs.get('time', None)
        self.time_units = kwargs.get('time_units', 'BMJD_TDB')
        self.flux = kwargs.get('flux', None)
        self.freenames = kwargs.get('freenames', None)
        self._parameters = kwargs.get('parameters', Parameters())
        self.longparamlist = kwargs.get('longparamlist', None)
        self.paramtitles = kwargs.get('paramtitles', None)
        self.modeltype = kwargs.get('modeltype', None)
        self.fmt = kwargs.get('fmt', None)

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
        attrs = ['flux', 'time']
        if not all([hasattr(other, attr) for attr in attrs]):
            raise TypeError('Only another Model instance may be multiplied.')

        # Combine the model parameters too
        parameters = self.parameters + other.parameters
        if self.paramtitles is None:
            paramtitles = other.paramtitles
        elif other.paramtitles is not None:
            paramtitles = self.paramtitles.append(other.paramtitles)
        else:
            paramtitles = self.paramtitles

        return CompositeModel([copy.copy(self), other], parameters=parameters,
                              paramtitles=paramtitles)

    @property
    def flux(self):
        """A getter for the flux."""
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
        if not isinstance(flux_array, (np.ndarray, tuple, list, type(None))):
            raise TypeError("flux axis must be a tuple, list, or numpy array.")

        # Set the array
        self._flux = np.ma.masked_array(flux_array)

    @property
    def time(self):
        """A getter for the time"""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time"""
        # Check the type
        if not isinstance(time_array, (np.ndarray, tuple, list, type(None))):
            raise TypeError("Time axis must be a tuple, list, or numpy array.")

        # Set the array
        self._time = np.ma.masked_array(time_array)

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

    def interp(self, new_time, nints, **kwargs):
        """Evaluate the model over a different time array.

        Parameters
        ----------
        new_time : sequence
            The time array.
        nints : list
            The number of integrations for each channel, for the new
            time array.
        **kwargs : dict
            Additional parameters to pass to self.eval().
        """
        # Save the current values
        old_time = copy.deepcopy(self.time)
        old_nints = copy.deepcopy(self.nints)

        # Evaluate the model on the new time array
        self.time = new_time
        self.nints = nints
        interp_flux = self.eval(**kwargs)

        # Reset the old values
        self.time = old_time
        self.nints = old_nints

        return interp_flux

    def update(self, newparams, **kwargs):
        """Update the model with new parameter values.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        **kwargs : dict
            Unused by the base
            eureka.S5_lightcurve_fitting.models.Model class.
        """
        for val, arg in zip(newparams, self.freenames):
            # For now, the dict and Parameter are separate
            self.parameters.dict[arg][0] = val
            getattr(self.parameters, arg).value = val
        self._parse_coeffs()
        return

    def _parse_coeffs(self):
        """A placeholder function to do any additional processing when
        calling update.
        """
        return

    def plot(self, components=False, ax=None, draw=False, color='blue',
             zorder=np.inf, share=False, chan=0, **kwargs):
        """Plot the model.

        Parameters
        ----------
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
        """
        # Make the figure
        if ax is None:
            fig = plt.figure(5103, figsize=(8, 6))
            ax = fig.gca()

        # Plot the model
        label = self.fitter
        if self.name != 'New Model':
            label += ': '+self.name

        if not share:
            channel = 0
        else:
            channel = chan
        model = self.eval(channel=channel, incl_GP=True, **kwargs)

        time = self.time
        if self.multwhite:
            # Split the arrays that have lengths of the original time axis
            time = split([time, ], self.nints, chan)[0]

        ax.plot(time, model, '.', ls='', ms=1, label=label, color=color,
                zorder=zorder)

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
        # Store the models
        self.components = models

        # Inherit from Model class
        super().__init__(**kwargs)

        self.GP = False
        for component in self.components:
            if component.modeltype == 'GP':
                self.GP = True

    @property
    def freenames(self):
        """A getter for the freenames."""
        return self._freenames

    @freenames.setter
    def freenames(self, freenames):
        """A setter for the freenames."""
        # Update the components' freenames
        for component in self.components:
            component.freenames = freenames

        # Set the freenames attribute
        self._freenames = freenames

    def eval(self, incl_GP=False, channel=None, **kwargs):
        """Evaluate the model components.

        Parameters
        ----------
        incl_GP : bool; optional
            Whether or not to include the GP's predictions in the
            evaluated model predictions.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
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

        if channel is None:
            nchan = self.nchannel_fitted
        else:
            nchan = 1

        if self.multwhite:
            time = self.time
            if channel is not None:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, channel)[0]
            flux = np.ma.ones(len(time))
        else:
            flux = np.ma.ones(len(self.time)*nchan)

        # Evaluate flux of each component
        for component in self.components:
            if component.time is None:
                component.time = self.time
            if component.modeltype != 'GP':
                flux *= component.eval(channel=channel, **kwargs)

        if incl_GP:
            flux += self.GPeval(flux, channel=channel, **kwargs)

        return flux

    def syseval(self, channel=None, incl_GP=False, **kwargs):
        """Evaluate the systematic model components only.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        incl_GP : bool; optional
            Whether or not to include the GP's predictions in the
            evaluated model predictions.
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

        if channel is None:
            nchan = self.nchannel_fitted
        else:
            nchan = 1

        if self.multwhite:
            time = self.time
            if channel is not None:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, channel)[0]
            flux = np.ma.ones(len(time))
        else:
            flux = np.ma.ones(len(self.time)*nchan)

        # Evaluate flux at each component
        for component in self.components:
            if component.modeltype == 'systematic':
                if component.time is None:
                    component.time = self.time
                flux *= component.eval(channel=channel, **kwargs)

        if incl_GP:
            flux += self.GPeval(flux, channel=channel, **kwargs)

        return flux

    def GPeval(self, fit, channel=None, **kwargs):
        """Evaluate the GP model components only.

        Parameters
        ----------
        fit : ndarray
            The model predictions (excluding the GP).
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
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

        if channel is None:
            nchan = self.nchannel_fitted
        else:
            nchan = 1

        if self.multwhite:
            time = self.time
            if channel is not None:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, channel)[0]
            flux = np.ma.zeros(len(time))
        else:
            flux = np.ma.zeros(len(self.time)*nchan)

        # Evaluate flux
        for component in self.components:
            if component.modeltype == 'GP':
                flux = component.eval(fit, channel=channel,
                                      **kwargs)
        return flux

    def physeval(self, interp=False, channel=None, **kwargs):
        """Evaluate the physical model components only.

        Parameters
        ----------
        interp : bool; optional
            Whether to uniformly sample in time or just use
            the self.time time points. Defaults to False.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        flux : ndarray
            The evaluated physical model predictions at the times self.time
            if interp==False, else at evenly spaced times between self.time[0]
            and self.time[-1] with spacing self.time[1]-self.time[0].
        new_time : ndarray
            The time values at which flux has been computed.
        nints_interp : list
            The number of time points per lightcurve for each lightcurve
            (after interpolation if interp is True).
        """
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel]

        if interp:
            if self.multwhite:
                new_time = []
                nints_interp = []
                for chan in channels:
                    # Split the arrays that have lengths of
                    # the original time axis
                    time = split([self.time, ], self.nints, chan)[0]

                    # Remove masked points at the start or end to avoid
                    # extrapolating out to those points
                    time = time[~np.ma.getmaskarray(time)]

                    # Get time step on full time array to ensure good steps
                    dt = np.min(np.diff(time))

                    # Interpolate as needed
                    steps = int(np.round((time[-1]-time[0])/dt+1))
                    nints_interp.append(steps)
                    new_time.extend(np.linspace(time[0], time[-1], steps,
                                                endpoint=True))
                new_time = np.array(new_time)
            else:
                time = self.time

                # Remove masked points at the start or end to avoid
                # extrapolating out to those points
                time = time[~np.ma.getmaskarray(time)]

                # Get time step on full time array to ensure good steps
                dt = np.min(np.diff(time))

                # Interpolate as needed
                dt = time[1]-time[0]
                steps = int(np.round((time[-1]-time[0])/dt+1))
                nints_interp = np.ones(nchan)*steps
                new_time = np.linspace(time[0], time[-1], steps, endpoint=True)
        else:
            new_time = self.time
            if self.multwhite and channel is not None:
                # Split the arrays that have lengths of the original time axis
                new_time = split([new_time, ], self.nints, channel)[0]
            nints_interp = self.nints

        # Setup the flux array
        if self.multwhite:
            flux = np.ma.ones(len(new_time))
        else:
            flux = np.ma.ones(len(new_time)*nchan)

        # Evaluate flux at each component
        for component in self.components:
            if component.modeltype == 'physical':
                if component.time is None:
                    component.time = self.time
                if interp:
                    flux *= component.interp(new_time, nints_interp, **kwargs)
                else:
                    flux *= component.eval(channel=channel, **kwargs)
        return flux, new_time, nints_interp

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
