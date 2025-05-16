import numpy as np
import jax.numpy as jnp

from . import JaxModel
from ...lib.split_channels import split


class PolynomialModel(JaxModel):
    """Polynomial Model"""
    def __init__(self, **kwargs):
        """Initialize the polynomial model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from Model class
        super().__init__(**kwargs)
        self.name = 'polynomial'

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

    @property
    def time(self):
        """A getter for the time."""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time."""
        if isinstance(time_array, np.ma.core.MaskedArray):
            # Convert to a numpy array with NaN masking
            time_array = time_array.filled(np.nan)
        self._time = time_array

        if self.time is not None:
            # Convert to local time
            if self.multwhite:
                self.time_local = np.zeros(0)
                for chan in self.fitted_channels:
                    # Split the arrays that have lengths
                    # of the original time axis
                    time = split([self.time,], self.nints, chan)[0]
                    self.time_local = np.append(
                        self.time_local, time-np.nanmean(time))
            else:
                self.time_local = self.time-np.nanmean(self.time)

    def eval(self, eval=True, channel=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            The value of the model at the times self.time.
        """
        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel, ]

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        if eval:
            lib = np
            model = self.fit
        else:
            lib = jnp
            model = self.model

        poly_coeffs = np.zeros((nchan, 10)).tolist()
        # Parse 'c#' keyword arguments as coefficients
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0
            for i in range(10):
                if chan == 0:
                    parname = f'c{i}'
                else:
                    parname = f'c{i}_ch{chan}'
                poly_coeffs[c][i] = getattr(model, parname, 0)

        # Create the polynomial from the coeffs
        lcfinal = lib.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time_local
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            lcpiece = lib.zeros(len(time))
            for power in range(len(poly_coeffs[chan])):
                lcpiece += poly_coeffs[chan][power] * time**power
            lcfinal = lib.concatenate([lcfinal, lcpiece])

        return lcfinal
