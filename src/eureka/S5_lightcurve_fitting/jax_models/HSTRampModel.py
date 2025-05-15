import numpy as np
import jax
import jax.numpy as jnp

from . import JaxModel
from ...lib.split_channels import split, get_trim

jax.config.update("jax_enable_x64", True)


class HSTRampModel(JaxModel):
    """Model for HST orbit-long exponential plus quadratic ramps"""
    def __init__(self, **kwargs):
        """Initialize the model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.jax_models.JaxModel.__init__().
        """
        # Inherit from JaxModel class
        super().__init__(**kwargs)
        self.name = 'hst ramp'

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
                self.time_local = np.zeros(self.time.shape)
                for chan in self.fitted_channels:
                    # Split the arrays that have lengths
                    # of the original time axis
                    trim1, trim2 = get_trim(self.nints, chan)
                    time = self.time[trim1:trim2]
                    self.time_local[trim1:trim2] = time-time[0]
            else:
                self.time_local = self.time-self.time[0]

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
        ndarray
            The value of the model at the times self.time.
        """
        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel, ]

        hst_coeffs = np.zeros((nchan, 6)).tolist()

        if eval:
            lib = np
            model = self.fit
        else:
            lib = jnp
            model = self.model

        # Parse 'h#' keyword arguments as coefficients
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            for i in range(6):
                if chan == 0:
                    parname = f'h{i}'
                else:
                    parname = f'h{i}_ch{chan}'
                hst_coeffs[c][i] = getattr(model, parname, 0)

        hst_flux = lib.zeros(0)
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time_local
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            h0, h1, h2, h3, h4, h5 = hst_coeffs[c]
            # Batch time is relative to the start of each HST orbit
            # h4 is the orbital period of HST (~96 minutes)
            self.time_batch = (time - h5) % h4
            lcpiece = (1 +
                       h0*lib.exp(-h1*self.time_batch) +
                       h2*self.time_batch +
                       h3*self.time_batch**2)
            hst_flux = lib.concatenate([hst_flux, lcpiece])

        return hst_flux
