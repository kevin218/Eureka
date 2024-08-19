import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from . import PyMC3Model
from ...lib.split_channels import split, get_trim


class HSTRampModel(PyMC3Model):
    """Model for HST orbit-long exponential plus quadratic ramps"""
    def __init__(self, **kwargs):
        """Initialize the model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.differentiable_models.PyMC3Model.__init__().
        """
        # Inherit from PyMC3Model class
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
        self._time = np.ma.masked_invalid(time_array)
        if self.time is not None:
            # Convert to local time
            if self.multwhite:
                self.time_local = np.ma.zeros(self.time.shape)
                for chan in self.fitted_channels:
                    # Split the arrays that have lengths
                    # of the original time axis
                    trim1, trim2 = get_trim(self.nints, chan)
                    time = self.time[trim1:trim2]
                    self.time_local[trim1:trim2] = time-time[0]
            else:
                self.time_local = self.time - self.time[0]

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
            lib = np.ma
            model = self.fit
        else:
            lib = tt
            model = self.model

        # Parse 'h#' keyword arguments as coefficients
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            for i in range(6):
                if chan == 0:
                    hst_coeffs[c][i] = getattr(model, f'h{i}', 0)
                else:
                    hst_coeffs[c][i] = getattr(model, f'h{i}_ch{chan}', 0)

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
            # h5 is the orbital period of HST (~96 minutes)
            self.time_batch = self.time_local % h5
            lcpiece = (1+h0*lib.exp(-h1*self.time_batch + h2)
                       + h3*self.time_batch + h4*self.time_batch**2)
            hst_flux = lib.concatenate([hst_flux, lcpiece])

        return hst_flux
