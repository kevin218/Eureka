import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from . import PyMC3Model


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

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

    @property
    def time(self):
        """A getter for the time."""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time."""
        self._time = time_array
        if self.time is not None:
            # Convert to local time
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
            nchan = self.nchan
            channels = np.arange(nchan)
        else:
            nchan = 1
            channels = [channel, ]

        hst_coeffs = np.zeros((nchan, 6)).tolist()

        if eval:
            lib = np
            model = self.fit
        else:
            lib = tt
            model = self.model

        # Parse 'h#' keyword arguments as coefficients
        for j in range(nchan):
            for i in range(6):
                try:
                    if channels[j] == 0:
                        hst_coeffs[j][i] = getattr(model, f'h{i}')
                    else:
                        hst_coeffs[j][i] = getattr(model,
                                                   f'h{i}_{channels[j]}')
                except AttributeError:
                    pass

        hst_flux = lib.zeros(0)
        for c in range(nchan):
            h0, h1, h2, h3, h4, h5 = hst_coeffs[c]
            # Batch time is relative to the start of each HST orbit
            # h5 is the orbital period of HST (~96 minutes)
            self.time_batch = self.time_local % h5
            lcpiece = (1+h0*lib.exp(-h1*self.time_batch + h2)
                       + h3*self.time_batch + h4*self.time_batch**2)
            hst_flux = lib.concatenate([hst_flux, lcpiece])

        return hst_flux
