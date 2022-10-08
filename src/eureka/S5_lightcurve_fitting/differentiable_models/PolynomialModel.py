import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from . import PyMC3Model


class PolynomialModel(PyMC3Model):
    def __init__(self, **kwargs):
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
            self.time_local = self.time - self.time.mean()

    def eval(self, eval=True, channel=None, **kwargs):
        if channel is None:
            nchan = self.nchan
            channels = np.arange(nchan)
        else:
            nchan = 1
            channels = [channel, ]

        poly_coeffs = np.zeros((nchan, 10)).tolist()

        if eval:
            lib = np
            model = self.fit
        else:
            lib = tt
            model = self.model
        
        # Parse 'c#' keyword arguments as coefficients
        for j in range(nchan):
            for i in range(10):
                try:
                    if channels[j] == 0:
                        poly_coeffs[j][i] = getattr(model, f'c{i}')
                    else:
                        poly_coeffs[j][i] = getattr(model,
                                                    f'c{i}_{channels[j]}')
                except AttributeError:
                    pass

        poly_flux = lib.zeros(0)
        for c in range(nchan):
            lcpiece = lib.zeros(len(self.time))
            for power in range(len(poly_coeffs[c])):
                lcpiece += poly_coeffs[c][power] * self.time_local**power
            poly_flux = lib.concatenate([poly_flux, lcpiece])

        return poly_flux
