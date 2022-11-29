import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from . import PyMC3Model


class ExpRampModel(PyMC3Model):
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

        ramp_coeffs = np.zeros((nchan, 6)).tolist()

        if eval:
            lib = np
            model = self.fit
        else:
            lib = tt
            model = self.model

        # Parse 'r#' keyword arguments as coefficients
        for j in range(nchan):
            for i in range(6):
                try:
                    if channels[j] == 0:
                        ramp_coeffs[j][i] = getattr(model, f'r{i}')
                    else:
                        ramp_coeffs[j][i] = getattr(model,
                                                    f'r{i}_{channels[j]}')
                except AttributeError:
                    pass

        ramp_flux = lib.zeros(0)
        for c in range(nchan):
            r0, r1, r2, r3, r4, r5 = ramp_coeffs[c]
            lcpiece = (r0*lib.exp(-r1*self.time_local + r2) +
                       r3*lib.exp(-r4*self.time_local + r5) +
                       1)
            ramp_flux = lib.concatenate([ramp_flux, lcpiece])

        return ramp_flux
