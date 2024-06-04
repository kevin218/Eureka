import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from . import PyMC3Model
from ...lib.split_channels import split


class StepModel(PyMC3Model):
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

        steps = np.zeros((nchan, 10)).tolist()
        steptimes = np.zeros((nchan, 10)).tolist()

        if eval:
            lib = np
            model = self.fit
        else:
            lib = tt
            model = self.model

        # Parse 'c#' keyword arguments as coefficients
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            for i in range(10):
                try:
                    if chan == 0:
                        steps[c][i] = getattr(model, f'step{i}')
                        steptimes[c][i] = getattr(model, f'steptime{i}')
                    else:
                        steps[c][i] = getattr(model, f'step{i}_{chan}')
                        steptimes[c][i] = getattr(model,
                                                  f'steptime{i}_{chan}')
                except AttributeError:
                    pass

        poly_flux = lib.zeros(0)
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            lcpiece = lib.ones(len(time))
            for s in range(10):
                lcpiece += steps[c][s]*(time >= steptimes[c][s])
            poly_flux = lib.concatenate([poly_flux, lcpiece])

        return poly_flux
