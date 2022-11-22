import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from . import PyMC3Model


class StepModel(PyMC3Model):
    def __init__(self, **kwargs):
        # Inherit from PyMC3Model class
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

    def eval(self, eval=True, channel=None, **kwargs):
        if channel is None:
            nchan = self.nchan
            channels = np.arange(nchan)
        else:
            nchan = 1
            channels = [channel, ]

        steps = np.zeros((self.nchan, 10)).tolist()
        steptimes = np.zeros((self.nchan, 10)).tolist()

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
                        steps[j][i] = getattr(model, f'step{i}')
                        steptimes[j][i] = getattr(model, f'steptime{i}')
                    else:
                        steps[j][i] = getattr(model, f'step{i}_{channels[j]}')
                        steptimes[j][i] = getattr(model,
                                                  f'steptime{i}_{channels[j]}')
                except AttributeError:
                    pass

        poly_flux = lib.zeros(0)
        for c in range(nchan):
            lcpiece = lib.ones(len(self.time))
            for s in np.where(steps[c] != 0)[0]:
                lcpiece += steps[c][s]*(self.time >= steptimes[c][s])
            poly_flux = lib.concatenate([poly_flux, lcpiece])

        return poly_flux
