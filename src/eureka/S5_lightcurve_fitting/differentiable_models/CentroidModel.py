import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import starry
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from . import PyMC3Model

starry.config.quiet = True
starry.config.lazy = True


class fit_class:
    def __init__(self):
        pass


class CentroidModel(PyMC3Model):
    def __init__(self, model, **kwargs):
        # Inherit from Model class
        super().__init__(**kwargs)

        self.model = model

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

        # Check for Parameters instance
        self.parameters = kwargs.get('parameters')
        # Set parameters for multi-channel fits
        self.longparamlist = kwargs.get('longparamlist')
        self.nchan = kwargs.get('nchan')
        self.paramtitles = kwargs.get('paramtitles')
        self.uniqueparams = np.unique(self.longparamlist)

        # Figure out if using xpos, ypos, xwidth, ywidth
        self.axis = kwargs.get('axis')
        self.centroid = kwargs.get('centroid')

        if self.nchan == 1:
            self.coeff_keys = [self.axis]
        else:
            self.coeff_keys = [f'{self.axis}_{i}' for i in range(self.nchan)]

    @property
    def centroid(self):
        """A getter for the centroid."""
        return self._centroid

    @centroid.setter
    def centroid(self, centroid_array):
        """A setter for the time."""
        self._centroid = centroid_array
        if self.centroid is not None:
            # Convert to local centroid
            self.centroid_local = self.centroid - self.centroid.mean()

    def eval(self, eval=True, channel=None):
        if channel is None:
            nchan = self.nchan
            channels = np.arange(nchan)
        else:
            nchan = 1
            channels = [channel, ]

        if eval:
            
            centroid_flux = np.zeros(0)
            for chan in range(nchan):
                c = channels[chan]
                coeff = self.fit_dict[self.coeff_keys[c]]
                lcpiece = 1 + self.centroid_local*coeff
                centroid_flux = np.concatenate([centroid_flux, lcpiece])

            return centroid_flux
        else:
            
            centroid_flux = tt.zeros(0)
            for chan in range(nchan):
                c = channels[chan]
                coeff = self.model[self.coeff_keys[c]]
                lcpiece = 1 + self.centroid_local*coeff
                centroid_flux = tt.concatenate([centroid_flux, lcpiece])

            return centroid_flux

    @property
    def fit_dict(self):
        return self._fit_dict

    @fit_dict.setter
    def fit_dict(self, input_fit_dict):
        self._fit_dict = input_fit_dict

        fit = fit_class()
        for key in self.fit_dict.keys():
            setattr(fit, key, self.fit_dict[key])

        for parname in self.uniqueparams:
            param = getattr(self.parameters, parname)
            if param.ptype == 'independent':
                continue
            elif param.ptype == 'fixed':
                setattr(fit, parname, param.value)
            elif param.ptype == 'shared':
                for c in range(1, self.nchan):
                    parname_temp = parname+'_'+str(c)
                    setattr(fit, parname_temp, getattr(fit, parname))
                    self._fit_dict[parname_temp] = getattr(fit, parname)
