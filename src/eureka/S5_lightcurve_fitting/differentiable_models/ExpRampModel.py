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


class ExpRampModel(PyMC3Model):
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

    def eval(self, eval=True, channel=None):
        if channel is None:
            nchan = self.nchan
            channels = np.arange(nchan)
        else:
            nchan = 1
            channels = [channel, ]

        if eval:
            # This is only called for things like plotting, so looping
            # doesn't matter as much
            ramp_coeffs = np.zeros((nchan, 6))
            
            # Add fitted parameters
            for j in range(nchan):
                for i in range(6):
                    try:
                        if channels[j] == 0:
                            ramp_coeffs[j, i] = self.fit_dict[f'r{i}']
                        else:
                            ramp_coeffs[j, i] = \
                                self.fit_dict[f'r{i}_{channels[j]}']
                    except KeyError:
                        pass

            ramp_flux = np.zeros(0)
            for c in range(nchan):
                r0, r1, r2, r3, r4, r5 = ramp_coeffs[c]
                lcpiece = (r0*np.exp(-r1*self.time_local + r2) +
                           r3*np.exp(-r4*self.time_local + r5) +
                           1)
                ramp_flux = np.concatenate([ramp_flux, lcpiece])

            return ramp_flux
        else:
            # This gets compiled before fitting, so looping doesn't matter
            ramp_coeffs = np.zeros((nchan, 6)).tolist()

            # Parse 'r#' keyword arguments as coefficients
            for j in range(nchan):
                for i in range(6):
                    try:
                        if channels[j] == 0:
                            ramp_coeffs[j][i] = getattr(self.model, f'r{i}')
                        else:
                            ramp_coeffs[j][i] = getattr(self.model,
                                                        f'r{i}_{channels[j]}')
                    except AttributeError:
                        pass

            ramp_flux = tt.zeros(0)
            for c in range(nchan):
                r0, r1, r2, r3, r4, r5 = ramp_coeffs[c]
                lcpiece = (r0*tt.exp(-r1*self.time_local + r2) +
                           r3*tt.exp(-r4*self.time_local + r5) +
                           1)
                ramp_flux = tt.concatenate([ramp_flux, lcpiece])

            return ramp_flux

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
