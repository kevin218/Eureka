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
        # Needed before setting time
        self.multwhite = kwargs.get('multwhite')
        self.mwhites_nexp = kwargs.get('mwhites_nexp')

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
            if self.multwhite:
                self.time_local = []
                for c in np.arange(self.nchan):
                    trim1 = np.nansum(self.mwhites_nexp[:c])
                    trim2 = trim1 + self.mwhites_nexp[c]
                    time = self.time[trim1:trim2]
                    self.time_local.extend(time - time[0])
                self.time_local = np.array(self.time_local)
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
            nchan = self.nchan
            channels = np.arange(nchan)
        else:
            nchan = 1
            channels = [channel, ]

        ramp_coeffs = np.zeros((nchan, 12)).tolist()

        if eval:
            lib = np
            model = self.fit
        else:
            lib = tt
            model = self.model

        # Parse 'r#' keyword arguments as coefficients
        for c in range(nchan):
            chan = channels[c]
            for i in range(12):
                try:
                    if chan == 0:
                        ramp_coeffs[c][i] = getattr(model, f'r{i}')
                    else:
                        ramp_coeffs[c][i] = getattr(model,
                                                    f'r{i}_{chan}')
                except AttributeError:
                    pass

        ramp_flux = lib.zeros(0)
        for c in range(nchan):
            if self.multwhite:
                chan = channels[c]
                trim1 = np.nansum(self.mwhites_nexp[:chan])
                trim2 = trim1 + self.mwhites_nexp[chan]
                time = self.time_local[trim1:trim2]
            else:
                time = self.time_local

            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11 = ramp_coeffs[c]
            lcpiece = (r0*lib.exp(-r1*time + r2) +
                       r3*lib.exp(-r4*time + r5) +
                       r6*lib.exp(-r7*time + r8) +
                       r9*lib.exp(-r10*time + r11) +
                       1)
            ramp_flux = lib.concatenate([ramp_flux, lcpiece])

        return ramp_flux
