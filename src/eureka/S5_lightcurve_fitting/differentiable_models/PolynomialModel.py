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
                for c in range(self.nchan):
                    trim1 = np.nansum(self.mwhites_nexp[:c])
                    trim2 = trim1 + self.mwhites_nexp[c]
                    time = self.time[trim1:trim2]
                    self.time_local.extend(time - time.mean())
                self.time_local = np.array(self.time_local)
            else:
                self.time_local = self.time - self.time.mean()

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

        poly_coeffs = np.zeros((nchan, 10)).tolist()

        if eval:
            lib = np
            model = self.fit
        else:
            lib = tt
            model = self.model
        
        # Parse 'c#' keyword arguments as coefficients
        for c in range(nchan):
            chan = channels[c]
            for i in range(10):
                try:
                    if chan == 0:
                        poly_coeffs[c][i] = getattr(model, f'c{i}')
                    else:
                        poly_coeffs[c][i] = getattr(model,
                                                    f'c{i}_{chan}')
                except AttributeError:
                    pass

        poly_flux = lib.zeros(0)
        for c in range(nchan):
            if self.multwhite:
                chan = channels[c]
                trim1 = np.nansum(self.mwhites_nexp[:chan])
                trim2 = trim1 + self.mwhites_nexp[chan]
                time = self.time_local[trim1:trim2]
            else:
                time = self.time_local

            lcpiece = lib.zeros(len(time))
            for power in range(len(poly_coeffs[c])):
                lcpiece += poly_coeffs[c][power] * time**power
            poly_flux = lib.concatenate([poly_flux, lcpiece])

        return poly_flux
