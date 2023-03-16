import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from . import PyMC3Model


class CentroidModel(PyMC3Model):
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

        # Figure out if using xpos, ypos, xwidth, ywidth
        self.axis = kwargs.get('axis')
        self.centroid = kwargs.get('centroid')

        if self.nchan == 1:
            self.coeff_keys = [self.axis, ]
        else:
            self.coeff_keys = [f'{self.axis}_{i}' if i > 0 else self.axis
                               for i in range(self.nchan)]

    @property
    def centroid(self):
        """A getter for the centroid."""
        return self._centroid

    @centroid.setter
    def centroid(self, centroid_array):
        """A setter for the centroid."""
        self._centroid = centroid_array
        if self.centroid is not None:
            # Convert to local centroid
            self.centroid_local = self.centroid - self.centroid.mean()

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

        coeffs = np.zeros(nchan).tolist()

        if eval:
            lib = np
            model = self.fit
        else:
            lib = tt
            model = self.model
        
        for c in range(nchan):
            coeffs[c] = getattr(model, self.coeff_keys[channels[c]])
        
        centroid_flux = lib.zeros(0)
        for c in range(nchan):
            lcpiece = 1 + self.centroid_local*coeffs[c]
            centroid_flux = lib.concatenate([centroid_flux, lcpiece])

        return centroid_flux
