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
        # Needed before setting time
        self.multwhite = kwargs.get('multwhite')
        self.mwhites_nexp = kwargs.get('mwhites_nexp')

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
        self._centroid = np.ma.masked_invalid(centroid_array)
        if self.centroid is not None:
            # Convert to local centroid
            if self.multwhite:
                self.centroid_local = []
                for c in np.arange(self.nchan):
                    trim1 = np.nansum(self.mwhites_nexp[:c])
                    trim2 = trim1 + self.mwhites_nexp[c]
                    centroid = self.centroid[trim1:trim2]
                    self.centroid_local.extend(centroid - centroid.mean())
                self.centroid_local = np.array(self.centroid_local)
            else:
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
            if self.multwhite:
                chan = channels[c]
                trim1 = np.nansum(self.mwhites_nexp[:chan])
                trim2 = trim1 + self.mwhites_nexp[chan]
                centroid = self.centroid_local[trim1:trim2]
            else:
                centroid = self.centroid_local

            lcpiece = 1 + centroid*coeffs[c]
            centroid_flux = lib.concatenate([centroid_flux, lcpiece])

        return centroid_flux
