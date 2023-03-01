import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from . import PyMC3Model


class SinusoidPhaseCurveModel(PyMC3Model):
    def __init__(self, starry_model=None, **kwargs):
        """Initialize the model.

        Parameters
        ----------
        transit_model : eureka.S5_lightcurve_fitting.differentiable_models.StarryModel
            The starry model to combined with this phase curve model.
            Defaults to None.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.differentiable_models.PyMC3Model.__init__().
        """  # NOQA: E501
        self.starry_model = starry_model

        # Inherit from PyMC3Model class
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        self.components = []
        if self.starry_model is not None:
            self.components.append(self.starry_model)

        orders = [int(key[6:]) for key in self.paramtitles
                  if 'AmpCos' in key or 'AmpSin' in key]
        if len(orders) > 0:
            self.maxOrder = np.max(orders)
        else:
            raise AssertionError('There are no AmpCos or AmpSin parameters to'
                                 'fit. Either remove sinusoid_pc or add some'
                                 'AmpCos or AmpSin terms to fit.')

    @property
    def time(self):
        """A getter for the time."""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time."""
        self._time = time_array
        if self.starry_model is not None:
            self.starry_model.time = time_array

    @property
    def model(self):
        """A getter for the model."""
        return self._model

    @model.setter
    def model(self, model):
        """A setter for the model."""
        self._model = model
        if self.starry_model is not None:
            self.starry_model.model = model

    def setup(self, **kwargs):
        super().setup(**kwargs)
        if self.starry_model is not None:
            self.starry_model.setup(**kwargs)

    def update(self, newparams, **kwargs):
        """Update the model with new parameter values.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.update().
        """
        super().update(newparams, **kwargs)
        if self.starry_model is not None:
            self.starry_model.update(newparams, **kwargs)

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

        if eval:
            lib = np
            model = self.fit
            if self.starry_model is not None:
                systems = self.starry_model.fit.systems
        else:
            lib = tt
            model = self.model
            if self.starry_model is not None:
                systems = self.starry_model.systems

        lcfinal = lib.zeros(0)
        for c in range(nchan):
            if self.multwhite:
                chan = channels[c]
                trim1 = np.nansum(self.mwhites_nexp[:chan])
                trim2 = trim1 + self.mwhites_nexp[chan]
                time = self.time[trim1:trim2]
            else:
                time = self.time

            phaseVars = lib.ones(len(time))
            
            # Compute orbital phase
            # FINDME: for now this assumes a circular orbit
            t = time - model.t0 - model.per/2.
            phi = 2.*np.pi/model.per*t

            for order in range(1, self.maxOrder+1):
                if self.nchannel_fitted == 1 or chan == 0:
                    suffix = ''
                else:
                    suffix = f'_{chan}'
                AmpCos = getattr(model, f'AmpCos{order}{suffix}', 0)
                AmpSin = getattr(model, f'AmpSin{order}{suffix}', 0)
                phaseVars += (AmpCos*(lib.cos(order*phi)-1.) +
                              AmpSin*lib.sin(order*phi))

            if self.starry_model is not None:
                # Combine with the starry model
                flux_star, flux_planet = systems[chan].flux(time,
                                                            total=False)
                lcpiece = flux_star + flux_planet*phaseVars
                if eval:
                    # Evaluate if needed
                    lcpiece = lcpiece.eval()
            else:
                lcpiece = phaseVars

            lcfinal = lib.concatenate([lcfinal, lcpiece])

        return lcfinal
