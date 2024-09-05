import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from .PyMC3Models import PyMC3Model
# Importing these here to give access to other differentiable models
from ..models.AstroModel import PlanetParams, get_ecl_midpt, true_anomaly  # NOQA: F401, E501
from ...lib.split_channels import split


class AstroModel(PyMC3Model):
    """A model that combines all astrophysical components."""
    def __init__(self, components, **kwargs):
        """Initialize the phase curve model.

        Parameters
        ----------
        components : list
            A list of eureka.S5_lightcurve_fitting.models.Model which together
            comprise the astrophysical model.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from Model class
        super().__init__(components=components, **kwargs)
        self.name = 'astrophysical model'

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

    @property
    def components(self):
        """A getter for the flux."""
        return self._components

    @components.setter
    def components(self, components):
        """A setter for the flux

        Parameters
        ----------
        flux_array : sequence
            The flux array
        """
        self._components = components
        self.starry_model = None
        self.phasevariation_models = []
        self.stellar_models = []
        for component in self.components:
            if 'starry' in component.name.lower():
                self.starry_model = component
            elif 'phase curve' in component.name.lower():
                self.phasevariation_models.append(component)
            else:
                self.stellar_models.append(component)

    def eval(self, channel=None, eval=True, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            The value of the model at the times self.time.
        """
        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel, ]

        # Currently can't separate starry models (given mutual occultations)
        pid_iter = range(self.num_planets)

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        if eval:
            lib = np.ma
        else:
            lib = tt

        # Set all parameters
        lcfinal = lib.zeros(0)
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            starFlux = lib.ones(len(time))
            for component in self.stellar_models:
                starFlux *= component.eval(channel=chan, eval=eval, **kwargs)
            if self.starry_model is not None:
                result = self.starry_model.eval(channel=chan, eval=eval,
                                                piecewise=True, **kwargs)[0]
                transits = result.pop(0)
                eclipses = result
                starFlux *= transits

            planetFluxes = lib.zeros(len(time))
            for pid in pid_iter:
                # Initial default value
                planetFlux = 0

                if self.starry_model is not None:
                    # User is fitting an eclipse model
                    planetFlux = eclipses[pid]
                elif len(self.phasevariation_models) > 0:
                    # User is dealing with phase variations of a
                    # non-eclipsing object
                    planetFlux = lib.ones(len(time))

                for model in self.phasevariation_models:
                    planetFlux *= model.eval(channel=chan, pid=pid, eval=eval,
                                             **kwargs)

                planetFluxes += planetFlux

            lcfinal = lib.concatenate([lcfinal, starFlux+planetFluxes])
        return lcfinal
