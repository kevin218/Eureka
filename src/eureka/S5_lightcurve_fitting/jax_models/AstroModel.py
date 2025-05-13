import jax
import jax.numpy as jnp

from .JaxModel import JaxModel
# Importing these here to give access to other differentiable models
from ..models.AstroModel import PlanetParams, get_ecl_midpt, true_anomaly  # NOQA: F401, E501
from ...lib.split_channels import split

jax.config.update("jax_enable_x64", True)


class AstroModel(JaxModel):
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
        """A getter for the components."""
        return self._components

    @components.setter
    def components(self, components):
        """A setter for the components

        Parameters
        ----------
        components : sequence
            The collection of astrophysical model components.
        """
        self._components = components
        self.jaxoplanet_model = None
        self.phasevariation_models = []
        self.stellar_models = []
        for component in self.components:
            if 'jaxoplanet' in component.name.lower():
                self.jaxoplanet_model = component
            elif 'phase curve' in component.name.lower():
                self.phasevariation_models.append(component)
            else:
                self.stellar_models.append(component)

    @property
    def fit(self):
        """A getter for the fit object."""
        return self._fit

    @fit.setter
    def fit(self, fit):
        """A setter for the fit object.

        Parameters
        ----------
        fit : object
            The fit object
        """
        self._fit = fit
        for component in self.components:
            component.fit = fit

    def eval(self, channel=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
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

        # Can't separately evaluate jaxoplanet models to allow for
        # mutual occultations
        pid_iter = range(self.num_planets)

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Set all parameters
        lcfinal = jnp.zeros(0)
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            starFlux = jnp.ones(len(time))
            for component in self.stellar_models:
                starFlux *= component.eval(channel=chan, **kwargs)
            if self.jaxoplanet_model is not None:
                starFlux *= self.jaxoplanet_model.eval(channel=chan,
                                                       **kwargs)

            planetFluxes = jnp.zeros(len(time))
            for pid in pid_iter:
                # Initial default value
                planetFlux = 0

                # Placeholder code for planet emitted/reflected fluxes

                planetFluxes += planetFlux

            lcfinal = jnp.concatenate([lcfinal, starFlux+planetFluxes])
        return lcfinal
