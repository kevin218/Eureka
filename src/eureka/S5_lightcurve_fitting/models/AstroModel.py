import numpy as np
from copy import deepcopy

from .Model import Model
from ...lib.split_channels import split


class AstroModel(Model):
    """A model which combines all astrophysical components."""
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
        self.transit_models = []
        self.eclipse_models = []
        self.phasevariation_models = []
        for component in self.components:
            if 'transit' in component.name.lower():
                self.transit_models.append(component)
            elif 'eclipse' in component.name.lower():
                self.eclipse_models.append(component)
            elif 'phase curve' in component.name.lower():
                self.phasevariation_models.append(component)

    def eval(self, channel=None, pid=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        pid : int; optional
            Planet ID, default is None which combines the eclipse models from
            all planets.
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

        pid_input = deepcopy(pid)
        if pid_input is None:
            pid_iter = range(self.num_planets)
        else:
            pid_iter = [pid_input,]

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Set all parameters
        lcplanet = np.ma.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            light_curve = np.ma.zeros(time.shape)
            for pid in pid_iter:
                phaseVars = 1
                for model in self.phasevariation_models:
                    phaseVars *= model.eval(channel=chan, pid=pid, **kwargs)

                eclipse = 1
                for model in self.eclipse_models:
                    eclipse *= model.eval(channel=chan, pid=pid, **kwargs)-1

                light_curve += eclipse*phaseVars

            lcplanet = np.ma.append(lcplanet, light_curve)

        transit = np.ones_like(lcplanet)
        for model in self.transit_models:
            transit *= model.eval(channel=channel, pid=pid_input, **kwargs)

        stellarModels = np.ones_like(lcplanet)
        for component in self.components:
            if component not in [*self.transit_models, *self.eclipse_models,
                                 *self.phasevariation_models]:
                stellarModels *= component.eval(
                    channel=channel, pid=pid_input, **kwargs)

        return transit*stellarModels + lcplanet
