import numpy as np
try:
    import batman
except ImportError:
    print("Could not import batman. Functionality may be limited.")

from .Model import Model
from ...lib.readEPF import Parameters


class SinusoidPhaseCurveModel(Model):
    """A sinusoidal phase curve model"""
    def __init__(self, transit_model=None, eclipse_model=None, **kwargs):
        """Initialize the phase curve model.

        Parameters
        ----------
        transit_model : eureka.S5_lightcurve_fitting.models.Model; optional
            The transit model to use for this phase curve model.
            Defaults to None.
        eclipse_model : eureka.S5_lightcurve_fitting.models.Model; optional
            The eclipse model to use for this phase curve model.
            Defaults to None.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from Model calss
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        # Check for Parameters instance
        self.parameters = kwargs.get('parameters')

        # Generate parameters from kwargs if necessary
        if self.parameters is None:
            self.parameters = Parameters(**kwargs)

        # Set parameters for multi-channel fits
        self.longparamlist = kwargs.get('longparamlist')
        self.nchan = kwargs.get('nchan')
        self.paramtitles = kwargs.get('paramtitles')

        self.components = None
        self.transit_model = transit_model
        self.eclipse_model = eclipse_model
        if transit_model is not None:
            self.components = [self.transit_model, ]
        if eclipse_model is not None:
            if self.components is None:
                self.components = [self.eclipse_model, ]
            else:
                self.components.append(self.eclipse_model)

    @property
    def time(self):
        """A getter for the time."""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time."""
        self._time = time_array
        if self.transit_model is not None:
            self.transit_model.time = time_array
        if self.eclipse_model is not None:
            self.eclipse_model.time = time_array

    def update(self, newparams, names, **kwargs):
        """Update the model with new parameter values.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        names : list
            Parameter names.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.update().
        """
        super().update(newparams, names, **kwargs)
        if self.transit_model is not None:
            self.transit_model.update(newparams, names, **kwargs)
        if self.eclipse_model is not None:
            self.eclipse_model.update(newparams, names, **kwargs)

    def eval(self, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            The value of the model at the times self.time.
        """
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Initialize model
        bm_params = batman.TransitParams()
        pc_params = {'AmpCos1': 0., 'AmpSin1': 0.,
                     'AmpCos2': 0., 'AmpSin2': 0.}

        # Set all parameters
        lcfinal = np.array([])
        for c in np.arange(self.nchan):
            # Set all parameters
            for index, item in enumerate(self.longparamlist[c]):
                if np.any([key in item for key in pc_params.keys()]):
                    pc_params[self.paramtitles[index]] = \
                        self.parameters.dict[item][0]
                else:
                    setattr(bm_params, self.paramtitles[index],
                            self.parameters.dict[item][0])

            bm_params.limb_dark = 'uniform'
            bm_params.u = []

            m_transit = None
            if not np.any(['t_secondary' in key
                           for key in self.longparamlist[c]]):
                # If not explicitly fitting for the time of eclipse, get the
                # time of eclipse from the time of transit, period,
                # eccentricity, and argument of periastron
                m_transit = batman.TransitModel(bm_params, self.time,
                                                transittype='primary')
                t_secondary = m_transit.get_t_secondary(bm_params)
            else:
                t_secondary = self.parameters.dict['t_secondary'][0]

            if self.parameters.dict['ecc'][0] == 0.:
                # the planet is on a circular orbit
                t = self.time - t_secondary
                freq = 2.*np.pi/self.parameters.dict['per'][0]
                phi = (freq*t)
            else:
                if m_transit is None:
                    # Avoid overhead of making a new transit model if avoidable
                    m_transit = batman.TransitModel(bm_params, self.time,
                                                    transittype='primary')
                anom = m_transit.get_true_anomaly()
                w = self.parameters.dict['w'][0]
                # the planet is on an eccentric orbit
                phi = anom + w*np.pi/180. + np.pi/2.

            # calculate the phase variations
            if pc_params['AmpCos2'] == 0. and pc_params['AmpSin2'] == 0.:
                # Skip multiplying by a bunch of zeros to speed up fitting
                phaseVars = (1. + pc_params['AmpCos1']*(np.cos(phi)-1.) +
                             pc_params['AmpSin1']*np.sin(phi))
            else:
                phaseVars = (1. + pc_params['AmpCos1']*(np.cos(phi)-1.) +
                             pc_params['AmpSin1']*np.sin(phi) +
                             pc_params['AmpCos2']*(np.cos(2.*phi)-1.) +
                             pc_params['AmpSin2']*np.sin(2.*phi))

            lcfinal = np.append(lcfinal, phaseVars)

        transit = self.transit_model.eval()
        eclipse = self.eclipse_model.eval()

        return transit + lcfinal*(eclipse-1)
