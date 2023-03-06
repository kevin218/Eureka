import numpy as np
try:
    import batman
except ImportError:
    print("Could not import batman. Functionality may be limited.")

from .Model import Model


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

        # Inherit from Model calss
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        # Check if should enforce positivity
        if not hasattr(self, 'force_positivity'):
            self.force_positivity = False

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
        if self.transit_model is not None:
            self.transit_model.update(newparams, **kwargs)
        if self.eclipse_model is not None:
            self.eclipse_model.update(newparams, **kwargs)

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

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Initialize model
        bm_params = batman.TransitParams()
        pc_params = {'AmpCos1': 0., 'AmpSin1': 0.,
                     'AmpCos2': 0., 'AmpSin2': 0.}

        # Set all parameters
        lcfinal = np.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            # Set all parameters
            for index, item in enumerate(self.longparamlist[chan]):
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
                           for key in self.longparamlist[chan]]):
                # If not explicitly fitting for the time of eclipse, get the
                # time of eclipse from the time of transit, period,
                # eccentricity, and argument of periastron
                m_transit = batman.TransitModel(bm_params, self.time,
                                                transittype='primary')
                t_secondary = m_transit.get_t_secondary(bm_params)
            else:
                t_secondary = self.parameters.dict['t_secondary'][0]

            if self.multwhite:
                trim1 = np.nansum(self.mwhites_nexp[:channels[c]])
                trim2 = trim1 + self.mwhites_nexp[channels[c]]
                time = self.time[trim1:trim2]
            else:
                time = self.time

            if self.parameters.dict['ecc'][0] == 0.:
                # the planet is on a circular orbit
                t = time - t_secondary
                freq = 2.*np.pi/self.parameters.dict['per'][0]
                phi = (freq*t)
            else:
                if m_transit is None:
                    # Avoid overhead of making a new transit model if avoidable
                    m_transit = batman.TransitModel(bm_params, time,
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

            # If requested, force positive phase variations
            if self.force_positivity and np.any(phaseVars < 0):
                # Returning nans or infs breaks the fits, so this was
                # the best I could think of
                phaseVars = 1e12*np.ones_like(time)

            lcfinal = np.append(lcfinal, phaseVars)

        if self.transit_model is None:
            transit = 1
        else:
            transit = self.transit_model.eval()
        if self.eclipse_model is None:
            eclipse = 1
        else:
            eclipse = self.eclipse_model.eval()

        return transit + lcfinal*(eclipse-1)
