import numpy as np
try:
    import batman
except ImportError:
    print("Could not import batman. Functionality may be limited.")

from .Model import Model
from .BatmanModels import PlanetParams
from ...lib.split_channels import split


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
        time_array = np.ma.masked_array(time_array)
        self._time = time_array
        if self.transit_model is not None:
            self.transit_model.time = time_array
        if self.eclipse_model is not None:
            self.eclipse_model.time = time_array

    @property
    def nints(self):
        """A getter for the nints."""
        return self._nints

    @nints.setter
    def nints(self, nints_array):
        """A setter for the nints."""
        self._nints = nints_array
        if self.transit_model is not None:
            self.transit_model.nints = nints_array
        if self.eclipse_model is not None:
            self.eclipse_model.nints = nints_array

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

        # Set all parameters
        lcfinal = np.ma.array([])
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
            for pid in range(self.num_planets):
                # Initialize model
                bm_params = PlanetParams(self, pid, chan)

                bm_params.limb_dark = 'uniform'
                bm_params.u = []

                m_transit = None
                if not np.any(['t_secondary' in key
                              for key in self.longparamlist[chan]]):
                    # If not explicitly fitting for the time of eclipse, get
                    # the time of eclipse from the time of transit, period,
                    # eccentricity, and argument of periastron
                    m_transit = batman.TransitModel(bm_params, time,
                                                    transittype='primary')
                    t_secondary = m_transit.get_t_secondary(bm_params)
                else:
                    t_secondary = self.parameters.dict['t_secondary'][0]

                if bm_params.ecc == 0.:
                    # the planet is on a circular orbit
                    t = time - t_secondary
                    freq = 2.*np.pi/bm_params.per
                    phi = (freq*t)
                else:
                    # the planet is on an eccentric orbit
                    if m_transit is None:
                        # Avoid overhead of making a new transit model,
                        # if avoidable
                        m_transit = batman.TransitModel(bm_params, time,
                                                        transittype='primary')
                    anom = m_transit.get_true_anomaly()
                    w = bm_params.w
                    phi = anom + w*np.pi/180. + np.pi/2.

                # calculate the phase variations
                if bm_params.AmpCos2 == 0. and bm_params.AmpSin2 == 0.:
                    # Skip multiplying by a bunch of zeros to speed up fitting
                    phaseVars = (1. + bm_params.AmpCos1*(np.ma.cos(phi)-1.) +
                                 bm_params.AmpSin1*np.ma.sin(phi))
                else:
                    phaseVars = (1. + bm_params.AmpCos1*(np.ma.cos(phi)-1.) +
                                 bm_params.AmpSin1*np.ma.sin(phi) +
                                 bm_params.AmpCos2*(np.ma.cos(2.*phi)-1.) +
                                 bm_params.AmpSin2*np.ma.sin(2.*phi))

                # If requested, force positive phase variations
                if self.force_positivity and np.ma.any(phaseVars < 0):
                    # Returning nans or infs breaks the fits, so this was
                    # the best I could think of
                    phaseVars = 1e12*np.ma.ones(time.shape)

                if self.eclipse_model is None:
                    eclipse = 1
                else:
                    eclipse = self.eclipse_model.eval(channel=chan,
                                                      pid=pid) - 1
                light_curve += eclipse*phaseVars

            lcfinal = np.ma.append(lcfinal, light_curve)

        if self.transit_model is None:
            transit = 1
        else:
            transit = self.transit_model.eval(channel=channel)

        return transit + lcfinal
