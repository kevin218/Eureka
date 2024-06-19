import numpy as np

from .Model import Model
from .AstroModel import PlanetParams, get_ecl_midpt, true_anomaly
from ...lib.split_channels import split


class SinusoidPhaseCurveModel(Model):
    """A sinusoidal phase curve model"""
    def __init__(self, **kwargs):
        """Initialize the phase curve model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from Model class
        super().__init__(**kwargs)
        self.name = 'sinusoid phase curve'

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        # Set default to not force positivity
        self.force_positivity = getattr(self, 'force_positivity', False)

    def eval(self, channel=None, pid=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        pid : int; optional
            Planet ID, default is None which combines the models from
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

        if pid is None:
            pid_iter = range(self.num_planets)
        else:
            pid_iter = [pid,]

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

            for pid in pid_iter:
                # Initialize model
                pl_params = PlanetParams(self, pid, chan)

                if (pl_params.AmpCos1 == 0 and pl_params.AmpSin1 == 0 and
                        pl_params.AmpCos2 == 0 and pl_params.AmpSin2 == 0):
                    # Don't waste time running the following code
                    phaseVars = np.ma.ones_like(time)
                    continue

                if pl_params.t_secondary is None:
                    # If not explicitly fitting for the time of eclipse, get
                    # the time of eclipse from the time of transit, period,
                    # eccentricity, and argument of periastron
                    pl_params.t_secondary = get_ecl_midpt(pl_params)

                if pl_params.ecc == 0.:
                    # the planet is on a circular orbit
                    t = time - pl_params.t_secondary
                    phi = 2*np.pi/pl_params.per*t
                else:
                    # the planet is on an eccentric orbit
                    anom = true_anomaly(pl_params, time)
                    phi = anom + pl_params.w*np.pi/180 + np.pi/2

                if self.force_positivity:
                    # Check a finely sampled phase range
                    phi2 = np.linspace(0, 2*np.pi, 1000)

                # calculate the phase variations
                if pl_params.AmpCos2 == 0 and pl_params.AmpSin2 == 0:
                    # Skip multiplying by a bunch of zeros to speed up fitting
                    phaseVars = (1 +
                                 pl_params.AmpCos1*(np.ma.cos(phi)-1) +
                                 pl_params.AmpSin1*np.ma.sin(phi))
                    if self.force_positivity:
                        phaseVars2 = (1 +
                                      pl_params.AmpCos1*(np.ma.cos(phi2)-1) +
                                      pl_params.AmpSin1*np.ma.sin(phi2))
                else:
                    phaseVars = (1 +
                                 pl_params.AmpCos1*(np.ma.cos(phi)-1) +
                                 pl_params.AmpSin1*np.ma.sin(phi) +
                                 pl_params.AmpCos2*(np.ma.cos(2*phi)-1) +
                                 pl_params.AmpSin2*np.ma.sin(2*phi))
                    if self.force_positivity:
                        phaseVars2 = (1 +
                                      pl_params.AmpCos1*(np.ma.cos(phi2)-1) +
                                      pl_params.AmpSin1*np.ma.sin(phi2) +
                                      pl_params.AmpCos2*(np.ma.cos(2*phi2)-1) +
                                      pl_params.AmpSin2*np.ma.sin(2*phi2))

                # If requested, force positive phase variations
                if self.force_positivity and np.ma.any(phaseVars2 <= 0):
                    # Returning nans or infs breaks the fits, so this was
                    # the best I could think of
                    phaseVars = 1e6*np.ma.ones(time.shape)

            lcfinal = np.ma.append(lcfinal, phaseVars)

        return lcfinal
