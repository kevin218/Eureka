import numpy as np

from .Model import Model
from .AstroModel import PlanetParams, get_ecl_midpt, true_anomaly
from ...lib.split_channels import split


class QuasiLambertianPhaseCurve(Model):
    """Quasi-Lambertian phase curve based on Agol+2007 for airless planets."""
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
        super().__init__(**kwargs,
                         name='quasi-lambertian phase curve',
                         modeltype='physical')

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

                if pl_params.quasi_gamma == 0:
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

                # calculate the phase variations
                phaseVars = np.abs(np.cos(
                                   (phi+pl_params.quasi_offset*np.pi/180)/2)
                                   )**pl_params.quasi_gamma

            lcfinal = np.ma.append(lcfinal, phaseVars)

        return lcfinal
