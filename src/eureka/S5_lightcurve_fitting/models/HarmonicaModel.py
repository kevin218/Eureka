import numpy as np
try:
    from harmonica import HarmonicaTransit
except ImportError:
    print("Could not import harmonica. Functionality may be limited.")

from .BatmanModels import BatmanTransitModel
from .AstroModel import PlanetParams
from ...lib.split_channels import split


class HarmonicaTransitModel(BatmanTransitModel):
    """Transit Model"""
    def __init__(self, **kwargs):
        """Initialize the transit model

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from BatmanTransitModel class
        super().__init__(**kwargs)
        self.name = 'harmonica transit'
        # Define transit model to be used
        self.transit_model = HarmonicaTransit

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
        lcfinal = np.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            light_curve = np.ma.ones(len(time))
            for pid in pid_iter:
                # Initialize planet
                pl_params = PlanetParams(self, pid, chan)

                # Enforce physicality to avoid crashes from Harmonica by
                # returning something that should be a horrible fit
                if (not ((0 < pl_params.per) and (0 < pl_params.inc <= 90) and
                         (1 < pl_params.a) and (-1 <= pl_params.ecosw <= 1) and
                         (-1 <= pl_params.esinw <= 1))
                    or (self.parameters.limb_dark.value == 'kipping2013' and
                        pl_params.u_original[0] <= 0)):
                    # Returning nans or infs breaks the fits, so this was the
                    # best I could think of
                    light_curve = 1e6*np.ma.ones(time.shape)
                    continue

                # Make the transit model
                ht = self.transit_model(time)
                ht.set_orbit(t0=pl_params.t0,
                             period=pl_params.per,
                             a=pl_params.a,
                             inc=pl_params.inc * np.pi / 180.,
                             ecc=pl_params.ecc,
                             omega=pl_params.w)
                ht.set_stellar_limb_darkening(
                    pl_params.u, limb_dark_law=pl_params.limb_dark)
                ht.set_planet_transmission_string(pl_params.ab)
                light_curve *= ht.get_transit_light_curve()

            lcfinal = np.ma.append(lcfinal, light_curve)

        return lcfinal
