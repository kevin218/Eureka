import numpy as np
try:
    import fleck.jax as fleck
    import jax.numpy as jnp
except ImportError:
    print("Could not import fleck.jax. Functionality may be limited.")

from .. import models as m
from .AstroModel import PlanetParams
from ...lib.split_channels import split


class FleckTransitModel(m.FleckTransitModel):
    """Transit Model with Star Spots"""
    def __init__(self, **kwargs):
        """Initialize the fleck model

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        raise NotImplementedError('There is currently a bug with the fleck.jax'
                                  ' package which prohibts the use of spot '
                                  'contrasts.\nOnce that is resolved, we will'
                                  'enable Eureka!\'s '
                                  'jax_models.FleckTransitModel.')

        # Inherit from models.FleckTransitModel class
        super().__init__(**kwargs)
        self.name = 'fleck transit'
        # Define transit model to be used
        self.transit_model = None

    def eval(self, eval=True, channel=None, pid=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
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

        if eval:
            lib = np
        else:
            lib = jnp

        # Set all parameters
        lcfinal = lib.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            # create arrays to hold values
            spotrad = lib.array([])
            spotlat = lib.array([])
            spotlon = lib.array([])
            spotcon = lib.array([])
            pl_params = PlanetParams(self, 0, chan, eval=eval)
            spotcon0 = getattr(pl_params, 'spotcon')
            for n in range(pl_params.nspots):
                # read radii, latitudes, longitudes, and contrasts
                if n > 0:
                    spot_id = f'{n}'
                else:
                    spot_id = ''
                spotrad = lib.concatenate([
                    spotrad, [getattr(pl_params, f'spotrad{spot_id}'),]])
                spotlat = lib.concatenate([
                    spotlat, [getattr(pl_params, f'spotlat{spot_id}'),]])
                spotlon = lib.concatenate([
                    spotlon, [getattr(pl_params, f'spotlon{spot_id}'),]])
                # If spotcon# isn't set, default to spotcon (from channel 0)
                spotcon = lib.concatenate([
                    spotcon, [getattr(pl_params, f'spotcon{spot_id}',
                                      spotcon0),]])

            if lib.any((lib.abs(spotlat) > 90) | (lib.abs(spotlon) > 180) |
                       (spotrad > 1)):
                # Returning nans or infs breaks the fits, so this was the
                # best I could think of
                return 1e6*lib.ones(time.shape)

            light_curve = lib.ones(len(time))
            for pid in pid_iter:
                # Initialize planet
                pl_params = PlanetParams(self, pid, chan, eval=eval)

                # Enforce physicality to avoid crashes from Harmonica by
                # returning something that should be a horrible fit
                if (not ((0 < pl_params.per) and (0 < pl_params.inc <= 90) and
                         (1 < pl_params.a) and (-1 <= pl_params.ecosw <= 1) and
                         (-1 <= pl_params.esinw <= 1))
                    or (self.parameters.limb_dark.value == 'kipping2013' and
                        pl_params.u_original[0] <= 0)):
                    # Returning nans or infs breaks the fits, so this was the
                    # best I could think of
                    light_curve = 1e6*lib.ones(time.shape)
                    continue

                inverse = False
                if pl_params.rp < 0:
                    # The planet's radius is negative, so need to do some
                    # tricks to avoid errors
                    inverse = True
                    pl_params.rp *= -1

                # Make the star object (fleck uses radians, not degrees)
                star = fleck.ActiveStar(
                    times=time, lon=spotlon*np.pi/180,
                    lat=spotlat*np.pi/180, rad=spotrad,
                    contrast=spotcon,
                    inclination=pl_params.spotstari*np.pi/180,
                    P_rot=pl_params.spotrot)

                # Compute the lightcurve (fleck uses radians, not degrees)
                lc, _, _, _ = star.transit_model(
                    t0=pl_params.t0, period=pl_params.per, rp=pl_params.rp,
                    a=pl_params.a, inclination=pl_params.inc*np.pi/180,
                    omega=pl_params.w*np.pi/180, ecc=pl_params.ecc,
                    u1=pl_params.u1, u2=pl_params.u2)

                # Invert the transit feature if rp<0
                if inverse:
                    lc = 2. - lc

                # Apply the current lightcurve
                light_curve *= lc

            lcfinal = lib.append(lcfinal, light_curve)

        return lcfinal
