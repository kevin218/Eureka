import numpy as np
try:
    import harmonica.jax as harmonica
    import jax.numpy as jnp
except (ImportError, AttributeError):
    # harmonica.jax is currently broken, so don't throw an error
    # There's also already a warning printed in the models.harmonica
    # file in the more general case of harmonica not being installed.
    # We'll throw an error later if folks try to use this model
    pass

from ..models.BatmanModels import BatmanTransitModel
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
        raise NotImplementedError('There is currently a bug with the '
                                  'harmonica.jax package. Once that is '
                                  'resolved, we will enable Eureka!\'s '
                                  'jax_models.HarmonicaTransitModel.')

        # Inherit from BatmanTransitModel class
        super().__init__(**kwargs)
        self.name = 'harmonica transit'
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

                if self.parameters.limb_dark.value in ['uniform', 'linear',
                                                       'quadratic',
                                                       'kipping2013']:
                    light_curve *= harmonica.harmonica_transit_quad_ld(
                        time, t0=pl_params.t0, period=pl_params.per,
                        a=pl_params.a, inc=pl_params.inc, ecc=pl_params.ecc,
                        omega=pl_params.w, u1=pl_params.u1, u2=pl_params.u2,
                        r=pl_params.ab)
                elif self.parameters.limb_dark.value == 'nonlinear':
                    light_curve *= harmonica.harmonica_transit_nonlinear_ld(
                        time, t0=pl_params.t0, period=pl_params.per,
                        a=pl_params.a, inc=pl_params.inc, ecc=pl_params.ecc,
                        omega=pl_params.w, u1=pl_params.u1, u2=pl_params.u2,
                        u3=pl_params.u3, u4=pl_params.u4, r=pl_params.ab)
                else:
                    raise NotImplementedError(
                        'The requested limb-darkening model '
                        f'{self.parameters.limb_dark.value} is not supported '
                        'by the HarmonicaTransitModel.')

            lcfinal = lib.append(lcfinal, light_curve)

        return lcfinal
