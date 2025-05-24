import numpy as np
import inspect
try:
    import harmonica.jax as harmonica
    import jax.numpy as jnp
except (ImportError, AttributeError):
    # harmonica.jax is currently broken, so don't throw an error
    # There's also already a warning printed in the models.harmonica
    # file in the more general case of harmonica not being installed.
    # We'll throw an error later if folks try to use this model
    pass

from . import JaxModel
from .AstroModel import PlanetParams
from ..limb_darkening_fit import ld_profile
from ...lib.split_channels import split


class HarmonicaTransitModel(JaxModel):
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
        # Inherit from Model class
        super().__init__(**kwargs)
        self.name = 'harmonica transit'

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        log = kwargs.get('log')

        # Store the ld_profile
        self.ld_from_S4 = kwargs.get('ld_from_S4')
        ld_func = ld_profile(self.parameters.limb_dark.value,
                             use_gen_ld=self.ld_from_S4)
        len_params = len(inspect.signature(ld_func).parameters)
        self.coeffs = ['u{}'.format(n) for n in range(1, len_params)]

        self.ld_from_file = kwargs.get('ld_from_file')

        # Replace u parameters with generated limb-darkening values
        if self.ld_from_S4 or self.ld_from_file:
            log.writelog("Using the following limb-darkening values:")
            self.ld_array = kwargs.get('ld_coeffs')
            for c in range(self.nchannel_fitted):
                chan = self.fitted_channels[c]
                if self.ld_from_S4:
                    ld_array = self.ld_array[len_params-2]
                else:
                    ld_array = self.ld_array
                for u in self.coeffs:
                    index = np.where(np.array(self.paramtitles) == u)[0]
                    if len(index) != 0:
                        item = self.longparamlist[c][index[0]]
                        param = int(item.split('_')[0][-1])
                        ld_val = ld_array[chan][param-1]
                        log.writelog(f"{item}: {ld_val}")
                        # Use the file value as the starting guess
                        self.parameters.dict[item][0] = ld_val
                        # In a normal prior, center at the file value
                        if (self.parameters.dict[item][-1] == 'N' and
                                self.recenter_ld_prior):
                            self.parameters.dict[item][-3] = ld_val
                        # Update the non-dictionary form as well
                        setattr(self.parameters, item,
                                self.parameters.dict[item])

        # Determine how many Fourier terms to keep (a1, b1, a2, b2, ...)
        # We'll store this as `self.max_fourier_term` to be used in eval
        self.max_fourier_term = 1
        for term in ['a3', 'b3', 'a2', 'b2', 'a1', 'b1']:
            if any(term in p for p in self.paramtitles):
                self.max_fourier_term = 2 * int(term[1])+1  # e.g., "a3" → 2*3+1 = 7
                break  # Keep highest used term only
        print(f"Using {self.max_fourier_term} Fourier terms in the model.")

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

        if eval:
            lib = np
        else:
            lib = jnp

        # Set all parameters
        light_curves = []
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
                ab = pl_params.ab[:self.max_fourier_term]

                if self.parameters.limb_dark.value in ['uniform', 'linear',
                                                       'quadratic',
                                                       'kipping2013']:
                    light_curve *= harmonica.harmonica_transit_quad_ld(
                        time, t0=pl_params.t0, period=pl_params.per,
                        a=pl_params.a, inc=pl_params.inc_rad,
                        ecc=pl_params.ecc, omega=pl_params.w_rad,
                        u1=pl_params.u1, u2=pl_params.u2, r=ab)
                elif self.parameters.limb_dark.value == 'nonlinear':
                    light_curve *= harmonica.harmonica_transit_nonlinear_ld(
                        time, t0=pl_params.t0, period=pl_params.per,
                        a=pl_params.a, inc=pl_params.inc_rad,
                        ecc=pl_params.ecc, omega=pl_params.w_rad,
                        u1=pl_params.u1, u2=pl_params.u2,
                        u3=pl_params.u3, u4=pl_params.u4, r=ab)
                else:
                    raise NotImplementedError(
                        'The requested limb-darkening model '
                        f'{self.parameters.limb_dark.value} is not supported '
                        'by the HarmonicaTransitModel.')

            light_curves.append(light_curve)
        lcfinal = lib.concatenate(light_curves)

        return lcfinal
