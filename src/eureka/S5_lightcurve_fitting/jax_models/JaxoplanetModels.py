from typing import Any, Dict, List, Optional

import inspect
import numpy as np
from astropy import constants as const
from jax.typing import ArrayLike
import jax.numpy as jnp

from jaxoplanet.orbits.keplerian import Central, System, Body
from jaxoplanet.light_curves import limb_dark_light_curve

from . import JaxModel
from .AstroModel import compute_astroparams
from ..limb_darkening_fit import ld_profile
from ...lib.split_channels import split


jnp_Rsun = jnp.array(const.R_sun.value)
jnp_Msun = jnp.array(const.M_sun.value)
jnp_G = jnp.array(const.G.value)


def evaluate_jaxoplanet_model_jax(
    time_global: ArrayLike,
    nints: List[int],
    multwhite: bool,
    fitted_channels: List[int],
    num_planets: int,
    param_dict: Dict[str, float],
) -> ArrayLike:
    """Evaluate the jaxoplanet transit model in a JAX-pure way.

    This function mirrors :meth:`JaxoplanetModel.eval` but takes all
    required inputs explicitly, making it suitable for use inside a
    future ``jax_eval_composite`` or NumPyro log-prob function.

    Parameters
    ----------
    time_global : array-like
        The full time array for the observation.
    nints : list[int]
        Number of integrations per channel.
    multwhite : bool
        Whether this is a multwhite fit.
    fitted_channels : list[int]
        List of channel indices to evaluate.
    num_planets : int
        Number of planetary components in the model.
    param_dict : dict
        Dictionary of parameter values (typically from a prior sample
        or a flat parameter array decoded into a dict).

    Returns
    -------
    jnp.ndarray
        The transit light curve evaluated across all selected channels.
    """
    time_global = jnp.array(time_global)
    nchan = len(fitted_channels)

    # Total length across selected channels
    if multwhite:
        total_len = sum(nints[chan] for chan in fitted_channels)
    else:
        total_len = time_global.size * nchan

    lcfinal = jnp.zeros(total_len)
    offset = 0

    for chan in fitted_channels:
        # Per-channel time array
        if multwhite:
            time = split([time_global], nints, chan)[0]
        else:
            time = time_global

        # Build stellar + planetary system for this channel
        astro = compute_astroparams(param_dict, channel=chan, pid=0)
        star = Central(radius=astro['Rs'], mass=0.)
        system = System(star)

        for pid in range(num_planets):
            astro = compute_astroparams(param_dict, channel=chan, pid=pid)

            a_m = astro['a'] * astro['Rs'] * jnp_Rsun
            p_s = astro['per'] * (24. * 3600.)
            Mp = ((2. * jnp.pi * a_m**(3. / 2.)) / p_s)**2
            Mp /= jnp_G * jnp_Msun

            planet = Body(
                mass=Mp,
                # Convert Rp/Rs to R_star units
                radius=jnp.abs(astro['rp'] * astro['Rs']),
                # Convert a/Rs to R_star units
                semimajor=astro['a'] * astro['Rs'],
                inclination=astro['inc_rad'],
                time_transit=astro['t0'],
                eccentricity=astro['ecc'],
                omega_peri=astro['w_rad'],
            )
            system = system.add_body(planet)

        # Use the limb-darkening coefficients from the last astro dict
        # (typically identical across planets; preserves legacy behavior).
        lc = limb_dark_light_curve(system, astro['u'])(time).reshape(-1)
        lcpiece = 1. + lc

        lcfinal = lcfinal.at[offset:offset + len(lcpiece)].set(lcpiece)
        offset += len(lcpiece)

    return lcfinal


class JaxoplanetModel(JaxModel):
    """Transit model wrapper for jaxoplanet."""
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the transit model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            ``eureka.S5_lightcurve_fitting.models.Model.__init__()``.
            Can pass in the ``parameters``, ``longparamlist``, ``nchan``,
            and ``paramtitles`` arguments here.
        """
        super().__init__(**kwargs)
        self.name = 'jaxoplanet transit'
        self.modeltype = 'physical'

        log = kwargs.get('log')

        required = np.array(['Rs'])
        missing = np.array([name not in self.paramtitles for name in required])
        if np.any(missing):
            message = (
                f'Missing required params {required[missing]} in your EPF. '
                'Make sure it is not set to \'independent\' as this is no '
                'longer a supported option; you can set these parameters to '
                'fixed if you want to maintain the old \'independent\' '
                'behavior.')
            raise AssertionError(message)

        # Store the ld_profile
        ld_name = self.parameters.limb_dark.value
        if ld_name not in ['uniform', 'linear', 'quadratic', 'kipping2013']:
            raise ValueError(f"Unsupported limb darkening model '{ld_name}'.")

        self.ld_from_S4 = kwargs.get('ld_from_S4')
        ld_func = ld_profile(ld_name, use_gen_ld=self.ld_from_S4)
        len_params = len(inspect.signature(ld_func).parameters)
        self.coeffs = [f'u{n}' for n in range(1, len_params)]

        self.ld_from_file = kwargs.get('ld_from_file')

        # Replace u parameters with generated limb-darkening values
        if self.ld_from_S4 or self.ld_from_file:
            log.writelog('Using the following limb-darkening values:')
            self.ld_array = kwargs.get('ld_coeffs')
            for c in range(self.nchannel_fitted):
                chan = self.fitted_channels[c]
                if self.ld_from_S4:
                    ld_array = self.ld_array[len_params - 2]
                else:
                    ld_array = self.ld_array
                for u in self.coeffs:
                    index = np.where(np.array(self.paramtitles) == u)[0]
                    if len(index) != 0:
                        item = self.longparamlist[c][index[0]]
                        param = int(item.split('_')[0][-1])
                        ld_val = ld_array[chan][param - 1]
                        log.writelog(f'{item}: {ld_val}')
                        # Use the file value as the starting guess
                        self.parameters.dict[item][0] = ld_val
                        # In a normal prior, center at the file value
                        if (self.parameters.dict[item][-1] == 'N'
                                and self.recenter_ld_prior):
                            self.parameters.dict[item][-3] = ld_val
                        # Update the non-dictionary form as well
                        setattr(self.parameters, item,
                                self.parameters.dict[item])

    def eval(
        self,
        param_dict: Optional[Dict[str, float]] = None,
        channel: Optional[int] = None,
        **kwargs: Any,
    ) -> ArrayLike:
        """Evaluate the jaxoplanet transit model.

        Parameters
        ----------
        param_dict : dict; optional
            If None, uses values from ``self.parameters`` (i.e., fitted mode).
        channel : int; optional
            If provided, only evaluate for the specified channel.
        **kwargs : dict
            Reserved for future keyword arguments (unused).

        Returns
        -------
        jnp.ndarray
            The model light curve evaluated at observation times.
        """
        if param_dict is None:
            param_dict = self._get_param_dict()

        if channel is None:
            fitted_channels = self.fitted_channels
        else:
            fitted_channels = [channel]

        return evaluate_jaxoplanet_model_jax(
            time_global=self.time,
            nints=self.nints,
            multwhite=self.multwhite,
            fitted_channels=fitted_channels,
            num_planets=self.num_planets,
            param_dict=param_dict,
        )
