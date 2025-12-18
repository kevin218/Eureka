from typing import Any, Dict, List, Optional

import jax.numpy as jnp
from jax.typing import ArrayLike

from . import JaxModel
from .AstroModel import compute_astroparams
from ...lib.split_channels import split


def evaluate_sinusoid_phase_curve_jax(
    time_global: ArrayLike,
    nints: List[int],
    multwhite: bool,
    fitted_channels: List[int],
    num_planets: int,
    param_dict: Dict[str, float],
) -> jnp.ndarray:
    """Evaluate the sinusoidal phase curve model in a JAX-pure way.

    This mirrors :class:`SinusoidPhaseCurveModel` but takes all inputs
    explicitly (no access to ``self``), so it can be used inside composite
    JAX/NumPyro models.

    Parameters
    ----------
    time_global : array-like
        Full time array for the observation.
    nints : list[int]
        Number of integrations per channel.
    multwhite : bool
        Whether this is a multwhite fit.
    fitted_channels : list[int]
        Channel indices to evaluate.
    num_planets : int
        Number of planets in the system.
    param_dict : dict
        Flat parameter dictionary containing amplitudes and orbit parameters.

    Returns
    -------
    jnp.ndarray
        Multiplicative phase curve factor evaluated over all channels
        (concatenated).
    """
    time_global = jnp.asarray(time_global)
    nchan = len(fitted_channels)

    if multwhite:
        total_len = sum(nints[chan] for chan in fitted_channels)
    else:
        total_len = time_global.size * nchan

    lcfinal = jnp.ones(total_len)
    offset = 0

    for chan in fitted_channels:
        if multwhite:
            time = split([time_global], nints, chan)[0]
        else:
            time = time_global

        # Start with a flat phase curve for this channel
        lcpiece = jnp.ones_like(time)

        for pid in range(num_planets):
            astro = compute_astroparams(param_dict, channel=chan, pid=pid)

            amps = jnp.array([astro.get(k) for k in
                              ['AmpCos1', 'AmpSin1', 'AmpCos2', 'AmpSin2']])

            # Secondary eclipse time:
            # if not explicitly present, approximate as half an orbit
            # after transit for now (exact for circular orbits).
            t_secondary = astro.get('t_secondary')
            if t_secondary is None:
                t_secondary = astro['t0'] + 0.5 * astro['per']

            per = jnp.asarray(astro['per'])
            phi = 2. * jnp.pi * (time - t_secondary) / per

            phase_vars = (
                1.
                + amps[0] * (jnp.cos(phi) - 1.)
                + amps[1] * jnp.sin(phi)
                + amps[2] * (jnp.cos(2. * phi) - 1.)
                + amps[3] * jnp.sin(2. * phi)
            )
            lcpiece = lcpiece * phase_vars

        lcfinal = lcfinal.at[offset:offset + len(lcpiece)].set(lcpiece)
        offset += len(lcpiece)

    return lcfinal


class SinusoidPhaseCurveModel(JaxModel):
    """A sinusoidal phase curve model (JAX-compatible)."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the phase curve model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            :class:`eureka.S5_lightcurve_fitting.jax_models.JaxModel`.
        """
        super().__init__(**kwargs, modeltype='physical',
                         name='sinusoid phase curve')

        # Preserve the old flag, even though we currently do not enforce
        # positivity in the JAX backend. This can be revisited later with
        # a JAX-friendly implementation.
        self.force_positivity = getattr(self, 'force_positivity', False)

    def eval(
        self,
        param_dict: Optional[Dict[str, float]] = None,
        channel: Optional[int] = None,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Evaluate the sinusoidal phase curve model.

        Parameters
        ----------
        param_dict : dict, optional
            If None, uses values from ``self.parameters`` (fitted mode).
        channel : int, optional
            If provided, only evaluate for the specified channel.
        **kwargs : dict
            Reserved for future keyword arguments (unused).

        Returns
        -------
        jnp.ndarray
            Multiplicative phase curve evaluated at observation times.
        """
        if param_dict is None:
            param_dict = self._get_param_dict()

        if channel is None:
            fitted_channels = self.fitted_channels
        else:
            fitted_channels = [channel]

        return evaluate_sinusoid_phase_curve_jax(
            time_global=self.time,
            nints=self.nints,
            multwhite=self.multwhite,
            fitted_channels=fitted_channels,
            num_planets=self.num_planets,
            param_dict=param_dict,
        )
