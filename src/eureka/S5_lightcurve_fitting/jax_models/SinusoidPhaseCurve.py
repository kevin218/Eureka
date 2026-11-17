from typing import Any, Dict, List, Optional

import jax.numpy as jnp
from jax.typing import ArrayLike

from . import JaxModel
from .AstroModel import compute_astroparams, orbital_phase_jax
from ...lib.split_channels import split


def evaluate_sinusoid_phase_curve_jax(
    time_global: ArrayLike,
    nints: List[int],
    multwhite: bool,
    fitted_channels: List[int],
    num_planets: int,
    param_dict: Dict[str, float],
    pid: Optional[int] = None,
    force_positivity: bool = False,
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

        for planet_id in (range(num_planets) if pid is None else [pid]):
            astro = compute_astroparams(
                param_dict, channel=chan, pid=planet_id, require_radius=False)

            amps = jnp.array([astro.get(k) for k in
                              ['AmpCos1', 'AmpSin1', 'AmpCos2', 'AmpSin2']])

            phi = orbital_phase_jax(astro, time)

            phase_vars = (
                1.
                + amps[0] * (jnp.cos(phi) - 1.)
                + amps[1] * jnp.sin(phi)
                + amps[2] * (jnp.cos(2. * phi) - 1.)
                + amps[3] * jnp.sin(2. * phi)
            )
            if force_positivity:
                grid = jnp.linspace(0., 2. * jnp.pi, 1000)
                minimum = jnp.min(
                    1. + amps[0] * (jnp.cos(grid) - 1.)
                    + amps[1] * jnp.sin(grid)
                    + amps[2] * (jnp.cos(2. * grid) - 1.)
                    + amps[3] * jnp.sin(2. * grid))
                phase_vars = jnp.where(minimum < 0.,
                                       1e6 * jnp.abs(minimum), phase_vars)
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
        super().__init__(**kwargs)
        self.modeltype = 'physical'
        self.name = 'sinusoid phase curve'

        self.force_positivity = getattr(self, 'force_positivity', False)

    def eval(
        self,
        param_dict: Optional[Dict[str, float]] = None,
        channel: Optional[int] = None,
        pid: Optional[int] = None,
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
        param_dict = self._prepare_param_dict(param_dict)

        if channel is None:
            fitted_channels = self.fitted_channels
        else:
            fitted_channels = [channel]

        return evaluate_sinusoid_phase_curve_jax(
            time_global=kwargs.get('time', self.time),
            nints=kwargs.get('nints', self.nints),
            multwhite=self.multwhite,
            fitted_channels=fitted_channels,
            num_planets=self.num_planets,
            param_dict=param_dict,
            pid=pid,
            force_positivity=self.force_positivity,
        )
