# src/eureka/S5_lightcurve_fitting/jax_models/QuasiLambertianPhaseCurve.py

from typing import Any, Dict, List, Optional

import jax.numpy as jnp
from jax.typing import ArrayLike

from . import JaxModel
from .AstroModel import compute_astroparams, orbital_phase_jax
from ...lib.split_channels import split


def evaluate_quasi_lambertian_phase_curve_jax(
    time_global: ArrayLike,
    nints: List[int],
    multwhite: bool,
    fitted_channels: List[int],
    num_planets: int,
    param_dict: Dict[str, float],
    pid: Optional[int] = None,
) -> jnp.ndarray:
    """Evaluate the quasi-Lambertian phase curve model in a JAX-pure way.

    This mirrors :class:`QuasiLambertianPhaseCurve` but takes all inputs
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
        Flat parameter dictionary containing quasi-Lambert parameters
        and orbit parameters.

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

        lcpiece = jnp.ones_like(time)

        for planet_id in (range(num_planets) if pid is None else [pid]):
            astro = compute_astroparams(
                param_dict, channel=chan, pid=planet_id, require_radius=False)

            gamma = jnp.asarray(astro.get('quasi_gamma', 0.))
            offset_deg = jnp.asarray(astro.get('quasi_offset', 0.))

            phi = orbital_phase_jax(astro, time)

            phase = 0.5 * (phi + offset_deg * jnp.pi / 180.)
            phase_vars = jnp.abs(jnp.cos(phase)) ** gamma

            lcpiece = lcpiece * phase_vars

        lcfinal = lcfinal.at[offset:offset + len(lcpiece)].set(lcpiece)
        offset += len(lcpiece)

    return lcfinal


class QuasiLambertianPhaseCurve(JaxModel):
    """Quasi-Lambertian phase curve based on Agol+2007 for airless planets."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            :class:`eureka.S5_lightcurve_fitting.jax_models.JaxModel`.
        """
        super().__init__(**kwargs)
        self.name = 'quasi-lambertian phase curve'
        self.modeltype = 'physical'

    def eval(
        self,
        param_dict: Optional[Dict[str, float]] = None,
        channel: Optional[int] = None,
        pid: Optional[int] = None,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Evaluate the quasi-Lambertian phase curve model.

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

        return evaluate_quasi_lambertian_phase_curve_jax(
            time_global=kwargs.get('time', self.time),
            nints=kwargs.get('nints', self.nints),
            multwhite=self.multwhite,
            fitted_channels=fitted_channels,
            num_planets=self.num_planets,
            param_dict=param_dict,
            pid=pid,
        )
