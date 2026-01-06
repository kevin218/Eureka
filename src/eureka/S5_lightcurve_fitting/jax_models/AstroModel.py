from typing import Any, Dict, Optional, List
import re

import jax.numpy as jnp
from jax.typing import ArrayLike

from .JaxModel import JaxModel
# Importing these here to give access to other differentiable models
from ..models.AstroModel import get_ecl_midpt, true_anomaly  # NOQA: F401, E501
from ...lib.split_channels import split


def compute_astroparams(
    param_dict: Dict[str, Any],
    channel: int = 0,
    pid: int = 0,
) -> Dict[str, Any]:
    """Compute derived astrophysical parameters for a planet and channel.

    Parameters
    ----------
    param_dict : dict
        Dictionary of model parameters. Keys may include global, per-planet,
        per-channel, or per-planet-per-channel names, e.g.:
        ``per``, ``per_pl1``, ``per_pl1_ch2``.
    channel : int; optional
        Channel index to select channel-specific parameters. Defaults to 0.
    pid : int; optional
        Planet ID index to select planet-specific parameters. Defaults to 0.

    Returns
    -------
    astro : dict
        Dictionary of derived and harmonized astrophysical parameters.
        Contains core orbital parameters, limb-darkening coefficients,
        Harmonica shape coefficients, and optional phase-curve / spot
        parameters where defined.
    """
    astro: Dict[str, Any] = {}

    pid_suffix = f"_pl{pid}" if pid != 0 else ""
    ch_suffix = f"_ch{channel}" if channel != 0 else ""

    def get_param(base_name: str, allow_global_fallback: bool = True,
                  default: float = None) -> Any:
        """Return the most specific matching parameter from param_dict."""
        if allow_global_fallback:
            keys: List[str] = [
                f"{base_name}{pid_suffix}{ch_suffix}",
                f"{base_name}{pid_suffix}",
                f"{base_name}{ch_suffix}",
                base_name,
            ]
        else:
            keys = [f"{base_name}{pid_suffix}{ch_suffix}"]

        for k in keys:
            if k in param_dict:
                return param_dict[k]
        return default

    # Core orbital and eclipse parameters
    astro["t0"] = get_param("t0")
    astro["per"] = get_param("per")
    astro["a"] = get_param("a")
    astro["ars"] = get_param("ars")
    astro["rp"] = get_param("rp")
    astro["rprs"] = get_param("rprs")
    astro["rp2"] = get_param("rp2")
    astro["rprs2"] = get_param("rprs2")
    astro["inc"] = get_param("inc")
    astro["b"] = get_param("b")
    astro["ecc"] = get_param("ecc")
    astro["w"] = get_param("w")
    astro["ecosw"] = get_param("ecosw")
    astro["esinw"] = get_param("esinw")
    astro["fp"] = get_param("fp")
    astro["fpfs"] = get_param("fpfs")
    astro["t_secondary"] = get_param("t_secondary")

    # Limb-darkening
    for i in range(1, 5):
        astro[f"u{i}"] = get_param(f"u{i}")

    # Harmonica shape coefficients: a1, b1, a2, b2, ..., aN, bN
    ab_keys = [
        k for k in param_dict
        if re.fullmatch(r"[ab][0-9]+(_pl[0-9]+)?(_ch[0-9]+)?", k)
    ]
    ab_keys = [
        k for k in ab_keys
        if ((f"_pl{pid}" in k) or (pid == 0 and "_pl" not in k))
        and ((f"_ch{channel}" in k) or (channel == 0 and "_ch" not in k))
    ]
    ab_keys = sorted(
        ab_keys,
        key=lambda s: (int(re.findall(r"[0-9]+", s)[0]), s[0]),
    )

    ab_coeffs: List[Any] = []
    if astro["rp"] is not None:
        ab_coeffs.append(astro["rp"])
    ab_coeffs.extend([param_dict[k] for k in ab_keys])
    astro["ab"] = jnp.array(ab_coeffs) if ab_coeffs else jnp.array([])

    # POET phase curve parameters
    for k in ["cos1_amp", "cos1_off", "cos2_amp", "cos2_off"]:
        astro[k] = get_param(k, default=0.)

    # Sinusoidal phase curve parameters
    for k in ["AmpCos1", "AmpSin1", "AmpCos2", "AmpSin2"]:
        astro[k] = get_param(k, default=0.)

    # Quasi-Lambertian parameters
    astro["quasi_gamma"] = get_param("quasi_gamma")  # No safe default
    astro["quasi_offset"] = get_param("quasi_offset", default=0.)

    # Identify which spot indices exist in param_dict for this planet/channel.
    # We let `get_param` handle _ch# suffixes, so here we only look at the
    # base names without _ch/_pl.
    spot_indices: List[int] = []
    for key in param_dict.keys():
        # Match spotrad, spotrad1, spotrad2, ... (ignore any _ch/_pl in key)
        m = re.match(r"^(spotrad)([0-9]*)(?:_pl[0-9]+)?(?:_ch[0-9]+)?$", key)
        if m is None:
            continue
        idx_str = m.group(2)
        idx = 0 if idx_str == "" else int(idx_str)
        if idx not in spot_indices:
            spot_indices.append(idx)
    spot_indices.sort()
    astro["nspots"] = len(spot_indices)

    for sidx in spot_indices:
        suffix = "" if sidx == 0 else str(sidx)
        astro[f"spotrad{suffix}"] = get_param(f"spotrad{suffix}", default=0.)
        astro[f"spotlat{suffix}"] = get_param(f"spotlat{suffix}", default=0.)
        astro[f"spotlon{suffix}"] = get_param(f"spotlon{suffix}", default=0.)
        astro[f"spotcon{suffix}"] = get_param(f"spotcon{suffix}", default=1.)

    # Global star-spot geometry (with defaults)
    astro["spotstari"] = get_param("spotstari", default=90.)
    astro["spotstarobl"] = get_param("spotstarobl", default=0.)
    astro["spotrot"] = get_param("spotrot", default=1e12)
    astro["spotnpts"] = get_param("spotnpts")

    # Additional pixel map and Ylm harmonics if defined
    for k in param_dict:
        if k.startswith("pixel") and (
            (f"_ch{channel}" in k) or channel == 0
        ):
            astro[k] = param_dict[k]
        if re.fullmatch(r"Y[0-9]+-?[0-9]+(_ch[0-9]+)?", k):
            if (f"_ch{channel}" in k) or channel == 0:
                astro[k] = param_dict[k]

    # Eccentricity / argument-of-periastron conversions
    if (
        astro.get("ecc") is None
        and astro.get("ecosw") is not None
        and astro.get("esinw") is not None
    ):
        ecosw = astro["ecosw"]
        esinw = astro["esinw"]
        astro["ecc"] = jnp.sqrt(ecosw**2 + esinw**2)
        astro["w"] = jnp.arctan2(esinw, ecosw) * 180. / jnp.pi
    elif astro.get("ecc") is not None and astro.get("w") is not None:
        ecc = astro["ecc"]
        w_rad = astro["w"] * jnp.pi / 180.
        astro["ecosw"] = ecc * jnp.cos(w_rad)
        astro["esinw"] = ecc * jnp.sin(w_rad)
    elif astro.get("ecc") is None:
        # Default to circular orbit
        astro["ecc"] = 0.
        astro["w"] = 90.
        astro["ecosw"] = 0.
        astro["esinw"] = 0.

    # Impact parameter / inclination conversion
    if (
        astro.get("b") is None
        and astro.get("a") is not None
        and astro.get("inc") is not None
    ):
        astro["b"] = astro["a"] * jnp.cos(astro["inc"] * jnp.pi / 180.)
    elif (
        astro.get("inc") is None
        and astro.get("a") is not None
        and astro.get("b") is not None
    ):
        astro["inc"] = (
            jnp.arccos(astro["b"] / astro["a"]) * 180. / jnp.pi
        )
    elif (
        astro.get("inc") is None and
        (astro.get("b") is None or astro.get("a") is None)
    ):
        raise AssertionError(
            'Either "inc" or "b" and "a" must be defined in parameters.'
        )

    # Harmonize radius and a/Rs variants
    if astro.get("rprs") is None and astro.get("rp") is not None:
        astro["rprs"] = astro["rp"]
    elif astro.get("rp") is None and astro.get("rprs") is not None:
        astro["rp"] = astro["rprs"]
    elif astro.get("rp") is None and astro.get("rprs") is None:
        raise AssertionError(
            'At least one of "rp" or "rprs" must be defined in parameters.'
        )

    if astro.get("rprs2") is None and astro.get("rp2") is not None:
        astro["rprs2"] = astro["rp2"]
    if astro.get("rp2") is None and astro.get("rprs2") is not None:
        astro["rp2"] = astro["rprs2"]

    if astro.get("ars") is None and astro.get("a") is not None:
        astro["ars"] = astro["a"]
    elif astro.get("a") is None and astro.get("ars") is not None:
        astro["a"] = astro["ars"]
    elif astro.get("a") is None and astro.get("ars") is None:
        raise AssertionError(
            'At least one of "a" or "ars" must be defined in parameters.'
        )

    if astro.get("fpfs") is None and astro.get("fp") is not None:
        astro["fpfs"] = astro["fp"]
    elif astro.get("fp") is None and astro.get("fpfs") is not None:
        astro["fp"] = astro["fpfs"]
    elif astro.get("fp") is None and astro.get("fpfs") is None:
        # Default to zero planet flux
        astro["fp"] = astro["fpfs"] = 0.

    # Stellar radius
    astro["Rs"] = get_param("Rs")

    # spotrot fallback
    if astro.get("spotrot") == 1e12:
        astro["fleck_fast"] = True
    else:
        astro["fleck_fast"] = False

    # Derived angle versions
    astro["inc_rad"] = astro["inc"] * jnp.pi / 180.
    astro["w_rad"] = astro["w"] * jnp.pi / 180.

    if param_dict.get("limb_dark") == "kipping2013":
        # Handle Kipping2013 transformation if needed
        q1 = astro["u1"]
        q2 = astro["u2"]
        astro["u1"] = 2. * jnp.sqrt(q1) * q2
        astro["u2"] = jnp.sqrt(q1) * (1. - 2. * q2)
        astro["u"] = jnp.array([astro["u1"], astro["u2"]])
    else:
        # Remove None limb-darkening coefficients and pack into array
        u_coeffs = [
            astro.get(f"u{i}")
            for i in range(1, 5)
            if astro.get(f"u{i}") is not None
        ]
        astro["u"] = jnp.array(u_coeffs)

    return astro


def evaluate_astrophysical_model_jax(
    time_global: ArrayLike,
    nints: List[int],
    multwhite: bool,
    fitted_channels: List[int],
    num_planets: int,
    param_dict: Dict[str, float],
    stellar_models: List[Any],
    jaxoplanet_model: Optional[Any],
    starry_model: Optional[Any],
    phasevariation_models: List[Any],
    **eval_kwargs: Any,
) -> ArrayLike:
    """Evaluate the full astrophysical model in a JAX-pure style.

    This function mirrors the logic in :meth:`AstroModel.eval` but does not
    depend on object state. All required inputs are supplied explicitly,
    making it suitable for use inside a future ``jax_eval_composite`` or
    NumPyro log-prob function.

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
        Dictionary of parameter values (typically from a prior sample or
        a flat parameter array decoded into a dict).
    stellar_models : list
        List of stellar/throughput models with an ``eval(param_dict, ...)``
        method.
    jaxoplanet_model : object or None
        Jaxoplanet-based transit/eclipse model with ``eval`` or ``None``.
    starry_model : object or None
        Starry-based model with ``eval`` or ``None``.
    phasevariation_models : list
        List of phase-curve / variability models with ``eval``.
    **eval_kwargs : dict
        Extra keyword arguments forwarded to the component ``eval`` calls.

    Returns
    -------
    jnp.ndarray
        The combined astrophysical model flux across all evaluated channels.
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

        # Stellar contribution
        star_flux = jnp.ones_like(time)
        for component in stellar_models:
            star_flux *= component.eval(
                param_dict, channel=chan, **eval_kwargs
            )

        if jaxoplanet_model is not None:
            star_flux *= jaxoplanet_model.eval(
                param_dict, channel=chan, **eval_kwargs
            )

        eclipses = None
        if starry_model is not None:
            result = starry_model.eval(
                param_dict, channel=chan, piecewise=True, **eval_kwargs
            )[0]
            transits = result.pop(0)
            eclipses = result
            star_flux *= transits

        # Planet contributions
        planet_fluxes = jnp.zeros_like(time)
        for pid in range(num_planets):
            if starry_model is not None and eclipses is not None:
                planet_flux = eclipses[pid]
            elif len(phasevariation_models) > 0:
                # Simple fp-only planet term if only phase-curve models exist
                fp_key = 'fp' if pid == 0 else f"fp_pl{pid}"
                planet_flux = param_dict.get(fp_key, 0.)
            else:
                planet_flux = 0.

            for model in phasevariation_models:
                planet_flux *= model.eval(
                    param_dict, pid, channel=chan, **eval_kwargs
                )

            planet_fluxes += planet_flux

        piece = star_flux + planet_fluxes
        lcfinal = lcfinal.at[offset:offset + len(piece)].set(piece)
        offset += len(piece)

    return lcfinal


class AstroModel(JaxModel):
    """A model that combines all astrophysical components."""
    def __init__(self, components: List[Any], **kwargs: Any) -> None:
        """Initialize the astrophysical model.

        Parameters
        ----------
        components : list
            A list of
            ``eureka.S5_lightcurve_fitting.models.Model`` instances that
            together comprise the astrophysical model.
        **kwargs : dict
            Additional parameters to pass to ``Model.__init__()``.
        """
        super().__init__(components=components, **kwargs)
        self.name = 'astrophysical model'
        self.modeltype = 'physical'

    @property
    def components(self) -> List[Any]:
        """Return the list of component models."""
        return self._components

    @components.setter
    def components(self, components: List[Any]) -> None:
        """Assign and classify component models."""
        self._components = components
        self.jaxoplanet_model: Optional[Any] = None
        self.starry_model: Optional[Any] = None
        self.phasevariation_models: List[Any] = []
        self.stellar_models: List[Any] = []
        for component in self._components:
            name = component.name.lower()
            if 'jaxoplanet' in name:
                self.jaxoplanet_model = component
            elif 'starry' in name:
                self.starry_model = component
            elif 'phase curve' in name:
                self.phasevariation_models.append(component)
            else:
                self.stellar_models.append(component)

    def eval(
        self,
        param_dict: Optional[Dict[str, float]] = None,
        channel: Optional[int] = None,
        **kwargs: Any,
    ) -> ArrayLike:
        """Evaluate the full astrophysical model.

        This evaluates all stellar and planetary components (e.g., transits,
        eclipses, phase curves) for one or more channels, combining their
        flux contributions additively.

        Parameters
        ----------
        param_dict : dict; optional
            If None, uses values from ``self.parameters`` (i.e., fitted mode).
        channel : int; optional
            If provided, evaluate only for the given channel.
        **kwargs : dict
            May contain the time array if not already set and any keyword
            arguments required by sub-components.

        Returns
        -------
        jnp.ndarray
            The combined astrophysical model flux across all evaluated
            channels.
        """
        if param_dict is None:
            param_dict = self._get_param_dict()

        if channel is None:
            fitted_channels = self.fitted_channels
        else:
            fitted_channels = [channel]

        return evaluate_astrophysical_model_jax(
            time_global=self.time,
            nints=self.nints,
            multwhite=self.multwhite,
            fitted_channels=fitted_channels,
            num_planets=self.num_planets,
            param_dict=param_dict,
            stellar_models=self.stellar_models,
            jaxoplanet_model=self.jaxoplanet_model,
            starry_model=self.starry_model,
            phasevariation_models=self.phasevariation_models,
            **kwargs,
        )
