from typing import Any, Dict, List, Optional, Tuple

import fleck.jax as fleck
import jax.numpy as jnp
from jax.typing import ArrayLike

from . import JaxModel
from .AstroModel import compute_astroparams
from ...lib.split_channels import split


def _get_channel_param(
    param_dict: Dict[str, Any],
    base_name: str,
    channel: int,
) -> Any:
    """Get a possibly channel-specific parameter from ``param_dict``.

    Tries ``f'{base_name}_ch{channel}'`` first, then falls back to
    ``base_name``. Returns ``None`` if neither is present.
    """
    if channel != 0:
        ch_key = f'{base_name}_ch{channel}'
        if ch_key in param_dict:
            return param_dict[ch_key]
    if base_name in param_dict:
        return param_dict[base_name]
    return None


def _build_spot_arrays(
    param_dict: Dict[str, Any],
    spot_bases: List[str],
    channel: int,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
    """Collect spot parameters into per-channel arrays.

    Parameters
    ----------
    param_dict : dict
        Flat parameter dictionary.
    spot_bases : list of str
        Base names for spots (e.g. ``['spotrad', 'spotrad1', ...]``),
        without channel suffixes.
    channel : int
        Channel index.

    Returns
    -------
    spotrad_arr, spotlat_arr, spotlon_arr, spotcon_arr : jnp.ndarray
        Arrays of spot radius (in units of stellar radius), latitude (deg),
        longitude (deg), and contrast, respectively.

    Notes
    -----
    fleck requires that all spots share the same contrast. This helper
    therefore enforces a single contrast per channel by taking the first
    non-None contrast it finds (``spotcon[_ch#]``) and repeating it for all
    spots. Any per-spot differences in contrast are ignored.
    """
    spot_rads: List[ArrayLike] = []
    spot_lats: List[ArrayLike] = []
    spot_lons: List[ArrayLike] = []

    # Global contrast value for all spots in this channel
    global_con = _get_channel_param(param_dict, 'spotcon', channel)

    for base in spot_bases:
        # base is 'spotrad', 'spotrad1', 'spotrad2', ...
        suffix = base[len('spotrad'):]
        rad = _get_channel_param(param_dict, f'spotrad{suffix}', channel)
        lat = _get_channel_param(param_dict, f'spotlat{suffix}', channel)
        lon = _get_channel_param(param_dict, f'spotlon{suffix}', channel)

        spot_rads.append(jnp.asarray(rad))
        spot_lats.append(jnp.asarray(lat))
        spot_lons.append(jnp.asarray(lon))

    if len(spot_rads) == 0:
        # No valid spots: return empty arrays (fleck can handle this).
        zero = jnp.array([])
        return zero, zero, zero, zero

    spotrad_arr = jnp.stack(spot_rads)
    spotlat_arr = jnp.stack(spot_lats)
    spotlon_arr = jnp.stack(spot_lons)
    # Repeat the same contrast value for every spot, as required by fleck.
    spotcon_arr = jnp.full(spotrad_arr.shape, global_con)

    return spotrad_arr, spotlat_arr, spotlon_arr, spotcon_arr


def evaluate_fleck_transit_model_jax(
    time_global: ArrayLike,
    nints: List[int],
    multwhite: bool,
    fitted_channels: List[int],
    num_planets: int,
    param_dict: Dict[str, Any],
    spot_bases: List[str],
) -> ArrayLike:
    """Evaluate the fleck-based starspot + transit model in a JAX-pure way.

    This mirrors :meth:`FleckTransitModel.eval` but takes all required inputs
    explicitly, making it suitable for use inside a composite JAX model or
    NumPyro log-prob function.

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
        Flat parameter dictionary (decoded from the free parameter vector).
    spot_bases : list of str
        Base names for starspot radii (e.g. ``['spotrad', 'spotrad1', ...]``).

    Returns
    -------
    jnp.ndarray
        The model light curve concatenated across all selected channels.
    """
    time_global = jnp.array(time_global)
    light_curves: List[ArrayLike] = []

    for chan in fitted_channels:
        # Per-channel time array
        if multwhite:
            time_chan = split([time_global], nints, chan)[0]
        else:
            time_chan = time_global

        time_chan = jnp.array(time_chan)
        lc_chan = jnp.ones(time_chan.shape[0])

        # Starspots for this channel
        spotrad, spotlat, spotlon, spotcon = _build_spot_arrays(
            param_dict=param_dict,
            spot_bases=spot_bases,
            channel=chan,
        )

        # Get star-oriented parameters (inclination, rotation) from pid=0
        astro_star = compute_astroparams(param_dict, channel=chan, pid=0)
        spotstari_deg = astro_star.get('spotstari')
        spotrot = astro_star.get('spotrot')

        # Construct the fleck ActiveStar (angles in radians)
        star = fleck.ActiveStar(
            times=time_chan,
            lon=spotlon * jnp.pi / 180.,
            lat=spotlat * jnp.pi / 180.,
            rad=spotrad,
            contrast=spotcon,
            inclination=spotstari_deg * jnp.pi / 180.,
            P_rot=spotrot,
        )

        for pid in range(num_planets):
            astro = compute_astroparams(param_dict, channel=chan, pid=pid)

            # Handle negative rp (signal inversion) without upsetting fleck.
            rp_val = jnp.asarray(astro['rp'])
            inverse = rp_val < 0.
            rp_used = jnp.abs(rp_val)

            lc_raw = star.transit_model(
                t0=astro['t0'],
                period=astro['per'],
                rp=rp_used,
                a=astro['a'],
                inclination=astro['inc_rad'],
                omega=astro['w_rad'],
                ecc=astro['ecc'],
                u1=astro['u1'],
                u2=astro['u2'],
                t0_rot=astro['t0']
            )[0].reshape(-1)

            # Invert if rp < 0
            lc_planet = jnp.where(inverse, 2. - lc_raw, lc_raw)

            lc_chan *= lc_planet

        light_curves.append(lc_chan)

    return jnp.concatenate(light_curves)


class FleckTransitModel(JaxModel):
    """Transit model with star spots using ``fleck.jax``."""
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the fleck model

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            ``eureka.S5_lightcurve_fitting.models.Model.__init__()``.
            Can pass in ``parameters``, ``longparamlist``, ``nchan``,
            and ``paramtitles`` arguments here.
        """
        super().__init__(**kwargs)
        self.name = 'fleck transit'
        self.modeltype = 'physical'

        # Precompute spot base names (no channel suffix), e.g. spotrad,
        # spotrad1, etc. This mirrors the starry model logic.
        self.spot_bases: List[str] = [
            name
            for name in self.parameters.dict.keys()
            if 'spotrad' in name and '_' not in name
        ]

    def eval(
        self,
        param_dict: Optional[Dict[str, Any]] = None,
        channel: Optional[int] = None,
        **kwargs: Any,
    ) -> ArrayLike:
        """Evaluate the fleck transit model.

        Parameters
        ----------
        param_dict : dict; optional
            If None, uses values from ``self.parameters`` (fitted mode).
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
            fitted_channels = list(self.fitted_channels)
        else:
            fitted_channels = [channel]

        return evaluate_fleck_transit_model_jax(
            time_global=self.time,
            nints=self.nints,
            multwhite=self.multwhite,
            fitted_channels=fitted_channels,
            num_planets=self.num_planets,
            param_dict=param_dict,
            spot_bases=self.spot_bases,
        )
