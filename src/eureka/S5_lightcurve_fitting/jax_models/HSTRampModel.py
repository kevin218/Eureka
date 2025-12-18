from typing import Any, Dict, List, Optional

import jax.numpy as jnp
from jax.typing import ArrayLike

from . import JaxModel
from ...lib.split_channels import split


def _get_channel_param(
    param_dict: Dict[str, Any],
    base_name: str,
    channel: int,
    default: float = 0.,
) -> jnp.ndarray:
    """Get a (possibly channel-specific) parameter from param_dict.

    Tries ``f"{base_name}_ch{channel}"`` first, then falls back to
    ``base_name``. Returns ``default`` if neither key is present.
    """
    if channel != 0:
        ch_key = f"{base_name}_ch{channel}"
        if ch_key in param_dict:
            return jnp.asarray(param_dict[ch_key])
    if base_name in param_dict:
        return jnp.asarray(param_dict[base_name])
    return jnp.asarray(default)


def evaluate_hst_ramp_model_jax(
    time_global: ArrayLike,
    nints: List[int],
    multwhite: bool,
    fitted_channels: List[int],
    param_dict: Dict[str, Any],
) -> jnp.ndarray:
    """Evaluate the HST orbit-long ramp model in a JAX-pure way.

    This mirrors :meth:`HSTRampModel.eval` but takes all required inputs
    explicitly, making it suitable for use inside JAX / NumPyro log-prob
    functions.

    Parameters
    ----------
    time_global : array-like
        Full time array for the observation (all channels).
    nints : list[int]
        Number of integrations per channel.
    multwhite : bool
        Whether this is a multwhite fit (time split by channel).
    fitted_channels : list[int]
        Channel indices to evaluate.
    param_dict : dict
        Dictionary of parameter values.

    Returns
    -------
    jnp.ndarray
        The HST ramp multiplicative factor evaluated across all selected
        channels, concatenated into a 1D array.
    """
    time_global = jnp.asarray(time_global)
    nchan = len(fitted_channels)

    # Total length across selected channels
    if multwhite:
        total_len = sum(nints[chan] for chan in fitted_channels)
    else:
        total_len = time_global.size * nchan

    hst_flux = jnp.zeros(total_len)
    offset = 0

    for chan in fitted_channels:
        # Per-channel time array
        if multwhite:
            time = split([time_global], nints, chan)[0]
        else:
            time = time_global

        # Local time (relative to the first exposure in this channel)
        time_local = time - time[0]

        # h0...h5, with per-channel override if present
        h0 = _get_channel_param(param_dict, "h0", chan, 0.)
        h1 = _get_channel_param(param_dict, "h1", chan, 0.)
        h2 = _get_channel_param(param_dict, "h2", chan, 0.)
        h3 = _get_channel_param(param_dict, "h3", chan, 0.)
        h4 = _get_channel_param(param_dict, "h4", chan, 0.)
        h5 = _get_channel_param(param_dict, "h5", chan, 0.)

        # Batch time is relative to the start of each HST orbit.
        # h4 is the orbital period (~96 minutes) in the same units as time.
        # We assume h4 > 0 in any sensible fit; if it's zero, the modulo
        # will raise, which is better than silently changing the model.
        time_batch = (time_local - h5) % h4

        lcpiece = (
            1.
            + h0 * jnp.exp(-h1 * time_batch)
            + h2 * time_batch
            + h3 * time_batch**2
        )

        hst_flux = hst_flux.at[offset:(offset+lcpiece.size)].set(lcpiece)
        offset += lcpiece.size

    return hst_flux


class HSTRampModel(JaxModel):
    """Model for HST orbit-long exponential plus quadratic ramps."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            :class:`eureka.S5_lightcurve_fitting.jax_models.JaxModel`.
        """
        super().__init__(**kwargs)
        self.name = "hst ramp"
        self.modeltype = "systematic"

    def eval(
        self,
        param_dict: Optional[Dict[str, Any]] = None,
        channel: Optional[int] = None,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Evaluate the HST ramp model.

        Parameters
        ----------
        param_dict : dict, optional
            If None, uses values from ``self.parameters`` (fitted mode).
        channel : int, optional
            If provided, only evaluate for the specified channel.
            Otherwise concatenates across all fitted channels.
        **kwargs : dict
            Reserved for future use.

        Returns
        -------
        jnp.ndarray
            The model evaluated at the observation times.
        """
        if param_dict is None:
            param_dict = self._get_param_dict()

        if channel is None:
            fitted_channels = self.fitted_channels
        else:
            fitted_channels = [channel]

        return evaluate_hst_ramp_model_jax(
            time_global=self.time,
            nints=self.nints,
            multwhite=self.multwhite,
            fitted_channels=fitted_channels,
            param_dict=param_dict,
        )
