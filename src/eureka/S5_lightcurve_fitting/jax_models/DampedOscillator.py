from typing import Any, Dict, List, Optional

import jax.numpy as jnp
from jax.typing import ArrayLike

from . import JaxModel
from ...lib.split_channels import split


def _get_channel_param(
    param_dict: Dict[str, Any],
    base_name: str,
    channel: int,
    default: Optional[float] = None,
) -> Any:
    """Helper to get a possibly channel-specific parameter from param_dict.

    Tries ``f"{base_name}_ch{channel}"`` first, then falls back to
    ``base_name``. Returns ``default`` if neither is present.
    """
    if channel != 0:
        ch_key = f"{base_name}_ch{channel}"
        if ch_key in param_dict:
            return param_dict[ch_key]
    return param_dict.get(base_name, default)


def evaluate_damped_oscillator_model_jax(
    time_global: ArrayLike,
    nints: List[int],
    multwhite: bool,
    fitted_channels: List[int],
    param_dict: Dict[str, Any],
) -> jnp.ndarray:
    """Evaluate the damped oscillator model in a JAX-pure way.

    This mirrors :meth:`DampedOscillatorModel.eval` but takes all required
    inputs explicitly, making it suitable for use inside a NumPyro / JAX
    log-prob function.

    Parameters
    ----------
    time_global : array-like
        Full time array for the observation (all channels).
    nints : list[int]
        Number of integrations per channel.
    multwhite : bool
        Whether this is a multwhite fit (time is split by channel).
    fitted_channels : list[int]
        Channel indices to evaluate.
    param_dict : dict
        Dictionary of parameter values.

    Returns
    -------
    jnp.ndarray
        The damped oscillator multiplicative factor evaluated across all
        selected channels, concatenated into a 1D array.
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

        # Fetch channel-specific or global parameters
        osc_amp = _get_channel_param(param_dict, "osc_amp", chan)
        osc_amp_decay = _get_channel_param(param_dict, "osc_amp_decay",
                                           chan, 0.)
        osc_per = _get_channel_param(param_dict, "osc_per", chan)
        osc_per_decay = _get_channel_param(param_dict, "osc_per_decay",
                                           chan, 0.)
        osc_t0 = _get_channel_param(param_dict, "osc_t0", chan, 0.)
        osc_t1 = _get_channel_param(param_dict, "osc_t1", chan)

        # These are required; if missing, we let JAX raise an error rather
        # than silently changing the semantics.
        if osc_amp is None or osc_per is None or osc_t1 is None:
            raise ValueError(
                "osc_amp, osc_per, and osc_t1 must all be defined for "
                "DampedOscillatorModel."
            )

        # Cast to JAX scalars
        osc_amp = jnp.asarray(osc_amp)
        osc_amp_decay = jnp.asarray(osc_amp_decay)
        osc_per = jnp.asarray(osc_per)
        osc_per_decay = jnp.asarray(osc_per_decay)
        osc_t0 = jnp.asarray(osc_t0)
        osc_t1 = jnp.asarray(osc_t1)

        # Damped oscillator:
        #   amp(t) = osc_amp * exp(-osc_amp_decay * (t - osc_t0))
        #   per(t) = osc_per * exp(-osc_per_decay * (t - osc_t0))
        #   osc(t) = 1 + amp(t) * sin[ 2*pi*(t - osc_t1)/per(t) ]
        dt_from_t0 = time - osc_t0
        amp = osc_amp * jnp.exp(-osc_amp_decay * dt_from_t0)
        per = osc_per * jnp.exp(-osc_per_decay * dt_from_t0)

        phase = 2. * jnp.pi * (time - osc_t1) / per
        osc = 1. + amp * jnp.sin(phase)

        # Before osc_t0, the model should be flat at 1
        mask_pre_t0 = time < osc_t0
        osc = osc.at[mask_pre_t0].set(1.)

        lcfinal = lcfinal.at[offset:(offset+osc.size)].set(osc)
        offset += osc.size

    return lcfinal


class DampedOscillatorModel(JaxModel):
    """A damped oscillator model."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the damped oscillator model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            :class:`eureka.S5_lightcurve_fitting.jax_models.JaxModel`.
            Typically includes ``parameters``, ``longparamlist``,
            ``nchan``, and ``paramtitles``.
        """
        super().__init__(**kwargs)
        self.name = "damped oscillator"
        self.modeltype = "physical"

    def eval(
        self,
        param_dict: Optional[Dict[str, Any]] = None,
        channel: Optional[int] = None,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Evaluate the damped oscillator model.

        Parameters
        ----------
        param_dict : dict, optional
            If None, uses values from ``self.parameters`` (fitted mode).
        channel : int, optional
            If provided, only evaluate for the specified channel.
            Otherwise all fitted channels are concatenated.
        **kwargs : dict
            Reserved for future use.

        Returns
        -------
        jnp.ndarray
            The model evaluated at observation times.
        """
        if param_dict is None:
            param_dict = self._get_param_dict()

        if channel is None:
            fitted_channels = self.fitted_channels
        else:
            fitted_channels = [channel]

        return evaluate_damped_oscillator_model_jax(
            time_global=self.time,
            nints=self.nints,
            multwhite=self.multwhite,
            fitted_channels=fitted_channels,
            param_dict=param_dict,
        )
