from typing import Any, Dict, List, Optional

import jax.numpy as jnp
from jax.typing import ArrayLike

from . import JaxModel
from ...lib.split_channels import split


def _maybe_get_channel_param(
    param_dict: Dict[str, Any],
    base_name: str,
    channel: int,
) -> Optional[float]:
    """Return a possibly channel-specific parameter or None.

    Tries ``f"{base_name}_ch{channel}"`` first, then ``base_name``.
    Returns None if neither is present.
    """
    if channel != 0:
        ch_key = f"{base_name}_ch{channel}"
        if ch_key in param_dict:
            return float(param_dict[ch_key])
    if base_name in param_dict:
        return float(param_dict[base_name])
    return None


def _get_required_param(
    param_dict: Dict[str, Any],
    base_name: str,
    channel: int,
) -> float:
    """Return a required (possibly channel-specific) parameter.

    Raises a KeyError if not found.
    """
    val = _maybe_get_channel_param(param_dict, base_name, channel)
    if val is None:
        raise KeyError(
            f"Required Lorentzian parameter '{base_name}' "
            f"(or '{base_name}_ch{channel}') not found."
        )
    return val


def evaluate_lorentzian_model_jax(
    time_global: ArrayLike,
    nints: List[int],
    multwhite: bool,
    fitted_channels: List[int],
    param_dict: Dict[str, Any],
) -> jnp.ndarray:
    """Evaluate the (possibly asymmetric) Lorentzian model in a JAX-pure way.

    This mirrors :meth:`LorentzianModel.eval` but takes all required inputs
    explicitly, making it suitable for use inside JAX / NumPyro log-prob
    functions.

    The logic matches the original implementation:

    * Asymmetric with baseline:
      - ``lor_hwhm`` and ``lor_amp`` both absent
      - Uses ``lor_amp_lhs``, ``lor_amp_rhs``,
        ``lor_hwhm_lhs``, ``lor_hwhm_rhs``
      - Baseline = 1 + lor_amp_lhs - lor_amp_rhs

    * Asymmetric with constant baseline:
      - ``lor_hwhm`` absent, and
        ``lor_amp_lhs`` and ``lor_amp_rhs`` both absent
      - Uses ``lor_amp`` plus ``lor_hwhm_lhs``, ``lor_hwhm_rhs``

    * Symmetric:
      - ``lor_hwhm_lhs``, ``lor_hwhm_rhs``,
        ``lor_amp_lhs``, ``lor_amp_rhs`` all absent
      - Uses ``lor_amp`` and ``lor_hwhm``

    Any other combination of parameters raises an Exception.
    """
    time_global = jnp.asarray(time_global)
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

        # Mask out non-finite entries but preserve them in the output
        full_time = time
        good = jnp.isfinite(time)
        time_good = time[good]

        # If all times are non-finite, just leave this segment as zeros
        # (i.e., multiplicative factor of 0 + 1 will be applied elsewhere)
        if time_good.size == 0:
            offset += full_time.size
            continue

        # Read parameters for this channel
        t0 = _get_required_param(param_dict, "lor_t0", chan)
        power = _maybe_get_channel_param(param_dict, "lor_power", chan)
        if power is None:
            power = 2.

        amp = _maybe_get_channel_param(param_dict, "lor_amp", chan)
        hwhm = _maybe_get_channel_param(param_dict, "lor_hwhm", chan)

        amp_lhs = _maybe_get_channel_param(param_dict, "lor_amp_lhs", chan)
        amp_rhs = _maybe_get_channel_param(param_dict, "lor_amp_rhs", chan)
        hwhm_lhs = _maybe_get_channel_param(param_dict, "lor_hwhm_lhs", chan)
        hwhm_rhs = _maybe_get_channel_param(param_dict, "lor_hwhm_rhs", chan)

        # Decide which Lorentzian variant we are using.
        #
        # 1) Asymmetric with baseline offset:
        #    lor_hwhm and lor_amp are both None:
        if hwhm is None and amp is None:
            if (
                amp_lhs is None
                or amp_rhs is None
                or hwhm_lhs is None
                or hwhm_rhs is None
            ):
                raise Exception(
                    "Asymmetric Lorentzian with baseline requires "
                    "lor_amp_lhs, lor_amp_rhs, lor_hwhm_lhs, lor_hwhm_rhs."
                )
            amp_lhs_arr = jnp.asarray(amp_lhs)
            amp_rhs_arr = jnp.asarray(amp_rhs)
            hwhm_lhs_arr = jnp.asarray(hwhm_lhs)
            hwhm_rhs_arr = jnp.asarray(hwhm_rhs)
            t0_arr = jnp.asarray(t0)
            power_arr = jnp.asarray(power)

            baseline = 1. + amp_lhs_arr - amp_rhs_arr
            ut = jnp.where(
                time_good <= t0_arr,
                (t0_arr - time_good) / hwhm_lhs_arr,
                (time_good - t0_arr) / hwhm_rhs_arr,
            )
            lorentzian_good = jnp.where(
                time_good <= t0_arr,
                1. + amp_lhs_arr / (1. + ut**power_arr),
                baseline + amp_rhs_arr / (1. + ut**power_arr),
            )

        # 2) Asymmetric with constant baseline:
        #    lor_hwhm is None, and lor_amp_lhs & lor_amp_rhs are None.
        #    (Original code referred to lor_amp_lhs here, which is
        #    inconsistent with the comment; we interpret this as using
        #    lor_amp with a constant baseline of 1.)
        elif hwhm is None and amp_lhs is None and amp_rhs is None:
            if hwhm_lhs is None or hwhm_rhs is None or amp is None:
                raise Exception(
                    "Asymmetric Lorentzian with constant baseline requires "
                    "lor_amp, lor_hwhm_lhs, lor_hwhm_rhs."
                )
            amp_arr = jnp.asarray(amp)
            hwhm_lhs_arr = jnp.asarray(hwhm_lhs)
            hwhm_rhs_arr = jnp.asarray(hwhm_rhs)
            t0_arr = jnp.asarray(t0)
            power_arr = jnp.asarray(power)

            ut = jnp.where(
                time_good <= t0_arr,
                (t0_arr - time_good) / hwhm_lhs_arr,
                (time_good - t0_arr) / hwhm_rhs_arr,
            )
            lorentzian_good = 1. + amp_arr / (1. + ut**power_arr)

        # 3) Symmetric Lorentzian:
        #    no side-specific widths or amplitudes.
        elif (
            hwhm_lhs is None
            and hwhm_rhs is None
            and amp_lhs is None
            and amp_rhs is None
        ):
            if hwhm is None or amp is None:
                raise Exception(
                    "Symmetric Lorentzian requires lor_amp and lor_hwhm."
                )
            amp_arr = jnp.asarray(amp)
            hwhm_arr = jnp.asarray(hwhm)
            t0_arr = jnp.asarray(t0)
            power_arr = jnp.asarray(power)

            ut = 2. * (time_good - t0_arr) / hwhm_arr
            lorentzian_good = 1. + amp_arr / (1. + ut**power_arr)

        else:
            # Unresolvable situation
            raise Exception(
                "Cannot determine the type of Lorentzian model to fit. "
                "Use one of the following options: "
                "1) lor_amp, lor_hwhm; "
                "2) lor_amp, lor_hwhm_lhs, lor_hwhm_rhs; "
                "3) lor_amp_lhs, lor_amp_rhs, lor_hwhm_lhs, lor_hwhm_rhs."
            )

        # Re-insert into full time array, keeping NaNs where original time
        # was non-finite.
        lorentzian_full = jnp.full_like(full_time, jnp.nan)
        lorentzian_full = lorentzian_full.at[good].set(lorentzian_good)

        lcfinal = lcfinal.at[offset:(offset+full_time.size)].set(
            lorentzian_full
        )
        offset += full_time.size

    return lcfinal


class LorentzianModel(JaxModel):
    """An asymmetric Lorentzian model."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the asymmetric Lorentzian model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            :class:`eureka.S5_lightcurve_fitting.jax_models.JaxModel`.
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        super().__init__(**kwargs)
        self.name = "lorentzian"
        self.modeltype = "physical"

    def eval(
        self,
        param_dict: Optional[Dict[str, Any]] = None,
        channel: Optional[int] = None,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Evaluate the Lorentzian model.

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

        return evaluate_lorentzian_model_jax(
            time_global=self.time,
            nints=self.nints,
            multwhite=self.multwhite,
            fitted_channels=fitted_channels,
            param_dict=param_dict,
        )
