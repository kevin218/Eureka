from typing import Any, Dict, List, Optional

import jax.numpy as jnp
from jax.typing import ArrayLike

from . import JaxModel
from ...lib.split_channels import split


def _get_channel_param(
    param_dict: Dict[str, float],
    base_name: str,
    channel: int,
) -> Optional[float]:
    """Get a possibly channel-specific parameter from param_dict.

    Tries ``f"{base_name}_ch{channel}"`` first, then falls back to
    ``base_name``. Returns ``None`` if neither is present.
    """
    if channel != 0:
        ch_key = f"{base_name}_ch{channel}"
        if ch_key in param_dict:
            return param_dict[ch_key]
    if base_name in param_dict:
        return param_dict[base_name]
    return None


def evaluate_step_model_jax(
    time_global: ArrayLike,
    nints: List[int],
    multwhite: bool,
    fitted_channels: List[int],
    param_dict: Dict[str, float],
    max_steps: int = 10,
) -> jnp.ndarray:
    """Evaluate the step-function systematics model in a JAX-pure way.

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
    param_dict : dict
        Flat parameter dictionary containing ``step{i}`` and ``steptime{i}``
        (and their optional ``_ch#`` versions).
    max_steps : int, optional
        Maximum number of steps to consider (defaults to 10).

    Returns
    -------
    jnp.ndarray
        Multiplicative step model evaluated over all channels (concatenated).
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

        # Start at unity and add step contributions
        lcpiece = jnp.ones_like(time)

        for i in range(max_steps):
            step_name = f"step{i}"
            t_name = f"steptime{i}"

            step_val = _get_channel_param(param_dict, step_name, chan)
            t_step = _get_channel_param(param_dict, t_name, chan)

            # If either is missing, treat this step as inactive
            if step_val is None or t_step is None:
                continue

            step_val_arr = jnp.asarray(step_val)
            t_step_arr = jnp.asarray(t_step)

            # Hard Heaviside-style step: 0 before steptime, 1 after.
            # Bool arrays are promoted to 0/1 in arithmetic with floats.
            lcpiece = lcpiece + step_val_arr * (time >= t_step_arr)

        lcfinal = lcfinal.at[offset:offset + len(lcpiece)].set(lcpiece)
        offset += len(lcpiece)

    return lcfinal


class StepModel(JaxModel):
    """Piecewise-constant step systematics model (JAX-compatible)."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            :class:`eureka.S5_lightcurve_fitting.jax_models.JaxModel`.
        """
        super().__init__(**kwargs)
        self.name = "step"
        self.modeltype = "systematic"

        # Allow configuration of how many step{i}/steptime{i} pairs to read.
        # Defaults to 10 to preserve legacy behaviour.
        self.max_steps: int = getattr(self, "max_steps", 10)

    def eval(
        self,
        param_dict: Optional[Dict[str, float]] = None,
        channel: Optional[int] = None,
        **kwargs: Any,
    ) -> jnp.ndarray:
        """Evaluate the step model.

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
            Multiplicative step model evaluated at observation times.
        """
        if param_dict is None:
            param_dict = self._get_param_dict()

        if channel is None:
            fitted_channels = self.fitted_channels
        else:
            fitted_channels = [channel]

        return evaluate_step_model_jax(
            time_global=self.time,
            nints=self.nints,
            multwhite=self.multwhite,
            fitted_channels=fitted_channels,
            param_dict=param_dict,
            max_steps=self.max_steps,
        )
