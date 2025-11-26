from jax.typing import ArrayLike
import jax.numpy as jnp
from typing import Optional, Any

from ...lib.split_channels import split, get_trim
from ...lib.readEPF import Parameters
from . import JaxModel
from .JaxModel import get_param_names, get_param_index_map, get_param_array


def evaluate_exp_ramp_model_jax(
    time_locals: ArrayLike,
    nints: list[int],
    ramp_index_matrix: ArrayLike,
    param_array: ArrayLike,
    multwhite: bool
) -> ArrayLike:
    """Evaluate a multichannel exponential ramp model in a JAX-pure way.

    Parameters
    ----------
    time_locals : jnp.ndarray
        The full (possibly concatenated) time array.
    nints : list[int]
        Number of integrations per channel.
    ramp_index_matrix : jnp.ndarray
        Integer index matrix of shape (nchan, n_ramp).
    param_array : jnp.ndarray
        Flattened parameter vector.
    multwhite : bool
        Whether this is a multwhite fit.

    Returns
    -------
    jnp.ndarray
        Evaluated model flux array.
    """
    nchan, n_ramp = ramp_index_matrix.shape
    total_len = sum(nints) if multwhite else time_locals.size * nchan

    lcfinal = jnp.zeros(total_len)
    offset = 0
    for chan in range(nchan):
        if multwhite:
            # Get the channel-specific portion of the time array
            start = sum(nints[:chan])
            stop = start + nints[chan]
            time = time_locals[start:stop]
        else:
            time = time_locals

        # Extract ramp parameters for the current channel
        ramp_indices = ramp_index_matrix[chan]
        ramp_params = param_array[ramp_indices]
        r0 = ramp_params[0]
        r1 = ramp_params[1]
        r2 = ramp_params[2] if n_ramp > 2 else 0.0
        r3 = ramp_params[3] if n_ramp > 3 else 0.0

        # Evaluate the ramp
        lcpiece = 1 + r0 * jnp.exp(-r1 * time) + r2 * jnp.exp(-r3 * time)

        # Store segment in output light curve array
        lcfinal = lcfinal.at[offset:offset + len(time)].set(lcpiece)
        offset += len(time)

    return lcfinal


class ExpRampModel(JaxModel):
    """Exponential ramp model for use with JAX fitters."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the exponential ramp model.

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments. Expected keys include:
                - parameters: a Parameters object
                - nchan: number of fitted channels
                - multwhite: whether using multwhite fits
                - fitted_channels: list of channels to fit
                - nints: integration counts per channel
        """
        super().__init__(**kwargs)
        self.name: str = 'exp. ramp'
        self.modeltype: str = 'systematic'

        self.parameters: Parameters = kwargs.get('parameters', Parameters())
        self.nchannel_fitted: int = kwargs.get('nchan', 1)
        self.multwhite: bool = kwargs.get('multwhite', False)
        self.fitted_channels: list[int] = kwargs.get(
            'fitted_channels', list(range(self.nchannel_fitted))
        )
        self.nints: list[int] = kwargs.get('nints', [0] * self.nchannel_fitted)

        # Infer number of ramp terms from global (non-channel) parameters only
        self.nramp: int = 0
        for i in range(4):
            if f"r{i}" in self.parameters.dict:
                self.nramp = i + 1
        assert self.nramp in (2, 4), (
            "Only 2 or 4 ramp parameters are supported."
        )

        self._time: Optional[ArrayLike] = None
        self.time_local: Optional[ArrayLike] = None
        self.total_length: Optional[int] = None
        self._ramp_index_matrix: Optional[ArrayLike] = None
        self._param_names: Optional[list[str]] = None
        self._param_index_map: Optional[dict[str, int]] = None

    @property
    def time(self) -> Optional[ArrayLike]:
        """Get the time array."""
        return self._time

    @time.setter
    def time(self, time_array: Any) -> None:
        """Set the time array and generate time_local centered arrays.

        Parameters
        ----------
        time_array : array-like
            The original observation time array (possibly masked).
        """
        if hasattr(time_array, "filled"):
            time_array = time_array.filled(jnp.nan)
        self._time = jnp.array(time_array)

        if self._time is not None:
            if self.multwhite:
                # Build channel-separated centered time arrays
                time_local = []
                self.total_length = 0
                for chan in self.fitted_channels:
                    trim1, trim2 = get_trim(self.nints, chan)
                    time = self._time[trim1:trim2]
                    self.total_length += time.size
                    time_local.append(time - time[0])
                self.time_local = jnp.concatenate(time_local)
            else:
                # Center full time array and tile for channels
                self.time_local = self._time - self._time[0]
                self.total_length = (
                    self.time_local.size * self.nchannel_fitted
                )

    def build_ramp_index_matrix(
        self, param_index_map: dict[str, int]
    ) -> ArrayLike:
        """Build an index matrix for ramp parameters (r0–r3) per channel.

        Parameters
        ----------
        param_index_map : dict[str, int]
            Maps parameter names to flat array indices.

        Returns
        -------
        jnp.ndarray
            Integer index matrix of shape (nchan, nramp).
        """
        index_matrix = []
        for chan in self.fitted_channels:
            row = []
            for i in range(self.nramp):
                key_specific = f"r{i}_ch{chan}"
                key_fallback = f"r{i}"
                if key_specific in param_index_map:
                    idx = param_index_map[key_specific]
                elif key_fallback in param_index_map:
                    idx = param_index_map[key_fallback]
                else:
                    raise KeyError(
                        f"Missing both {key_specific} and {key_fallback} "
                        f"in param_index_map."
                    )
                row.append(idx)
            index_matrix.append(row)
        return jnp.array(index_matrix)

    def eval(
        self,
        param_dict: Optional[dict[str, float]] = None,
        channel: Optional[int] = None,
        **kwargs: Any
    ) -> ArrayLike:
        """
        Evaluate the exponential ramp model using JAX.

        Parameters
        ----------
        param_dict : dict, optional
            If None, uses values from self.parameters (i.e., fitted mode).
        channel : int, optional
            If specified, evaluates only for a given channel.
        **kwargs : dict
            Optional extra keyword arguments (unused).

        Returns
        -------
        jnp.ndarray
            The model flux array at the times in self.time.
        """
        if param_dict is None:
            param_dict = self._get_param_dict()
        if self._param_names is None:
            self._param_names = get_param_names(self)
        if self._param_index_map is None:
            self._param_index_map = get_param_index_map(self._param_names)
        if self._ramp_index_matrix is None:
            self._ramp_index_matrix = self.build_ramp_index_matrix(
                self._param_index_map)
        param_array = get_param_array(param_dict, self._param_names)

        if channel is not None:
            if channel not in self.fitted_channels:
                raise ValueError(
                    f"Channel {channel} not in fitted_channels "
                    f"{self.fitted_channels}."
                )
            chan_idx = list(self.fitted_channels).index(channel)
            nints = [self.nints[channel]]
            ramp_index_matrix = self._ramp_index_matrix[
                chan_idx:chan_idx + 1
            ]
            if self.multwhite:
                time = split([self.time_local], self.nints, channel)[0]
            else:
                time = self.time_local
        else:
            # Use all fitted channels, keeping nints aligned with their order
            nints = [self.nints[c] for c in self.fitted_channels]
            ramp_index_matrix = self._ramp_index_matrix
            time = self.time_local

        return evaluate_exp_ramp_model_jax(
            time_locals=time,
            nints=nints,
            ramp_index_matrix=ramp_index_matrix,
            param_array=param_array,
            multwhite=self.multwhite
        )
