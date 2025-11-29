from jax.typing import ArrayLike
import jax.numpy as jnp
from typing import Optional, Any

from ...lib.split_channels import split
from ...lib.readEPF import Parameters
from . import JaxModel
from .JaxModel import get_param_names, get_param_index_map, get_param_array


def evaluate_polynomial_model_jax(
    time_locals: ArrayLike,
    nints: list[int],
    coeff_index_matrix: ArrayLike,
    param_array: ArrayLike,
    multwhite: bool
) -> ArrayLike:
    """Evaluate a multichannel polynomial model in a JAX-pure way.

    Parameters
    ----------
    time_locals : jnp.ndarray
        The full (possibly concatenated) time array.
    nints : list[int]
        Number of integrations per channel.
    coeff_index_matrix : jnp.ndarray
        Integer index matrix of shape (nchan, order).
    param_array : jnp.ndarray
        Flattened parameter vector.
    multwhite : bool
        Whether this is a multwhite fit.

    Returns
    -------
    jnp.ndarray
        Evaluated model flux array.
    """
    nchan, order = coeff_index_matrix.shape
    total_len = (
        sum(nints) if multwhite else time_locals.size * nchan
    )

    # Preallocate the output array
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

        # Extract polynomial coefficients for the current channel
        coeff_indices = coeff_index_matrix[chan]
        coeff_row = param_array[coeff_indices]

        # Build a Vandermonde matrix with increasing powers of time, e.g.,
        # [[1, t₀, t₀², ..., t₀ⁿ],
        #  [1, t₁, t₁², ..., t₁ⁿ],
        #  ...]
        # where each row corresponds to a time value and each column is
        # time**i. We transpose it so we can perform a dot product with
        # the coefficients.
        vander = jnp.vander(time, N=order, increasing=True).T

        # Evaluate the polynomial at each time point by summing cᵢ * t**i.
        lcpiece = jnp.dot(coeff_row, vander)

        # Fill the final light curve array at correct offset
        lcfinal = lcfinal.at[offset:offset + len(time)].set(lcpiece)
        offset += len(time)

    return lcfinal


class PolynomialModel(JaxModel):
    """Polynomial model for use with JAX fitters."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the polynomial model.

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
        self.name: str = 'polynomial'
        self.modeltype: str = 'systematic'

        self.parameters: Parameters = kwargs.get('parameters', Parameters())

        # Infer maximum polynomial order from existing coefficients
        ckeys = [
            k for k in self.parameters.dict
            if k.startswith("c") and k[1:].split("_")[0].isdigit()
        ]
        self.order: int = 1 + max([
            int(k[1:].split("_")[0]) for k in ckeys
        ], default=-1)
        assert self.order > 0, (
            "No valid polynomial coefficients (e.g., c0) found in parameters."
        )

        # Load configuration values
        self.nchannel_fitted: int = kwargs.get('nchan', 1)
        self.multwhite: bool = kwargs.get('multwhite', False)
        self.fitted_channels: list[int] = kwargs.get(
            'fitted_channels', list(range(self.nchannel_fitted))
        )
        self.nints: list[int] = kwargs.get('nints', [0] * self.nchannel_fitted)

        # Initialize internal variables
        self._time: Optional[ArrayLike] = None
        self.time_local: Optional[ArrayLike] = None
        self.total_length: Optional[int] = None
        self._coeff_index_matrix: Optional[ArrayLike] = None
        self._param_names: Optional[list[str]] = None
        self._param_index_map: Optional[dict[str, int]] = None

    @property
    def time(self) -> Optional[ArrayLike]:
        """Get the time array."""
        return self._time

    @time.setter
    def time(self, time_array: Any) -> None:
        """Set the time array and generate time_local centered arrays."""
        if hasattr(time_array, "filled"):
            time_array = time_array.filled(jnp.nan)
        self._time = jnp.array(time_array)

        if self._time is not None:
            if self.multwhite:
                # Build channel-separated centered time arrays,
                # concatenated in fitted_channels order.
                time_local = []
                self.total_length = 0
                for chan in self.fitted_channels:
                    time = split([self._time], self.nints, chan)[0]
                    self.total_length += time.size
                    time_local.append(time - jnp.nanmean(time))
                self.time_local = jnp.concatenate(time_local)
            else:
                # Center full time array and tile for channels
                self.time_local = self._time - jnp.nanmean(self._time)
                self.total_length = (
                    self.time_local.size * self.nchannel_fitted
                )

    def build_coeff_index_matrix(
        self,
        param_index_map: dict[str, int]
    ) -> ArrayLike:
        """Build an index matrix mapping (channel, order) -> param_array index.

        Parameters
        ----------
        param_index_map : dict[str, int]
            Maps parameter names to flat array indices.

        Returns
        -------
        jnp.ndarray
            Integer index matrix of shape (nchan, order).
        """
        index_matrix = []
        for chan in self.fitted_channels:
            row = []
            for i in range(self.order):
                key_specific = f"c{i}_ch{chan}"
                key_fallback = f"c{i}"
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
        Evaluate the polynomial model using JAX.

        Parameters
        ----------
        param_dict : dict; optional
            If None, uses values from self.parameters (i.e., fitted mode).
        channel : int; optional
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
        if self._coeff_index_matrix is None:
            self._coeff_index_matrix = self.build_coeff_index_matrix(
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
            coeff_index_matrix = self._coeff_index_matrix[
                chan_idx:chan_idx + 1
            ]

            if self.multwhite:
                start = sum(
                    self.nints[c] for c in self.fitted_channels[:chan_idx]
                )
                stop = start + self.nints[channel]
                time = self.time_local[start:stop]
            else:
                time = self.time_local
        else:
            nints = [self.nints[c] for c in self.fitted_channels]
            coeff_index_matrix = self._coeff_index_matrix
            time = self.time_local

        return evaluate_polynomial_model_jax(
            time_locals=time,
            nints=nints,
            coeff_index_matrix=coeff_index_matrix,
            param_array=param_array,
            multwhite=self.multwhite
        )
