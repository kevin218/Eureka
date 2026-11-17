from jax.typing import ArrayLike
import jax.numpy as jnp
from typing import Optional, Any

from ...lib.split_channels import split
from . import JaxModel
from .JaxModel import get_param_index_map, get_param_array


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
        param_dict = self._prepare_param_dict(param_dict)
        channels = list(self._channels(channel)[1])
        nints_global = kwargs.get('nints', self.nints)
        data = jnp.asarray(kwargs.get('time', self.time))
        pieces = []
        for chan in channels:
            piece = split([data], nints_global, chan)[
                0] if self.multwhite else data
            pieces.append(piece - jnp.nanmean(piece))
        data_local = jnp.concatenate(pieces) if self.multwhite else pieces[0]
        nints = [len(piece) for piece in pieces]

        param_dict = dict(param_dict)
        for i in range(self.order):
            param_dict.setdefault(f'c{i}', 0.)
        param_names = [key for key, value in param_dict.items()
                       if not isinstance(value, str) and value is not None]
        param_array = get_param_array(param_dict, param_names)
        param_index_map = get_param_index_map(param_names)
        matrix = self.build_coeff_index_matrix(param_index_map)
        positions = [list(self.fitted_channels).index(chan)
                     for chan in channels]
        matrix = matrix[jnp.array(positions)]
        return evaluate_polynomial_model_jax(
            time_locals=data_local,
            nints=nints,
            coeff_index_matrix=matrix,
            param_array=param_array,
            multwhite=self.multwhite,
        )
