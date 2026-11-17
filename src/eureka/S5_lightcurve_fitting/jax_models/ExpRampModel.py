from jax.typing import ArrayLike
import jax.numpy as jnp
from typing import Optional, Any

from ...lib.split_channels import split
from . import JaxModel
from .JaxModel import get_param_index_map, get_param_array


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
        r2 = ramp_params[2] if n_ramp > 2 else 0.
        r3 = ramp_params[3] if n_ramp > 3 else 0.

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

        # Include channel and wavelength variants when determining the shape.
        terms = [int(key.split('_')[0][1:]) for key in self.parameters.dict
                 if key.split('_')[0] in ('r0', 'r1', 'r2', 'r3')]
        if not terms:
            raise ValueError('No exponential ramp parameters were provided.')
        self.nramp = 4 if max(terms) >= 2 else 2

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
            pieces.append(piece - jnp.nanmin(piece))
        data_local = jnp.concatenate(pieces) if self.multwhite else pieces[0]
        nints = [len(piece) for piece in pieces]

        param_dict = dict(param_dict)
        for i in range(self.nramp):
            param_dict.setdefault(f'r{i}', 0.)
        param_names = [key for key, value in param_dict.items()
                       if not isinstance(value, str) and value is not None]
        param_array = get_param_array(param_dict, param_names)
        param_index_map = get_param_index_map(param_names)
        matrix = self.build_ramp_index_matrix(param_index_map)
        positions = [list(self.fitted_channels).index(chan)
                     for chan in channels]
        matrix = matrix[jnp.array(positions)]
        return evaluate_exp_ramp_model_jax(
            time_locals=data_local,
            nints=nints,
            ramp_index_matrix=matrix,
            param_array=param_array,
            multwhite=self.multwhite,
        )
