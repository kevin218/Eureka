from jax.typing import ArrayLike
import jax.numpy as jnp
from typing import Optional, Any

from ...lib.split_channels import split, get_trim
from ...lib.readEPF import Parameters
from . import JaxModel
from .JaxModel import get_param_names, get_param_index_map, get_param_array


def evaluate_centroid_model_jax(
    centroids: ArrayLike,
    nints: list[int],
    coeff_index_matrix: ArrayLike,
    param_array: ArrayLike,
    multwhite: bool
) -> ArrayLike:
    """Evaluate a multichannel centroid model in a JAX-pure way.

    Parameters
    ----------
    centroids : jnp.ndarray
        The full (possibly concatenated) centroid array.
    nints : list[int]
        Number of integrations per channel.
    coeff_index_matrix : jnp.ndarray
        Integer index matrix of shape (nchan, 1).
    param_array : jnp.ndarray
        Flattened parameter vector.
    multwhite : bool
        Whether this is a multwhite fit.

    Returns
    -------
    jnp.ndarray
        Evaluated model flux array.
    """
    nchan = coeff_index_matrix.shape[0]
    total_len = sum(nints) if multwhite else centroids.size * nchan

    lcfinal = jnp.zeros(total_len)
    offset = 0
    for chan in range(nchan):
        if multwhite:
            start = sum(nints[:chan])
            stop = start + nints[chan]
            centroid = centroids[start:stop]
        else:
            centroid = centroids

        coeff_idx = coeff_index_matrix[chan, 0]
        coeff = param_array[coeff_idx]
        lcpiece = 1 + coeff * centroid

        lcfinal = lcfinal.at[offset:offset + len(centroid)].set(lcpiece)
        offset += len(centroid)

    return lcfinal


class CentroidModel(JaxModel):
    """Centroid decorrelation model for use with JAX fitters.

    This model applies a linear decorrelation against a single PSF metric,
    such as:
      - `xpos` or `xwidth`
      - `ypos` or `ywidth`

    The `axis` keyword determines which centroid quantity to use.
    Channel-specific coefficients (e.g., xpos_ch1) are supported.
    """
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the centroid model.

        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments. Expected keys include:
                - parameters: a Parameters object
                - nchan: number of fitted channels
                - multwhite: whether using multwhite fits
                - fitted_channels: list of channels to fit
                - nints: integration counts per channel
                - axis: one of 'xpos', 'xwidth', 'ypos', or 'ywidth'
        """
        super().__init__(**kwargs)
        self.axis: str = kwargs.get('axis', 'xpos')
        self.name: str = self.axis
        self.modeltype: str = 'systematic'

        self.parameters: Parameters = kwargs.get('parameters', Parameters())
        self.nchannel_fitted: int = kwargs.get('nchan', 1)
        self.multwhite: bool = kwargs.get('multwhite', False)
        self.fitted_channels: list[int] = kwargs.get(
            'fitted_channels', list(range(self.nchannel_fitted))
        )
        self.nints: list[int] = kwargs.get('nints', [0] * self.nchannel_fitted)

    @property
    def centroid(self) -> Optional[ArrayLike]:
        """Get the centroid array."""
        return self._centroid

    @centroid.setter
    def centroid(self, centroid_array: Any) -> None:
        """Set and center the centroid array.

        Parameters
        ----------
        centroid_array : array-like
            Original centroid measurements along the selected `axis`.
        """
        if hasattr(centroid_array, "filled"):
            centroid_array = centroid_array.filled(jnp.nan)
        self._centroid = jnp.array(centroid_array)

        if self._centroid is not None:
            if self.multwhite:
                # Build channel-separated centered centroid arrays
                centroid_local = []
                self.total_length = 0
                for chan in self.fitted_channels:
                    trim1, trim2 = get_trim(self.nints, chan)
                    centroid = self._centroid[trim1:trim2]
                    self.total_length += centroid.size
                    centroid_local.append(centroid - jnp.nanmean(centroid))
                self.centroid_local = jnp.concatenate(centroid_local)
            else:
                self.centroid_local = (
                    self._centroid - jnp.nanmean(self._centroid)
                )
                self.total_length = (
                    self.centroid_local.size * self.nchannel_fitted
                )

    def build_coeff_index_matrix(
        self, param_index_map: dict[str, int]
    ) -> ArrayLike:
        """Build an index matrix for centroid coefficients per channel.

        Parameters
        ----------
        param_index_map : dict[str, int]
            Maps parameter names to flat array indices.

        Returns
        -------
        jnp.ndarray
            Integer index matrix of shape (nchan, 1).
        """
        index_matrix = []
        for chan in self.fitted_channels:
            key_specific = f"{self.axis}_ch{chan}"
            key_fallback = self.axis
            if key_specific in param_index_map:
                idx = param_index_map[key_specific]
            elif key_fallback in param_index_map:
                idx = param_index_map[key_fallback]
            else:
                raise KeyError(
                    f"Missing both {key_specific} and {key_fallback} "
                    f"in param_index_map."
                )
            index_matrix.append([idx])
        return jnp.array(index_matrix)

    def eval(
        self,
        param_dict: Optional[dict[str, float]] = None,
        channel: Optional[int] = None,
        **kwargs: Any
    ) -> ArrayLike:
        """
        Evaluate the centroid model using JAX.

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
            The model flux array at the centroid values.
        """
        if param_dict is None:
            param_dict = self._get_param_dict()
        param_names = get_param_names(self)
        param_array = get_param_array(param_dict, param_names)
        param_index_map = get_param_index_map(param_names)
        coeff_index_matrix = self.build_coeff_index_matrix(
            param_index_map)

        if channel is not None:
            if channel not in self.fitted_channels:
                raise ValueError(
                    f"Channel {channel} not in fitted_channels"
                    f" {self.fitted_channels}."
                )
            chan_idx = list(self.fitted_channels).index(channel)
            nints = [self.nints[channel]]
            coeff_index_matrix = coeff_index_matrix[
                chan_idx:chan_idx + 1
            ]
            if self.multwhite:
                centroid = split(
                    [self.centroid_local], self.nints, channel
                )[0]
            else:
                centroid = self.centroid_local
        else:
            nints = self.nints
            coeff_index_matrix = coeff_index_matrix
            centroid = self.centroid_local

        return evaluate_centroid_model_jax(
            centroids=centroid,
            nints=nints,
            coeff_index_matrix=coeff_index_matrix,
            param_array=param_array,
            multwhite=self.multwhite
        )
