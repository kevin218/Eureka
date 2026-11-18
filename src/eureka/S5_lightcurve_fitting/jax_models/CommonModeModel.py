"""Common-mode decorrelation with a pure JAX kernel and shared file loading."""
import jax.numpy as jnp
import numpy as np

from ...lib import astropytable
from ...lib.split_channels import split
from .JaxModel import JaxModel


def evaluate_common_mode_jax(time, cm_flux, nints, multwhite,
                             fitted_channels, param_dict):
    """Evaluate a quadratic decorrelation against a fixed time series."""
    pieces = []
    for chan in fitted_channels:
        x = split([cm_flux], nints, chan)[0] if multwhite else cm_flux
        t = split([time], nints, chan)[0] if multwhite else time
        suffix = '' if chan == 0 else f'_ch{chan}'
        cm1 = param_dict.get(f'cm1{suffix}', param_dict.get('cm1', 0.))
        cm2 = param_dict.get(f'cm2{suffix}', param_dict.get('cm2', 0.))
        piece = 1. + cm1 * x + cm2 * x**2
        pieces.append(jnp.where(jnp.isfinite(t), piece, jnp.nan))
    return jnp.concatenate(pieces)


class CommonModeModel(JaxModel):
    """Read a common-mode series once, then evaluate it with a pure kernel."""

    def __init__(self, meta, log, **kwargs):
        super().__init__(**kwargs)
        self.name = 'common_mode'
        self.modeltype = 'systematic'
        if (not hasattr(meta, 'common_mode_file') or
                not hasattr(meta, 'common_mode_name')):
            raise ValueError("meta must define 'common_mode_file' and "
                             "'common_mode_name'.")
        log.writelog(
            f'Reading {meta.common_mode_name} values from common-mode '
            f'systematics file: {meta.common_mode_file}.',
            mute=(not meta.verbose))
        table = astropytable.readtable(meta.common_mode_file)
        common = np.ma.masked_invalid(table[meta.common_mode_name])
        self.cm_flux = np.ma.filled(common - common.mean(), np.nan)

    def eval(self, channel=None, param_dict=None, **kwargs):
        return evaluate_common_mode_jax(
            jnp.asarray(kwargs.get('time', self.time)),
            jnp.asarray(self.cm_flux), kwargs.get('nints', self.nints),
            self.multwhite, self._channels(channel)[1],
            self._prepare_param_dict(param_dict))
