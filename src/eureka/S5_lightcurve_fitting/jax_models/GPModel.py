"""Functional tinygp likelihoods and an ordinary Stage 5 model wrapper."""
import jax.numpy as jnp
import numpy as np
import numpyro
import tinygp

from ...lib.split_channels import split
from .JaxModel import JaxModel, build_scatter_array_jax


def build_tinygp_kernel(kernel_name, log_amplitude, log_metric, *,
                        quasiseparable=False):
    """Build the shared tinygp kernel for ordinary and gradient fitters."""
    scale = jnp.exp(2. * log_metric)
    fast_kernels = {'Matern32': tinygp.kernels.quasisep.Matern32,
                    'Exp': tinygp.kernels.quasisep.Exp}
    if quasiseparable and kernel_name in fast_kernels:
        return fast_kernels[kernel_name](
            scale=scale, sigma=jnp.exp(0.5 * log_amplitude))
    kernels = {'Matern32': tinygp.kernels.Matern32,
               'ExpSquared': tinygp.kernels.ExpSquared,
               'RationalQuadratic': tinygp.kernels.RationalQuadratic,
               'Exp': tinygp.kernels.Exp}
    if kernel_name not in kernels:
        raise AssertionError(f'Unsupported kernel for tinygp: {kernel_name}. '
                             f'Supported: {", ".join(kernels)}.')
    kwargs = {'alpha': 1.} if kernel_name == 'RationalQuadratic' else {}
    kernel = kernels[kernel_name](scale=scale, **kwargs)
    return jnp.exp(log_amplitude) * kernel


def build_gp_jax(time, uncertainty, param_dict, channel=0, normalize=False):
    """Build a scalable Matern-3/2 GP on sorted times.

    ``A`` is log covariance amplitude and ``m`` is log square-root scale,
    preserving the existing tinygp parameter convention in Stage 5.
    """
    suffix = '' if channel == 0 else f'_ch{channel}'
    amplitude = param_dict.get(f'A{suffix}', param_dict.get('A'))
    metric = param_dict.get(f'm{suffix}', param_dict.get('m'))
    if amplitude is None or metric is None:
        raise ValueError('A and m must be defined for the tinygp kernel.')
    x = jnp.asarray(time)
    if normalize:
        std = jnp.std(x)
        x = (x - jnp.mean(x)) / jnp.where(std > 0., std, 1.)
    kernel = build_tinygp_kernel('Matern32', amplitude, metric,
                                 quasiseparable=True)
    return tinygp.GaussianProcess(kernel, X=x,
                                  diag=jnp.asarray(uncertainty)**2, mean=0.,
                                  assume_sorted=True)


class GPModel(JaxModel):
    """A tinygp wrapper whose numerical operations never mutate model state."""

    def __init__(self, kernel_types, kernel_input_names, lc,
                 gp_code_name='tinygp', normalize=False, **kwargs):
        super().__init__(kernel_types=kernel_types,
                         kernel_input_names=kernel_input_names,
                         gp_code_name=gp_code_name, normalize=normalize,
                         flux=lc.flux, unc=lc.unc, unc_fit=lc.unc_fit,
                         **kwargs)
        self.name = 'GP'
        self.modeltype = 'GP'
        if gp_code_name != 'tinygp':
            raise ValueError('JAX fitters require GP_package = tinygp.')
        if list(kernel_types) != ['Matern32']:
            raise ValueError(
                'The JAX GP currently supports one Matern32 kernel.')
        if list(kernel_input_names) != ['time']:
            raise ValueError('The JAX GP currently supports only time inputs.')

        self.configure_mask(self.time, self.flux, self.unc)

    def configure_mask(self, time, flux, unc):
        """Freeze an observation mask before constructing the traced model."""
        # Resolve missing observations once, outside all JAX transformations.
        # These integer indices keep GP matrix shapes static under JIT.
        self.good_indices = {}
        for chan in self.fitted_channels:
            channel_time = self._time_for_channel(time, chan)
            channel_flux = self._data_for_channel(flux, chan)
            channel_unc = self._data_for_channel(unc, chan)
            good = (np.isfinite(np.ma.filled(channel_time, np.nan)) &
                    np.isfinite(np.ma.filled(channel_flux, np.nan)) &
                    np.isfinite(np.ma.filled(channel_unc, np.nan)))
            self.good_indices[int(chan)] = np.flatnonzero(good)

    def _time_for_channel(self, time, channel):
        if self.multwhite:
            return split([time], self.nints, channel)[0]
        return time

    def _data_for_channel(self, data, channel):
        if self.nchannel_fitted == 1:
            return data
        return split([data], self.nints, channel)[0]

    def _channel_gp(self, channel, param_dict, time, scatter):
        indices = jnp.asarray(self.good_indices[int(channel)])
        time = jnp.asarray(self._time_for_channel(time, channel))
        # The solver requires sorted coordinates. Keep the corresponding
        # observations and errors aligned, even for unsorted input times.
        indices = indices[jnp.argsort(time[indices])]
        unc = jnp.asarray(self._data_for_channel(scatter, channel))[indices]
        gp = build_gp_jax(time[indices], unc, param_dict, channel,
                          self.normalize)
        return gp, indices

    def _scatter(self, param_dict, time):
        return build_scatter_array_jax(
            self.unc, time, self.nints, self.fitted_channels,
            self.nchannel_fitted, self.multwhite, param_dict)

    def sample_likelihood(self, fit_lc, param_dict, *, time, flux, scatter):
        """Register a GP likelihood using the current hyperparameters."""
        for chan in self.fitted_channels:
            indices = self.good_indices[int(chan)]
            if not len(indices):
                continue
            gp, indices = self._channel_gp(chan, param_dict, time, scatter)
            observed = jnp.asarray(self._data_for_channel(flux, chan))
            residual = (observed[indices]
                        - self._data_for_channel(fit_lc, chan)[indices])
            numpyro.sample(f'obs_{chan}', gp.numpyro_dist(), obs=residual)

    def eval(self, fit_lc, channel=None, param_dict=None, **kwargs):
        """Condition the GP on residuals and return its predictive mean."""
        params = self._prepare_param_dict(param_dict)
        time = kwargs.get('time', self.time)
        flux = kwargs.get('flux', self.flux)
        scatter = kwargs.get('scatter', self._scatter(params, time))
        pieces = []
        for chan in self._channels(channel)[1]:
            indices = self.good_indices[int(chan)]
            x = self._time_for_channel(time, chan)
            if len(indices):
                gp, indices = self._channel_gp(chan, params, time, scatter)
                fit = (self._data_for_channel(fit_lc, chan)
                       if channel is None else fit_lc)
                observed = jnp.asarray(self._data_for_channel(flux, chan))
                residual = (observed[indices]
                            - jnp.asarray(fit)[indices])
                # Omitting prediction coordinates uses the fast path at the
                # training points, without an explicit dense cross covariance.
                mu = gp.predict(residual)
                piece = jnp.full(len(x), jnp.nan).at[indices].set(mu)
            else:
                piece = jnp.full(len(x), jnp.nan)
            pieces.append(piece)
        return jnp.concatenate(pieces)

    def loglikelihood(self, fit_lc, channel=None, param_dict=None, **kwargs):
        """Evaluate the NumPyro likelihood for ordinary fitters."""
        params = self._prepare_param_dict(param_dict)
        time = kwargs.get('time', self.time)
        flux = kwargs.get('flux', self.flux)
        scatter = kwargs.get('scatter', self._scatter(params, time))
        log_prob = 0.
        for chan in self._channels(channel)[1]:
            indices = self.good_indices[int(chan)]
            if not len(indices):
                continue
            gp, indices = self._channel_gp(chan, params, time, scatter)
            fit = (self._data_for_channel(fit_lc, chan)
                   if channel is None else fit_lc)
            observed = jnp.asarray(self._data_for_channel(flux, chan))
            residual = (observed[indices]
                        - jnp.asarray(fit)[indices])
            log_prob += gp.log_probability(residual)
        return log_prob
