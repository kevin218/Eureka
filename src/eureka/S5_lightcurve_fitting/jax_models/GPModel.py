"""Functional tinygp likelihoods and an ordinary Stage 5 model wrapper."""
import jax.numpy as jnp
import numpy as np
import numpyro
import tinygp

from ...lib.split_channels import split
from .JaxModel import JaxModel, build_scatter_array_jax


class _SmoothSHO(tinygp.kernels.quasisep.SHO):
    """SHO with a transition that is differentiable at critical damping."""

    def transition_matrix(self, X1, X2):
        dt = X2 - X1
        omega = self.omega
        damping = 0.5 / self.quality
        x = omega * dt
        discriminant = (damping - 1.) * (damping + 1.)
        z2 = discriminant * x**2
        small = jnp.abs(z2) < 1e-4
        decay = jnp.exp(-damping * x)

        # Expand cosh(z) and sinh(z)/z in z² to differentiate through Q=1/2.
        z2_small = jnp.where(small, z2, 0.)
        c_small = decay * (1. + z2_small * (0.5 + z2_small / 24.))
        s_small = decay * dt * (1. + z2_small * (1./6. + z2_small / 120.))

        # Keep both branches finite when vmapped or differentiated.
        underdamped = discriminant < 0.
        g = jnp.sqrt(jnp.where(underdamped, -discriminant, 1.))
        c_under = decay * jnp.cos(g * x)
        s_under = decay * jnp.sin(g * x) / (omega * g)
        f = jnp.sqrt(jnp.where(discriminant > 0., discriminant, 1.))
        # Combine the exponential with the hyperbolic functions to avoid
        # overflow at large lags. (damping - f) = 1 / (damping + f).
        slow = jnp.exp(-x / (damping + f))
        half_diff = -0.5 * slow * jnp.expm1(-2. * f * x)
        c_over = slow - half_diff
        s_over = half_diff / (omega * f)

        c = jnp.where(small, c_small, jnp.where(underdamped, c_under, c_over))
        s = jnp.where(small, s_small, jnp.where(underdamped, s_under, s_over))
        rate = damping * omega
        return jnp.array([[c + rate * s, -omega**2 * s],
                          [s, c - rate * s]])


def build_tinygp_kernel(kernel_name, log_amplitude, log_metric, *,
                        quasiseparable=False, quality=None):
    """Build the shared tinygp kernel for ordinary and gradient fitters."""
    if kernel_name == 'SHO':
        if quality is None:
            raise ValueError('Q must be defined for the SHO kernel.')
        return _SmoothSHO(omega=jnp.sqrt(3.) * jnp.exp(-log_metric),
                          quality=quality, sigma=jnp.exp(0.5 * log_amplitude))
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


def _gp_parameter(param_dict, name, channel):
    suffix = '' if channel == 0 else f'_ch{channel}'
    return param_dict.get(f'{name}{suffix}', param_dict.get(name))


def _valid_quality(quality):
    return jnp.isfinite(quality) & (quality > 0.)


def build_gp_jax(time, uncertainty, param_dict, channel=0, normalize=False,
                 kernel_name='Matern32'):
    """Build a scalable GP on sorted times.

    ``A`` is log covariance amplitude. The Matern-3/2 scale is ``exp(2*m)``.
    The SHO frequency is ``sqrt(3)*exp(-m)``, with linear quality factor ``Q``.
    """
    amplitude = _gp_parameter(param_dict, 'A', channel)
    metric = _gp_parameter(param_dict, 'm', channel)
    if amplitude is None or metric is None:
        raise ValueError('A and m must be defined for the tinygp kernel.')
    x = jnp.asarray(time)
    if normalize:
        std = jnp.std(x)
        x = (x - jnp.mean(x)) / jnp.where(std > 0., std, 1.)
    quality = None
    if kernel_name == 'SHO':
        quality = _gp_parameter(param_dict, 'Q', channel)
        if quality is None:
            raise ValueError('Q must be defined for the SHO kernel.')
        # The likelihood rejects invalid proposals. Keep the solver finite
        # so invalid Q cannot contaminate gradients through that rejection.
        quality = jnp.where(_valid_quality(quality), quality, 0.5)
    kernel = build_tinygp_kernel(kernel_name, amplitude, metric,
                                 quasiseparable=True, quality=quality)
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
        if (len(kernel_types) != 1 or
                kernel_types[0] not in ('Matern32', 'SHO')):
            raise ValueError(
                'The JAX GP requires a single Matern32 or SHO kernel.')
        self.kernel_name = kernel_types[0]
        if list(kernel_input_names) != ['time']:
            raise ValueError('The JAX GP currently supports only time inputs.')
        if self.kernel_name == 'SHO':
            params = self._prepare_param_dict()
            for chan in self.fitted_channels:
                quality = _gp_parameter(params, 'Q', chan)
                if quality is None or not _valid_quality(quality):
                    raise ValueError(
                        f'SHO requires finite, positive Q for channel {chan}.')

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
                          self.normalize, self.kernel_name)
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
            if self.kernel_name == 'SHO':
                quality = _gp_parameter(param_dict, 'Q', chan)
                support = jnp.where(_valid_quality(quality), 0., -jnp.inf)
                numpyro.factor(f'sho_support_{chan}', support)
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
            log_prob_chan = gp.log_probability(residual)
            if self.kernel_name == 'SHO':
                quality = _gp_parameter(params, 'Q', chan)
                log_prob_chan = jnp.where(_valid_quality(quality),
                                          log_prob_chan, -jnp.inf)
            log_prob += log_prob_chan
        return log_prob
