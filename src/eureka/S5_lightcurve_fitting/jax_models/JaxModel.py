from copy import copy, deepcopy
import re
import numpy as np
import numpyro
import jax.numpy as jnp
from numpyro.distributions import (
    Normal, Uniform, LogUniform, LogNormal, TruncatedNormal, ImproperUniform,
    constraints)

from ..models.Model import Model, CompositeModel
from ...lib.readEPF import Parameters
from ...lib.split_channels import split


def get_param_names(model):
    """Returns a sorted list of numeric parameter names from the model.

    Only parameters whose current `.value` is a float, int, or array
    are included. This is consistent with `_get_param_dict`, which
    also drops non-numeric parameters such as strings.

    Parameters
    ----------
    model : object
        The model with a `.parameters` object.

    Returns
    -------
    list of str
        Deterministically ordered parameter names.
    """
    numeric_names = []
    for name in model.parameters.params:
        value = getattr(model.parameters, name).value
        if isinstance(value, (float, int, np.ndarray, jnp.ndarray)):
            numeric_names.append(name)
    return sorted(numeric_names)


def get_param_index_map(param_names):
    """Returns a name -> index lookup table.

    Parameters
    ----------
    param_names : list of str
        The ordered list of parameter names.

    Returns
    -------
    dict
        Dictionary mapping each name to its index.
    """
    return {name: i for i, name in enumerate(param_names)}


def get_param_array(param_dict, param_names):
    """Builds a flat JAX array from a parameter dictionary.

    Parameters
    ----------
    param_dict : dict
        Dictionary of param values (usually from `model._get_param_dict()`).
    param_names : list of str
        Parameter names, ordered.

    Returns
    -------
    jnp.ndarray
        Flat parameter array, in order of param_names.
    """
    return jnp.array([param_dict[name] for name in param_names])


def channel_parameters(param_dict, fitted_channels, wl_groups):
    """Resolve wavelength-shared parameters without changing the input dict.

    The numerical kernels use channel keys. Aliasing wavelength keys here
    keeps their precedence identical to ``Model._get_param_value`` while
    preserving gradients back to the original sampled parameter.
    """
    result = dict(param_dict)
    roots = {re.sub(r'_(?:ch|wl)\d+$', '', key) for key in param_dict}
    for chan, wl in zip(fitted_channels, wl_groups):
        for root in roots:
            wl_key = f'{root}_wl{wl}'
            ch_key = root if chan == 0 else f'{root}_ch{chan}'
            if wl and wl_key in param_dict:
                result[ch_key] = param_dict[wl_key]
            elif ch_key in param_dict:
                result[ch_key] = param_dict[ch_key]
            elif root in param_dict:
                result[ch_key] = param_dict[root]
    return result


def build_scatter_array_jax(
    lc_unc,
    time,
    nints,
    fitted_channels,
    nchannel_fitted,
    multwhite,
    param_dict,
):
    """Pure helper to construct the per-point scatter array.

    This mirrors CompositeJaxModel._build_scatter_array but as a function
    that takes all required inputs and returns a concatenated JAX array.

    Parameters
    ----------
    lc_unc : array-like
        Per-point light-curve uncertainties.
    time : array-like
        Time array for the observations.
    nints : list[int]
        Number of integrations per channel.
    fitted_channels : list[int]
        List of fitted channel indices.
    nchannel_fitted : int
        Number of fitted channels.
    multwhite : bool
        Whether using multwhite fits.
    param_dict : dict
        Dictionary of model parameters, typically from priors.

    Returns
    -------
    jnp.ndarray
        Concatenated scatter array across all channels.
    """
    lc_unc = jnp.array(lc_unc)
    time = jnp.array(time)

    scatter_list = []
    model_attrs = param_dict

    for c in range(nchannel_fitted):
        chan = fitted_channels[c]
        size = nints[chan] if multwhite and len(nints) > 1 else (
            nints[0] if multwhite else time.size)

        suffix = '' if chan == 0 else f'_ch{chan}'
        pname_ppm = f'scatter_ppm{suffix}'
        pname_mult = f'scatter_mult{suffix}'
        if pname_ppm not in model_attrs:
            pname_ppm = 'scatter_ppm'
        if pname_mult not in model_attrs:
            pname_mult = 'scatter_mult'

        if nchannel_fitted == 1:
            unc = lc_unc
        else:
            unc = split([lc_unc], nints, chan)[0]

        if pname_ppm in model_attrs:
            scatter = model_attrs[pname_ppm] * jnp.ones(size) / 1e6
        elif pname_mult in model_attrs:
            good_unc = jnp.isfinite(unc)
            safe_unc = jnp.where(good_unc, unc, 0.)
            scatter = jnp.where(good_unc,
                                safe_unc * model_attrs[pname_mult], jnp.nan)
        else:
            scatter = unc

        scatter_list.append(jnp.array(scatter))

    return jnp.concatenate(scatter_list)


def build_param_dict_from_array(param_array, param_names):
    """Inverse of get_param_array: flat vector -> parameter dictionary.

    Parameters
    ----------
    param_array : jnp.ndarray
        Flattened parameter array, in the same order as param_names.
    param_names : sequence of str
        Parameter names corresponding to entries in param_array.

    Returns
    -------
    dict
        Dictionary mapping each parameter name to its value.
    """
    return {name: param_array[i] for i, name in enumerate(param_names)}


def _ensure_param_dict(params, param_names):
    """Utility: accept either dict or (array, names) and return a dict."""
    if param_names is None:
        # Assume params is already a dict-like
        return params
    return build_param_dict_from_array(params, param_names)


def evaluate_composite_model(
    composite_model,
    params,
    param_names=None,
    channel=None,
    incl_GP=False,
    **kwargs,
):
    """Deterministic forward model for a CompositeJaxModel.

    Parameters
    ----------
    composite_model : CompositeJaxModel
        The composite model instance.
    params : dict or jnp.ndarray
        Either a parameter dict, or a flat array if param_names is given.
    param_names : list[str]; optional
        If not None, names corresponding to entries in `params`.
    channel : int; optional
        If provided, evaluate only the given channel. Defaults to None.
    incl_GP : bool; optional
        Whether to include GP contributions. Defaults to False.
    **kwargs : dict
        Extra keyword arguments forwarded to component .eval().

    Returns
    -------
    jnp.ndarray
        Model flux evaluated at composite_model.time.
    """
    param_dict = _ensure_param_dict(params, param_names)

    nchan = 1 if channel is not None else composite_model.nchannel_fitted
    time = kwargs.get('time', composite_model.time)
    nints = kwargs.get('nints', composite_model.nints)
    if composite_model.multwhite:
        channels = composite_model._channels(channel)[1]
        flux_length = sum(nints[c] if len(nints) > 1 else nints[0]
                          for c in channels)
    else:
        flux_length = len(time) * nchan
    model_flux = jnp.ones(flux_length)

    # Non-GP components
    for component in composite_model.components:
        if component.modeltype != 'GP':
            model_flux *= component.eval(
                channel=channel,
                param_dict=param_dict,
                **kwargs,
            )

    if incl_GP:
        model_flux += evaluate_composite_gp(
            composite_model,
            param_dict,
            channel=channel,
            fit=model_flux,
            **kwargs,
        )

    return model_flux


def evaluate_composite_scatter(
    composite_model,
    params,
    param_names=None,
):
    """Map parameters -> per-point scatter array for a composite model.

    Parameters
    ----------
    composite_model : CompositeJaxModel
        The composite model instance with .lc_unc, .time, .nints, etc.
        already configured.
    params : dict or jnp.ndarray
        Either a parameter dict, or a flat array if param_names is given.
    param_names : list[str]; optional
        If not None, names corresponding to entries in `params`.

    Returns
    -------
    jnp.ndarray
        Concatenated scatter array across all fitted channels.
    """
    param_dict = _ensure_param_dict(params, param_names)
    return build_scatter_array_jax(
        composite_model.lc_unc,
        composite_model.time,
        composite_model.nints,
        composite_model.fitted_channels,
        composite_model.nchannel_fitted,
        composite_model.multwhite,
        param_dict,
    )


def evaluate_composite_systematics(
    composite_model,
    params,
    param_names=None,
    channel=None,
    incl_GP=False,
    **kwargs,
):
    """Evaluate only the systematics part of a composite model.

    Parameters
    ----------
    composite_model : CompositeJaxModel
        The composite model instance.
    params : dict or jnp.ndarray
        Either a parameter dict, or a flat array if param_names is given.
    param_names : list[str]; optional
        If not None, names corresponding to entries in `params`.
    channel : int; optional
        If provided, evaluate only the given channel. Defaults to None.
    incl_GP : bool; optional
        Whether to include GP contributions. Defaults to False.
    **kwargs : dict
        Extra keyword arguments forwarded to component .eval().

    Returns
    -------
    jnp.ndarray
        Systematics model evaluated at composite_model.time.
    """
    param_dict = _ensure_param_dict(params, param_names)

    nchan = 1 if channel is not None else composite_model.nchannel_fitted
    time = kwargs.get('time', composite_model.time)
    nints = kwargs.get('nints', composite_model.nints)
    if composite_model.multwhite:
        channels = composite_model._channels(channel)[1]
        flux_length = sum(nints[c] if len(nints) > 1 else nints[0]
                          for c in channels)
    else:
        flux_length = len(time) * nchan
    flux = jnp.ones(flux_length)

    for component in composite_model.components:
        if component.modeltype == 'systematic':
            flux *= component.eval(
                channel=channel,
                param_dict=param_dict,
                **kwargs,
            )

    if incl_GP:
        flux += evaluate_composite_gp(
            composite_model,
            param_dict,
            channel=channel,
            fit=flux,
            **kwargs,
        )

    return flux


def evaluate_composite_physical(
    composite_model,
    params,
    param_names=None,
    channel=None,
    interp=False,
    **kwargs,
):
    """Evaluate only the physical part of a composite model.

    Parameters
    ----------
    composite_model : CompositeJaxModel
        The composite model instance.
    params : dict or jnp.ndarray
        Either a parameter dict, or a flat array if param_names is given.
    param_names : list[str]; optional
        If not None, names corresponding to entries in `params`.
    channel : int; optional
        If provided, evaluate only the given channel. Defaults to None.
    interp : bool; optional
        Whether to interpolate onto a uniform time grid. Defaults to False.
    **kwargs : dict
        Extra keyword arguments forwarded to component .eval()/.interp().

    Returns
    -------
    flux : jnp.ndarray
        Physical model evaluated on new_time.
    new_time : jnp.ndarray
        Time grid used by the physical model.
    nints_interp : jnp.ndarray or list
        Integration counts for the interpolated grid.
    """
    param_dict = _ensure_param_dict(params, param_names)

    channels = list(composite_model._channels(channel)[1])
    nchan = len(channels)
    time_full = kwargs.pop('time', composite_model.time)
    nints_interp = kwargs.pop('nints', composite_model.nints)
    if interp:
        # Grid sizes are host-side plotting configuration, never traced values.
        def grid(time):
            time = np.asarray(time)
            time = time[np.isfinite(time)]
            if len(time) < 2:
                return time
            dt = np.min(np.diff(time))
            steps = int(np.round((time[-1] - time[0]) / dt)) + 1
            return np.linspace(time[0], time[-1], steps)

        if composite_model.multwhite:
            grids = [grid(split([time_full], nints_interp, chan)[0])
                     for chan in composite_model.fitted_channels]
            nints_interp = [len(t) for t in grids]
            time_full = np.concatenate(grids)
        else:
            time_full = grid(time_full)
            nints_interp = [len(time_full)] * nchan

    new_time = time_full
    if composite_model.multwhite and channel is not None:
        new_time = split([time_full], nints_interp, channel)[0]
    size = len(new_time) if composite_model.multwhite else len(
        new_time) * nchan
    flux = jnp.ones(size)
    for component in composite_model.components:
        if component.modeltype == 'physical':
            flux *= component.eval(channel=channel, param_dict=param_dict,
                                   time=time_full, nints=nints_interp,
                                   **kwargs)
    return flux, new_time, nints_interp


def evaluate_composite_gp(
    composite_model,
    params,
    param_names=None,
    fit=None,
    channel=None,
    **kwargs,
):
    """Evaluate only the GP part of a composite model.

    Parameters
    ----------
    composite_model : CompositeJaxModel
        The composite model instance.
    params : dict or jnp.ndarray
        Either a parameter dict, or a flat array if param_names is given.
    param_names : list[str]; optional
        If not None, names corresponding to entries in `params`.
    fit : jnp.ndarray; optional
        Model predictions excluding GP contributions, passed to GP .eval().
    channel : int; optional
        If provided, evaluate only the given channel. Defaults to None.
    **kwargs : dict
        Extra keyword arguments forwarded to GP component .eval().

    Returns
    -------
    jnp.ndarray
        GP model evaluated at composite_model.time.
    """
    param_dict = _ensure_param_dict(params, param_names)

    if channel is None:
        nchan = composite_model.nchannel_fitted
    else:
        nchan = 1

    time = kwargs.get('time', composite_model.time)
    nints = kwargs.get('nints', composite_model.nints)
    if composite_model.multwhite:
        channels = composite_model._channels(channel)[1]
        flux_length = sum(nints[c] if len(nints) > 1 else nints[0]
                          for c in channels)
    else:
        flux_length = len(time) * nchan
    flux = jnp.zeros(flux_length)

    for component in composite_model.components:
        if component.modeltype == 'GP':
            flux += component.eval(
                fit,
                param_dict=param_dict,
                channel=channel,
                **kwargs,
            )

    return flux


def make_numpyro_composite_model(composite_model, time=None, flux=None,
                                 lc_unc=None):
    """Freeze model configuration and build a NumPyro function with no writes.

    Data and sampled parameters are explicit inputs to the numerical kernels.
    The observation mask is fixed at construction for GP fits, so the number
    of GP training points is static under JIT. Rebuild this function when the
    mask or channel configuration changes. Non-JAX components remain supported
    by the ordinary CompositeModel and derivative-free fitters.
    """
    model = deepcopy(composite_model)
    if time is not None:
        model.time = np.ma.filled(time, np.nan)
    if flux is not None:
        model.flux = np.ma.filled(flux, np.nan)
    if lc_unc is not None:
        model.lc_unc = np.ma.filled(lc_unc, np.nan)

    def validate(component):
        if not isinstance(component, JaxModel):
            raise TypeError(f"{component.name} is not JAX-compatible; use a "
                            "derivative-free fitter for mixed models.")
        for child in component.components:
            validate(child)

    for component in model.components:
        validate(component)
        if component.modeltype == 'GP' and hasattr(model, 'lc_unc'):
            component.configure_mask(model.time, model.flux, model.lc_unc)

    def numpyro_model(time, flux, lc_unc):
        param_dict = model._prepare_param_dict(model.sample_prior_dict())
        time = jnp.asarray(time)
        flux = jnp.asarray(flux)
        lc_unc = jnp.asarray(lc_unc)
        # Invalid timestamps must not enter orbital computations: masking a
        # NaN likelihood afterwards is insufficient to make its gradients safe.
        if model.multwhite:
            pieces = []
            for chan in model.fitted_channels:
                piece = split([time], model.nints, chan)[0]
                mean_time = jnp.nan_to_num(jnp.nanmean(piece))
                pieces.append(jnp.where(jnp.isfinite(piece), piece, mean_time))
            safe_time = jnp.concatenate(pieces)
        else:
            mean_time = jnp.nan_to_num(jnp.nanmean(time))
            safe_time = jnp.where(jnp.isfinite(time), time, mean_time)
        scatter = build_scatter_array_jax(
            lc_unc, time, model.nints, model.fitted_channels,
            model.nchannel_fitted, model.multwhite, param_dict)
        fit_lc = evaluate_composite_model(model, param_dict, time=safe_time)
        if model.GP:
            gps = [c for c in model.components if c.modeltype == 'GP']
            if len(gps) != 1:
                raise ValueError('Use one GP component per composite model.')
            gps[0].sample_likelihood(
                fit_lc, param_dict, time=time, flux=flux, scatter=scatter)
        else:
            valid_time = (jnp.isfinite(time) if model.multwhite else
                          jnp.tile(jnp.isfinite(time), model.nchannel_fitted))
            good = valid_time & jnp.isfinite(flux) & jnp.isfinite(lc_unc)
            # Finite substitutes avoid NaNs in masked distribution operations.
            mean = jnp.where(good, fit_lc, 0.)
            scale = jnp.where(good, scatter, 1.)
            observed = jnp.where(good, flux, 0.)
            numpyro.sample('obs', Normal(mean, scale).mask(good), obs=observed)

    return numpyro_model


class JaxModel(Model):
    def __init__(self, **kwargs):
        """Create a model instance.

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to the base Model. These typically
            include parameters, fitted channel config, etc.
            Any parameter named 'log' is skipped for compatibility with
            multiprocessing (Logedit objects cannot be pickled).
        """
        # Support the existing coefficient-dictionary constructor without a
        # second numerical model implementation in the compatibility layer.
        if kwargs.get('parameters') is None:
            parameters = Parameters()
            for key, value in kwargs.get('coeff_dict', {}).items():
                if key == 'name':
                    kwargs.setdefault('name', value)
                else:
                    setattr(parameters, key, value)
            kwargs['parameters'] = parameters

        self.default_name = 'New JaxModel'
        kwargs['name'] = kwargs.get('name', self.default_name)
        kwargs.setdefault('nchannel_fitted', kwargs.get('nchan', 1))
        kwargs.setdefault('fitted_channels',
                          list(range(kwargs['nchannel_fitted'])))
        kwargs.setdefault('num_planets', 1)

        # Inherit from Model class
        super().__init__(**kwargs)

    def _channels(self, channel=None):
        if channel is not None and channel not in self.fitted_channels:
            raise ValueError(f'Channel {channel} not in fitted_channels '
                             f'{self.fitted_channels}.')
        return super()._channels(channel)

    def _prepare_param_dict(self, param_dict=None):
        """Combine static configuration with numerical parameter overrides.

        Strings such as the limb-darkening law remain Python constants;
        sampled arrays stay in JAX and are never stored on the wrapper.
        """
        values = {name: getattr(self.parameters, name).value
                  for name in self.parameters.params}
        if param_dict is not None:
            values.update(param_dict)
        return channel_parameters(values, self.fitted_channels, self.wl_groups)

    def _get_param_dict(self):
        """Construct a dictionary of parameter values from self.parameters,
        excluding non-JAX-compatible types like strings.

        Returns
        -------
        dict
            Dictionary mapping parameter names to scalar or JAX array values.
        """
        param_dict = {}
        for name in self.parameters.params:
            value = getattr(self.parameters, name).value
            if isinstance(value, (float, int)):
                param_dict[name] = float(value)
            elif isinstance(value, (np.ndarray, jnp.ndarray)):
                param_dict[name] = jnp.array(value)
        return param_dict

    def sample_prior_dict(self):
        """Draw a sample from all parameter priors.

        Returns
        -------
        dict
            Dictionary mapping parameter names to samples drawn from
            their respective prior distributions (or fixed values).
        """
        return {
            name: (self.sample_prior(getattr(self.parameters, name), name)
                   if self.freenames is None or name in self.freenames
                   else getattr(self.parameters, name).value)
            for name in self.parameters.params
        }

    def __mul__(self, other):
        """Multiply two JaxModel instances to create a composite model.

        Parameters
        ----------
        other : eureka.S5_lightcurve_fitting.jax_models.JaxModel
            The model to multiply.

        Returns
        -------
        eureka.S5_lightcurve_fitting.jax_models.CompositeJaxModel
            The combined model.
        """
        attrs = ['flux', 'time']
        if not all([hasattr(other, attr) for attr in attrs]):
            raise TypeError('Only another Model instance may be multiplied.')

        if not isinstance(other, JaxModel):
            return Model.__mul__(self, other)

        # Combine the model parameters too
        parameters = self.parameters + other.parameters
        if self.paramtitles is None and other.paramtitles is None:
            paramtitles = None
        elif self.paramtitles is None:
            paramtitles = other.paramtitles
        elif other.paramtitles is None:
            paramtitles = self.paramtitles
        else:
            paramtitles = self.paramtitles + other.paramtitles

        return CompositeJaxModel(
            [copy(self), other],
            parameters=parameters,
            paramtitles=paramtitles,
        )

    def _get_flux_length(self, channel, nchan):
        if self.multwhite and channel is None:
            return len(self.time)
        elif self.multwhite:
            return self.nints[channel]
        else:
            return len(self.time) * nchan

    @property
    def flux(self):
        """A getter for the flux."""
        return self._flux

    @flux.setter
    def flux(self, flux_array):
        """A setter for the flux

        Parameters
        ----------
        flux_array : sequence
            The flux array
        """
        array_types = (np.ndarray, jnp.ndarray, tuple, list, type(None))
        if not isinstance(flux_array, array_types):
            raise TypeError('Flux must be a tuple, list, NumPy or JAX array.')

        if hasattr(flux_array, "filled"):
            # Convert masked arrays to regular arrays with NaNs
            flux_array = flux_array.filled(np.nan)

        self._flux = flux_array

    @property
    def time(self):
        """A getter for the time"""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time"""
        array_types = (np.ndarray, jnp.ndarray, tuple, list, type(None))
        if not isinstance(time_array, array_types):
            raise TypeError('Time must be a tuple, list, NumPy or JAX array.')

        if hasattr(time_array, "filled"):
            # Convert masked arrays to regular arrays with NaNs
            time_array = time_array.filled(np.nan)

        self._time = time_array

        # Propagate to components
        for component in self.components:
            component.time = time_array

    def interp(self, new_time, nints, eval=True, channel=None, **kwargs):
        """Evaluate the model over a different time array.

        Parameters
        ----------
        new_time : sequence
            The time array.
        nints : list
            The number of integrations for each channel, for the new
            time array.
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        **kwargs : dict
            Additional parameters to pass to self.eval().
        """
        # Inherit from Model class but add the eval argument for jax models
        return self.eval(time=new_time, nints=nints, channel=channel, **kwargs)

    def sample_prior(self, param, parname):
        """Sample from a parameter's prior distribution using numpyro.

        Parameters
        ----------
        param : Parameter
            The parameter to sample.
        parname : str
            The name of the parameter.

        Returns
        -------
        sample : jnp.DeviceArray or float
            A JAX-compatible sample from the parameter's prior, suitable for
            use in NumPyro models.
        """
        if param.ptype in ['fixed', 'independent']:
            return param.value
        elif param.prior is None:
            return numpyro.sample(parname, ImproperUniform(
                constraints.real, (), ()))
        elif param.prior == 'U':
            return numpyro.sample(
                parname, Uniform(param.priorpar1, param.priorpar2),
            )
        elif param.prior == 'N':
            if any(sub in parname for sub in ['ecosw', 'esinw']):
                # ecosw and esinw are defined on [-1,1]
                return numpyro.sample(
                    parname,
                    TruncatedNormal(
                        param.priorpar1, param.priorpar2,
                        low=-1., high=1.,
                    ),
                )
            elif ('ecc' in parname or
                  (any(sub in parname for sub in ['u1', 'u2']) and
                   self.parameters.limb_dark.value == 'kipping2013')):
                # Kipping2013 parameters are only on [0,1]
                # Eccentricity is only [0,1]
                return numpyro.sample(
                    parname,
                    TruncatedNormal(
                        param.priorpar1, param.priorpar2, low=0., high=1.
                    ),
                )
            elif any(sub in parname for sub in ['per', 'scatter_mult',
                                                'scatter_ppm', 'c0',
                                                'r1', 'r3']):
                # These parameters are only on [0, inf)
                return numpyro.sample(
                    parname,
                    TruncatedNormal(
                        param.priorpar1, param.priorpar2, low=0.,
                    ),
                )
            elif 'inc' in parname:
                # An inclination > 90 is not meaningful
                return numpyro.sample(
                    parname,
                    TruncatedNormal(
                        param.priorpar1, param.priorpar2, high=90.,
                    ),
                )
            else:
                return numpyro.sample(
                    parname, Normal(param.priorpar1, param.priorpar2),
                )
        elif param.prior == 'LU':
            return numpyro.sample(
                parname, LogUniform(jnp.exp(param.priorpar1),
                                    jnp.exp(param.priorpar2)),
            )
        elif param.prior == 'LN':
            return numpyro.sample(
                parname, LogNormal(param.priorpar1, param.priorpar2),
            )
        else:
            raise ValueError(
                f"Unsupported prior '{param.prior}' for '{parname}'. "
                f"Expected one of ['U', 'N', 'LU', 'LN'].",
            )


class CompositeJaxModel(JaxModel, CompositeModel):
    """A class to create composite models."""

    def __init__(self, components, **kwargs):
        """Initialize the composite model.

        Parameters
        ----------
        components : sequence
            The list of model components.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.jax_models.JaxModel.__init__().
        """
        JaxModel.__init__(self, components=components, **kwargs)

        self.GP = False
        for component in self.components:
            if component.modeltype == 'GP':
                self.GP = True

    def _build_scatter_array(self, param_dict):
        """Construct the scatter_array used in the likelihood function.

        Parameters
        ----------
        param_dict : dict
            Dictionary of parameter values sampled from priors or provided
            externally.
        """
        self.scatter_array = evaluate_composite_scatter(self, param_dict)

    def __call__(self, time, flux, lc_unc):
        """Evaluate a NumPyro model without changing this wrapper's state."""
        return make_numpyro_composite_model(self)(time, flux, lc_unc)

    def eval(self, channel=None, incl_GP=False, param_dict=None, **kwargs):
        """Evaluate the model components.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        incl_GP : bool; optional
            Whether or not to include the GP's predictions in the
            evaluated model predictions.
        param_dict : dict; optional
            If None, uses values from self.parameters (i.e., fitted mode).
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        flux : ndarray
            The evaluated model predictions at the times self.time.
        """
        if param_dict is None:
            param_dict = self._get_param_dict()

        return evaluate_composite_model(
            self,
            param_dict,
            channel=channel,
            incl_GP=incl_GP,
            **kwargs,
        )

    def syseval(self, channel=None, incl_GP=False, param_dict=None, **kwargs):
        """Evaluate the systematic model components only.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        incl_GP : bool; optional
            Whether or not to include the GP's predictions in the
            evaluated model predictions.
        param_dict : dict; optional
            If None, uses values from self.parameters (i.e., fitted mode).
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        flux : ndarray
            The evaluated systematics model predictions at the times self.time.
        """
        if param_dict is None:
            param_dict = self._get_param_dict()

        return evaluate_composite_systematics(
            self,
            param_dict,
            channel=channel,
            incl_GP=incl_GP,
            **kwargs,
        )

    def GPeval(self, fit, channel=None, param_dict=None, **kwargs):
        """Evaluate the GP model components only.

        Parameters
        ----------
        fit : ndarray
            The model predictions excluding GP contributions, used as input to
            GP evaluation.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        param_dict : dict; optional
            If None, uses values from self.parameters (i.e., fitted mode).
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        flux : ndarray
            The evaluated GP model predictions at the times self.time.
        """
        if param_dict is None:
            param_dict = self._get_param_dict()

        return evaluate_composite_gp(
            self,
            param_dict,
            fit=fit,
            channel=channel,
            **kwargs,
        )

    def physeval(self, channel=None, interp=False, param_dict=None, **kwargs):
        """Evaluate the physical model components only.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        interp : bool; optional
            Whether to uniformly sample in time or just use
            the self.time time points. Defaults to False.
        param_dict : dict; optional
            If None, uses values from self.parameters (i.e., fitted mode).
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        flux : ndarray
            The evaluated physical model predictions at the times self.time
            if interp==False, else at evenly spaced times between self.time[0]
            and self.time[-1] with spacing self.time[1]-self.time[0].
        new_time : ndarray
            The value of self.time if interp==False, otherwise the time points
            used in the temporally interpolated model.
        """
        if param_dict is None:
            param_dict = self._get_param_dict()

        flux, new_time, nints_interp = evaluate_composite_physical(
            self,
            param_dict,
            channel=channel,
            interp=interp,
            **kwargs,
        )
        return flux, new_time, nints_interp

    def compute_fp(self, theta=0):
        """Compute the planetary flux at an arbitrary orbital position.

        This will only be run on the first starry model contained in the
        components list.

        Parameters
        ----------
        theta : int, ndarray; optional
            The orbital angle(s) in degrees with respect to mid-eclipse.
            Defaults to 0.

        Returns
        -------
        ndarray
            The disk-integrated planetary flux for each value of theta.
        """
        for component in self.components:
            if component.name == 'starry':
                return component.compute_fp(theta=theta)

        raise ValueError("No 'starry' model found in components.")
