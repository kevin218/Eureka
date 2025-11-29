from copy import copy
import numpy as np
import numpyro
import jax.numpy as jnp
from numpyro.distributions import (
    Normal, Uniform, LogUniform, LogNormal, TruncatedNormal)

from ..models import Model, CompositeModel
from ...lib.split_channels import split, get_trim


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
        size = nints[c] if multwhite else time.size

        pname_ppm = 'scatter_ppm' if c == 0 else f'scatter_ppm_ch{c}'
        pname_mult = 'scatter_mult' if c == 0 else f'scatter_mult_ch{c}'

        if pname_ppm in model_attrs:
            scatter = model_attrs[pname_ppm] * jnp.ones(size) / 1e6
        elif pname_mult in model_attrs:
            chan = fitted_channels[c]
            trim1, trim2 = get_trim(nints, chan)
            unc = lc_unc[trim1:trim2]
            scatter = unc * model_attrs[pname_mult]
        else:
            if nchannel_fitted == 1:
                scatter = lc_unc
            else:
                scatter = split([lc_unc], nints, fitted_channels[c])[0]

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
    flux_length = composite_model._get_flux_length(channel, nchan)
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
    flux_length = composite_model._get_flux_length(channel, nchan)
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

    if channel is None:
        nchan = composite_model.nchannel_fitted
        channels = composite_model.fitted_channels
    else:
        nchan = 1
        channels = [channel]

    if interp:
        if composite_model.multwhite:
            new_time = []
            nints_interp = []
            for chan_i in channels:
                time = split(
                    [composite_model.time],
                    composite_model.nints,
                    chan_i,
                )[0]
                time = time[jnp.isfinite(time)]
                dt = jnp.min(jnp.diff(time))
                steps = jnp.round(
                    (time[-1] - time[0]) / dt + 1,
                ).astype(int)
                nints_interp.append(steps)
                new_time.append(
                    jnp.linspace(time[0], time[-1], steps, endpoint=True),
                )
            new_time = jnp.concatenate(new_time)
        else:
            time = composite_model.time
            time = time[jnp.isfinite(time)]
            dt = jnp.min(jnp.diff(time))
            steps = jnp.round(
                (time[-1] - time[0]) / dt + 1,
            ).astype(int)
            nints_interp = jnp.ones(nchan) * steps
            new_time = jnp.linspace(time[0], time[-1], steps, endpoint=True)
    else:
        new_time = composite_model.time
        if composite_model.multwhite and channel is not None:
            new_time = split([new_time], composite_model.nints, channel)[0]
        nints_interp = composite_model.nints

    if composite_model.multwhite:
        flux = jnp.ones(len(new_time))
    else:
        flux = jnp.ones(len(new_time) * nchan)

    for component in composite_model.components:
        if component.modeltype == 'physical':
            if interp:
                flux *= component.interp(
                    new_time,
                    nints_interp,
                    channel=channel,
                    param_dict=param_dict,
                    **kwargs,
                )
            else:
                flux *= component.eval(
                    channel=channel,
                    param_dict=param_dict,
                    **kwargs,
                )

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

    flux_length = composite_model._get_flux_length(channel, nchan)
    flux = jnp.zeros(flux_length)

    for component in composite_model.components:
        if component.modeltype == 'GP':
            flux += component.eval(
                param_dict,
                fit,
                channel=channel,
                **kwargs,
            )

    return flux


def make_numpyro_composite_model(composite_model):
    """Build a numpyro-ready model function from a CompositeJaxModel.

    The returned function has the signature ``model(time, flux, lc_unc)``
    and internally:

    * sets the model's data attributes (time, flux, lc_unc)
    * samples all parameters via ``sample_prior_dict``
    * builds the scatter array using ``evaluate_composite_scatter``
    * evaluates the mean model with ``evaluate_composite_model``
    * calls ``_define_likelihood`` to register the numpyro likelihood
      (including any GP components).
    """
    def numpyro_model(time, flux, lc_unc):
        # Attach data to the composite model for component access
        composite_model.time = time
        composite_model.flux = flux
        composite_model.lc_unc = lc_unc

        # Sample all parameters from their priors (or use fixed values)
        param_dict = composite_model.sample_prior_dict()

        # Build scatter array and evaluate the mean model (no GP here)
        composite_model.scatter_array = evaluate_composite_scatter(
            composite_model,
            param_dict,
        )
        fit_lc = evaluate_composite_model(
            composite_model,
            param_dict,
            incl_GP=False,
        )

        # Define likelihood, including GP if present
        composite_model._define_likelihood(fit_lc, param_dict)

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
        self.default_name = 'New JaxModel'
        kwargs['name'] = kwargs.get('name', self.default_name)

        # Inherit from Model class
        super().__init__(**kwargs)

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
            name: self.sample_prior(getattr(self.parameters, name), name)
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
        if not isinstance(flux_array, (np.ndarray, tuple, list, type(None))):
            raise TypeError('flux axis must be a tuple, list, or numpy array.')

        if isinstance(flux_array, np.ma.core.MaskedArray):
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
        if not isinstance(time_array, (np.ndarray, tuple, list, type(None))):
            raise TypeError('Time axis must be a tuple, list, or numpy array.')

        if isinstance(time_array, np.ma.core.MaskedArray):
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
        return super().interp(
            new_time, nints, eval=eval, channel=channel, **kwargs,
        )

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
                        param.priorpar1, param.priorpar2, low=0., high=1.0
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
                parname, LogUniform(param.priorpar1, param.priorpar2),
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

    def _define_likelihood(self, fit_lc, param_dict):
        """Define the numpyro likelihood function for the model.

        This function uses the current model prediction (`fit_lc`) and
        the scatter_array to define the observational likelihood. If
        any GP components are included, they are used to define a
        GP-based likelihood for each channel. Otherwise, a Gaussian
        (Normal) likelihood is assumed.

        Parameters
        ----------
        fit_lc : jnp.ndarray
            The model flux array excluding GP contributions.
        param_dict : dict
            Dictionary of parameter values sampled from priors or provided
            externally.
        """
        if self.GP:
            # Identify the GP component and its associated GP objects
            for component in self.components:
                if component.modeltype == 'GP':
                    gps = component.gps

            # Loop through channels to apply GP likelihoods
            for c in range(self.nchannel_fitted):
                if self.nchannel_fitted > 1:
                    chan = self.fitted_channels[c]
                    flux, fit_temp = split(
                        [self.flux, fit_lc],
                        self.nints, chan,
                    )
                    if self.multwhite:
                        time = split([self.time], self.nints, chan)[0]
                    else:
                        time = self.time
                else:
                    chan = 0
                    flux = self.flux
                    fit_temp = fit_lc
                    time = self.time

                residuals = flux - fit_temp

                # Remove poorly handled masked values
                good = jnp.isfinite(time) & jnp.isfinite(residuals)
                residuals = residuals[good]
                setattr(
                    self.model, f'obs_{c}',
                    numpyro.sample(
                        f'obs_{c}', gps[c].numpyro_dist(), obs=residuals,
                    ),
                )
        else:
            numpyro.sample(
                'obs', Normal(fit_lc, self.scatter_array), obs=self.flux,
            )

    def __call__(self, time, flux, lc_unc):
        """Setup the model for either plotting or probabilistic sampling.

        Parameters
        ----------
        time : array-like
            The time axis to use.
        flux : array-like
            The observed flux.
        lc_unc : array-like
            The estimated uncertainties from Stages 3-4.
        """
        self.time = time
        self.flux = flux
        self.lc_unc = lc_unc

        # Sample model parameters from their priors
        param_dict = self.sample_prior_dict()

        # Build scatter array and evaluate mean model (no GP here)
        self.scatter_array = evaluate_composite_scatter(self, param_dict)
        fit_lc = evaluate_composite_model(self, param_dict, incl_GP=False)

        # Define likelihood, including GP if present
        self._define_likelihood(fit_lc, param_dict)

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
