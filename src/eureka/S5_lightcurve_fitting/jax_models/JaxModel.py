from copy import copy
import numpy as np
import numpyro
import jax.numpy as jnp
from numpyro.distributions import (
    Normal, Uniform, LogUniform, LogNormal, TruncatedNormal)

from ..models import Model, CompositeModel
from ...lib.split_channels import split, get_trim


class model_class:
    def __init__(self):
        pass


class fit_class:
    def __init__(self):
        pass


class JaxModel(Model):
    def __init__(self, **kwargs):
        """Create a model instance.

        Parameters
        ----------
        **kwargs : dict
            Parameters to set in the PyMC3JaxModelModel object.
            Any parameter named log will not be loaded into the
            JaxModel object as Logedit objects cannot be pickled
            which is required for multiprocessing.
        """
        self.default_name = 'New JaxModel'
        # Set up PyMC3-specific default attributes
        kwargs['name'] = kwargs.get('name', self.default_name)

        # Inherit from Model class
        super().__init__(**kwargs)

        # Initialize fit with all parameters (including fixed and independent)
        # which won't get changed throughout the fit
        self.fit = fit_class()
        self.fit.parameters = self.parameters
        for key in self.parameters.dict.keys():
            setattr(self.fit, key, getattr(self.parameters, key).value)

    def __mul__(self, other):
        """Multiply model components to make a combined model.

        Parameters
        ----------
        other : eureka.S5_lightcurve_fitting.jax_models.JaxModel
            The model to multiply.

        Returns
        -------
        eureka.S5_lightcurve_fitting.jax_models.CompositeJaxModel
            The combined model.
        """
        # Make sure it is the right type
        attrs = ['flux', 'time']
        if not all([hasattr(other, attr) for attr in attrs]):
            raise TypeError('Only another Model instance may be multiplied.')

        # Combine the model parameters too
        parameters = self.parameters + other.parameters
        if self.paramtitles is None:
            paramtitles = other.paramtitles
        elif other.paramtitles is not None:
            paramtitles = self.paramtitles.append(other.paramtitles)
        else:
            paramtitles = self.paramtitles

        return CompositeJaxModel([copy(self), other],
                                 parameters=parameters,
                                 paramtitles=paramtitles)

    @property
    def model(self):
        """A getter for the model."""
        return self._model

    @model.setter
    def model(self, model):
        """A setter for the model."""
        self._model = model
        # Update the components' model
        for component in self.components:
            component.model = model

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
        # Check the type
        if not isinstance(flux_array, (np.ndarray, tuple, list, type(None))):
            raise TypeError("flux axis must be a tuple, list, or numpy array.")

        if isinstance(flux_array, np.ma.core.MaskedArray):
            # Convert to a numpy array with NaN masking
            flux_array = flux_array.filled(np.nan)

        # Set the array
        self._flux = flux_array

    @property
    def time(self):
        """A getter for the time"""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time"""
        # Check the type
        if not isinstance(time_array, (np.ndarray, tuple, list, type(None))):
            raise TypeError("Time axis must be a tuple, list, or numpy array.")

        if isinstance(time_array, np.ma.core.MaskedArray):
            # Convert to a numpy array with NaN masking
            time_array = time_array.filled(np.nan)

        # Set the array
        self._time = time_array

        # Set the array for the components
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
        return super().interp(new_time, nints, eval=eval, channel=channel,
                              **kwargs)

    def update(self, newparams, **kwargs):
        """Update the model with new parameter values.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        **kwargs : dict
            Unused by the base
            eureka.S5_lightcurve_fitting.jax_models.JaxModel class.
        """
        for val, arg in zip(newparams, self.freenames):
            # For now, the dict and Parameter are separate
            self.parameters.dict[arg][0] = val
            getattr(self.parameters, arg).value = val
            setattr(self.fit, arg, val)

        for component in self.components:
            component.update(newparams, **kwargs)

    def setup(self, **kwargs):
        """A placeholder function to do any additional setup.
        """
        for component in self.components:
            component.setup(**kwargs)


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
        # self.issetup = False

        # Inherit from JaxModel class
        JaxModel.__init__(self, components=components, **kwargs)

        # Setup Jax model (which will also be stored in components)
        self.model = model_class()

        self.GP = False
        for component in self.components:
            if component.modeltype == 'GP':
                self.GP = True

    @property
    def fit(self):
        """A getter for the fit."""
        return self._fit

    @fit.setter
    def fit(self, fit):
        """A setter for the model."""
        self._fit = fit
        # Update the components' fit
        for component in self.components:
            component.fit = fit

    def setup(self, time, flux, lc_unc, newparams):
        """Setup a model for evaluation and fitting.

        Parameters
        ----------
        time : array-like
            The time axis to use.
        flux : array-like
            The observed flux.
        lc_unc : array-like
            The estimated uncertainties from Stages 3-4.
        newparams : ndarray
            New parameter values.
        """
        self.time = time
        self.flux = flux
        self.lc_unc = lc_unc

        # Setup Jax model parameters
        for parname in self.parameters.params:
            param = getattr(self.parameters, parname)
            if param.ptype in ['fixed', 'independent']:
                setattr(self.model, parname, param.value)
            elif param.ptype not in ['free', 'shared', 'white_free',
                                     'white_fixed', 'independent']:
                message = (f'ptype {param.ptype} for parameter '
                           f'{param.name} is not recognized.')
                raise ValueError(message)
            else:
                if param.prior == 'U':
                    setattr(self.model, parname, numpyro.sample(
                        parname, Uniform(low=param.priorpar1,
                                         high=param.priorpar2)))
                elif param.prior == 'N':
                    if any(substring in parname
                            for substring in ['ecosw', 'esinw']):
                        # ecosw and esinw are defined on [-1,1]
                        setattr(self.model, parname, numpyro.sample(
                            parname, TruncatedNormal(loc=param.priorpar1,
                                                     scale=param.priorpar2,
                                                     low=-1.0, high=1.0)))
                    elif ('ecc' in parname or
                            (any(substring in parname
                                 for substring in ['u1', 'u2'])
                             and self.parameters.limb_dark.value ==
                             'kipping2013')):
                        # Kipping2013 parameters are only on [0,1]
                        # Eccentricity is only [0,1]
                        setattr(self.model, parname, numpyro.sample(
                            parname, TruncatedNormal(loc=param.priorpar1,
                                                     scale=param.priorpar2,
                                                     low=0., high=1.)))
                    elif any(substring in parname
                             for substring in ['per', 'scatter_mult',
                                               'scatter_ppm', 'c0',
                                               'r1', 'r3']):
                        setattr(self.model, parname, numpyro.sample(
                            parname, TruncatedNormal(loc=param.priorpar1,
                                                     scale=param.priorpar2,
                                                     low=0.)))
                    elif 'inc' in parname:
                        # An inclination > 90 is not meaningful
                        setattr(self.model, parname, numpyro.sample(
                            parname, TruncatedNormal(loc=param.priorpar1,
                                                     scale=param.priorpar2,
                                                     high=90.)))
                    else:
                        setattr(self.model, parname, numpyro.sample(
                            parname, Normal(loc=param.priorpar1,
                                            scale=param.priorpar2)))
                elif param.prior == 'LU':
                    setattr(self.model, parname, numpyro.sample(
                            parname, LogUniform(low=param.priorpar1,
                                                high=param.priorpar2)))
                elif param.prior == 'LN':
                    setattr(self.model, parname, numpyro.sample(
                            parname, LogNormal(loc=param.priorpar1,
                                               scale=param.priorpar2)))

        if hasattr(self.model, 'scatter_ppm'):
            for c in range(self.nchannel_fitted):
                if c == 0:
                    parname_temp = 'scatter_ppm'
                else:
                    parname_temp = f'scatter_ppm_ch{c}'

                if self.multwhite:
                    size = self.nints[c]
                else:
                    size = self.time.size
                unc = getattr(self.model, parname_temp)*jnp.ones(size)/1e6

                if c == 0:
                    self.scatter_array = unc
                else:
                    self.scatter_array = \
                        jnp.concatenate([self.scatter_array, unc])
        elif hasattr(self.model, 'scatter_mult'):
            # Fitting the noise level as a multiplier
            for c in range(self.nchannel_fitted):
                if c == 0:
                    parname_temp = 'scatter_mult'
                else:
                    parname_temp = f'scatter_mult_ch{c}'

                chan = self.fitted_channels[c]
                trim1, trim2 = get_trim(self.nints, chan)
                unc = self.lc_unc[trim1:trim2]

                scatter_mult = getattr(self.model, parname_temp)
                if c == 0:
                    self.scatter_array = unc*scatter_mult
                else:
                    self.scatter_array = \
                        jnp.concatenate([self.scatter_array,
                                         unc*scatter_mult])
        else:
            # Not fitting the noise level
            self.scatter_array = self.lc_unc

        for component in self.components:
            # Do any component setup needed after model initialization and
            # before evaluating the model
            component.setup(newparams=newparams)

        # This is how we tell jax about our observations;
        # we are assuming they are normally distributed about
        # the true model. These blocks effectively define our
        # likelihood function.
        if self.GP:
            for component in self.components:
                if component.modeltype == 'GP':
                    gps = component.gps
                    gp_component = component

            fit_lc = self.eval(eval=False, incl_GP=False)
            for c in range(self.nchannel_fitted):
                if self.nchannel_fitted > 1:
                    chan = self.fitted_channels[c]
                    # get flux and uncertainties for current channel
                    flux, unc_fit, fit_temp = split(
                        [self.flux, self.scatter_array, fit_lc],
                        self.nints, chan)
                    if self.multwhite:
                        time = split([self.time], self.nints, chan)[0]
                    else:
                        time = self.time
                else:
                    chan = 0
                    # get flux and uncertainties for current channel
                    flux = self.flux
                    unc_fit = self.scatter_array
                    fit_temp = fit_lc
                    time = self.time
                residuals = flux-fit_temp

                # Remove poorly handled masked values
                good = jnp.isfinite(time)
                unc_fit = unc_fit[good]
                residuals = residuals[good]

                kernel_inputs = gp_component.kernel_inputs[chan][0][good]
                gps[c].compute(kernel_inputs, yerr=unc_fit)
                setattr(self.model, f"obs_{c}",
                        numpyro.sample(f"obs_{c}", gps[c].numpyro_dist(),
                                       obs=residuals))
        else:
            # The likelihood function assuming Gaussian uncertainty
            self.model.obs = numpyro.sample("obs", Normal(
                self.eval(eval=False), self.scatter_array), obs=self.flux)

    def eval(self, channel=None, incl_GP=False, eval=True, **kwargs):
        """Evaluate the model components.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        incl_GP : bool; optional
            Whether or not to include the GP's predictions in the
            evaluated model predictions.
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        flux : ndarray
            The evaluated model predictions at the times self.time.
        """
        # Get the time
        if self.time is None:
            # This also updates all components
            self.time = kwargs.get('time')

        if channel is None:
            nchan = self.nchannel_fitted
        else:
            nchan = 1

        if self.multwhite and channel is None:
            # Evaluating all channels of a multwhite fit
            flux_length = len(self.time)
        elif self.multwhite:
            # Evaluating a single channel of a multwhite fit
            flux_length = self.nints[channel]
        else:
            # Evaluating a non-multwhite fit (individual or shared)
            flux_length = len(self.time)*nchan

        if eval:
            flux = np.ones(flux_length)
        else:
            flux = jnp.ones(flux_length)

        # Evaluate flux of each component
        for component in self.components:
            if component.modeltype != 'GP':
                flux *= component.eval(channel=channel, eval=eval, **kwargs)

        if incl_GP:
            flux += self.GPeval(flux, channel=channel, eval=eval, **kwargs)

        return flux

    def syseval(self, channel=None, incl_GP=False, eval=True, **kwargs):
        """Evaluate the systematic model components only.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        incl_GP : bool; optional
            Whether or not to include the GP's predictions in the
            evaluated model predictions.
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        flux : ndarray
            The evaluated systematics model predictions at the times self.time.
        """
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        if channel is None:
            nchan = self.nchannel_fitted
        else:
            nchan = 1

        if self.multwhite and channel is None:
            # Evaluating all channels of a multwhite fit
            flux_length = len(self.time)
        elif self.multwhite:
            # Evaluating a single channel of a multwhite fit
            flux_length = self.nints[channel]
        else:
            # Evaluating a non-multwhite fit (individual or shared)
            flux_length = len(self.time)*nchan

        flux = jnp.ones(flux_length)

        # Evaluate flux at each component
        for component in self.components:
            if component.modeltype == 'systematic':
                if component.time is None:
                    component.time = self.time
                flux *= component.eval(channel=channel, eval=eval, **kwargs)

        if incl_GP:
            flux += self.GPeval(flux, channel=channel, eval=eval, **kwargs)

        return flux

    def GPeval(self, fit, channel=None, eval=True, **kwargs):
        """Evaluate the GP model components only.

        Parameters
        ----------
        fit : ndarray
            The model predictions (excluding the GP).
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        flux : ndarray
            The evaluated GP model predictions at the times self.time.
        """
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        if channel is None:
            nchan = self.nchannel_fitted
        else:
            nchan = 1

        if self.multwhite and channel is None:
            # Evaluating all channels of a multwhite fit
            flux_length = len(self.time)
        elif self.multwhite:
            # Evaluating a single channel of a multwhite fit
            flux_length = self.nints[channel]
        else:
            # Evaluating a non-multwhite fit (individual or shared)
            flux_length = len(self.time)*nchan

        flux = jnp.zeros(flux_length)

        # Evaluate flux
        for component in self.components:
            if component.modeltype == 'GP':
                flux = component.eval(fit, channel=channel, eval=eval,
                                      **kwargs)
        return flux

    def physeval(self, channel=None, interp=False, eval=True, **kwargs):
        """Evaluate the physical model components only.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        interp : bool; optional
            Whether to uniformly sample in time or just use
            the self.time time points. Defaults to False.
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
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
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel]

        if interp:
            if self.multwhite:
                new_time = []
                nints_interp = []
                for chan in channels:
                    # Split the arrays that have lengths of
                    # the original time axis
                    time = split([self.time, ], self.nints, chan)[0]

                    # Remove masked points at the start or end to avoid
                    # extrapolating out to those points
                    time = time[jnp.isfinite(time)]

                    # Get time step on full time array to ensure good steps
                    dt = jnp.min(jnp.diff(time))

                    # Interpolate as needed
                    steps = int(jnp.round((time[-1]-time[0])/dt+1))
                    nints_interp.append(steps)
                    new_time.extend(jnp.linspace(time[0], time[-1], steps,
                                                 endpoint=True))
                new_time = jnp.array(new_time)
            else:
                time = self.time

                # Remove masked points at the start or end to avoid
                # extrapolating out to those points
                time = time[jnp.isfinite(time)]

                # Get time step on full time array to ensure good steps
                dt = jnp.min(jnp.diff(time))

                # Interpolate as needed
                dt = time[1]-time[0]
                steps = int(jnp.round((time[-1]-time[0])/dt+1))
                nints_interp = jnp.ones(nchan)*steps
                new_time = jnp.linspace(time[0], time[-1], steps,
                                        endpoint=True)
        else:
            new_time = self.time
            if self.multwhite and channel is not None:
                # Split the arrays that have lengths of the original time axis
                new_time = split([new_time, ], self.nints, channel)[0]
            nints_interp = self.nints

        if eval:
            lib = np
        else:
            lib = jnp

        # Setup the flux array
        if self.multwhite:
            flux = lib.ones(len(new_time))
        else:
            flux = lib.ones(len(new_time)*nchan)

        # Evaluate flux at each component
        for component in self.components:
            if component.modeltype == 'physical':
                if interp:
                    flux *= component.interp(new_time, nints_interp,
                                             channel=channel, eval=eval,
                                             **kwargs)
                else:
                    flux *= component.eval(channel=channel, eval=eval,
                                           **kwargs)
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
        # Evaluate flux at each model
        for component in self.components:
            if component.name == 'starry':

                return component.compute_fp(theta=theta)
