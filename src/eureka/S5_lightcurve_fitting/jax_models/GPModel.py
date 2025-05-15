import numpy as np
import jax
import jax.numpy as jnp

import celerite2.jax as celerite2

from . import JaxModel
from ..likelihood import update_uncertainty
from ...lib.split_channels import split

jax.config.update("jax_enable_x64", True)


class GPModel(JaxModel):
    """Model for Gaussian Process (GP)"""
    def __init__(self, kernel_types, kernel_input_names, lc,
                 gp_code_name='celerite', normalize=False,
                 **kwargs):
        """Initialize the GP model.

        Parameters
        ----------
        kernel_types : list
            The types of GP kernels to use.
        kernel_input_names : list
            The names of the GP kernel inputs.
        lc : eureka.S5_lightcurve_fitting.lightcurve
            The current lightcurve object.
        gp_code : str; optional
            Type GP package to use from ('celerite'),
            by default 'celerite'.
        normalize : bool; optional
            If True, normalize the covariate by mean subtracting it and
            dividing by the standard deviation. By default, False.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.jax_models.JaxModel.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """  # noqa: E501
        # Inherit from JaxModel class
        super().__init__(kernel_types=kernel_types,
                         nkernels=len(kernel_types),
                         kernel_input_names=kernel_input_names,
                         kernel_inputs=None,
                         gp_code_name=gp_code_name, normalize=normalize,
                         fit_lc=np.ones(lc.flux.shape),
                         flux=lc.flux, unc=lc.unc, unc_fit=lc.unc_fit,
                         **kwargs)
        self.name = 'GP'

        # Define model type (physical, systematic, other)
        self.modeltype = 'GP'

        # Do some initial sanity checks and raise errors if needed
        if self.gp_code_name != 'celerite':
            raise AssertionError('Currently celerite2 is the only GP package '
                                 'that can be used with the jax methods.')
        elif self.nkernels > 1:
            raise AssertionError('Our celerite2 implementation cannot compute '
                                 'multi-dimensional GPs, please choose a '
                                 'different GP code.')
        elif self.kernel_types[0] != 'Matern32':
            raise AssertionError('Our celerite2 implementation currently only '
                                 'supports a Matern32 kernel.')

    def setup(self, newparams):
        """Setup a model for evaluation and fitting.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        """
        self.gps = []
        for c in range(self.nchannel_fitted):
            if self.nchannel_fitted > 1:
                chan = self.fitted_channels[c]
            else:
                chan = 0

            # Make the gp object
            gp = self.setup_GP(chan, eval=False)
            self.gps.append(gp)

    def update(self, newparams, **kwargs):
        # Inherit from Model class
        super().update(newparams, **kwargs)

        self.unc_fit = update_uncertainty(newparams, self.nints, self.unc,
                                          self.freenames, self.nchannel_fitted)

    def eval(self, fit_lc, channel=None, gp=None, eval=True, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        fit_lc : ndarray
            The rest of the current model evaluated.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        gp : celerite2.GP; optional
            The input GP object. Defaults to None.
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            Predicted systematics model
        """
        input_gp = gp

        if eval:
            lib = np
        else:
            lib = jnp

        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel, ]

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        lcfinal = lib.zeros(0)
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
                # get flux and uncertainties for current channel
                flux, unc_fit = split([self.flux, self.unc_fit],
                                      self.nints, chan)
                if channel is None:
                    fit_lc_temp = split([fit_lc, ], self.nints, chan)[0]
                else:
                    # If only a specific channel is being evaluated, then only
                    # that channel's fitted mode will be passed in
                    fit_lc_temp = fit_lc
            else:
                chan = 0
                # get flux and uncertainties for current channel
                flux = self.flux
                fit_lc_temp = fit_lc
                unc_fit = self.unc_fit
            residuals = flux-fit_lc_temp

            if self.multwhite:
                time = split([self.time, ], self.nints, chan)[0]
            else:
                time = self.time

            # Remove poorly handled invalid values
            good = lib.isfinite(time)
            unc_fit = unc_fit[good]
            residuals = residuals[good]

            # Create the GP object with current parameters
            if input_gp is None:
                gp = self.setup_GP(chan, eval=eval)
            else:
                gp = input_gp

            kernel_inputs = self.kernel_inputs[chan][0][good]
            gp.compute(kernel_inputs, yerr=unc_fit)
            mu = gp.predict(residuals)

            # Re-insert and mask bad values
            mu_full = np.nan*lib.ones(len(time))
            if eval:
                mu_full[good] = mu
            else:
                # Jax arrays are immutable, so we need to update the array
                # in a different way
                mu_full = mu_full.at[good].set(mu)

            # Append this channel to the outputs
            lcfinal = lib.concatenate([lcfinal, mu_full])

        return lcfinal

    def setup_inputs(self, lib):
        """Setting up kernel inputs as array and standardizing them if asked.

        For details on the benefits of normalization, see e.g.
        Evans et al. 2017.

        Parameters
        ----------
        lib : module
            The library to use (np or jnp).
        """
        self.kernel_inputs = []
        for c in range(self.nchannel_fitted):
            if self.nchannel_fitted > 1:
                chan = self.fitted_channels[c]
            else:
                chan = 0

            if self.multwhite:
                time = split([self.time, ], self.nints, chan)[0]
            else:
                time = self.time

            kernel_inputs_channel = np.zeros((0, time.size))
            for name in self.kernel_input_names:
                if name == 'time':
                    x = np.copy(self.time)
                else:
                    # add more input options here
                    raise ValueError('Currently, only GPs as a function of '
                                     'time are supported, but you have '
                                     'specified a GP as a function of '
                                     f'{name}.')

                if self.multwhite:
                    x = split([x, ], self.nints, chan)[0]

                if self.normalize:
                    x = (x-np.nanmean(x))/np.nanstd(x)

                kernel_inputs_channel = np.append(kernel_inputs_channel,
                                                  x[np.newaxis], axis=0)

            self.kernel_inputs.append(kernel_inputs_channel)

    def setup_GP(self, c=0, eval=True):
        """Set up GP kernels and GP object.

        Parameters
        ----------
        c : int; optional
            The current channel index. Defaults to 0.
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.

        Returns
        -------
        celerite2.GP
            The GP object to use for this fit.
        """
        if c == 0:
            chankey = ''
        else:
            chankey = f'_ch{c}'

        if eval:
            lib = np
            model = self.fit
        else:
            lib = jnp
            model = self.model

        # Parse model attributes as coefficients
        coeffs = np.zeros((self.nkernels, 2)).tolist()
        for i, par in enumerate(['A', 'm']):
            for k in range(self.nkernels):
                if k == 0:
                    kernelkey = ''
                else:
                    kernelkey = str(k)
                parname = f'{par}{kernelkey}{chankey}'
                coeffs[k][i] = getattr(model, parname)

        if self.kernel_inputs is None:
            self.setup_inputs(lib=lib)

        # get the kernel which is the sum of the individual kernel functions
        kernel = self.get_kernel(lib, self.kernel_types[0], coeffs, 0, c)
        for k in range(1, self.nkernels):
            kernel += self.get_kernel(lib, self.kernel_types[k], coeffs, k, c)

        # Make the gp object
        return celerite2.GaussianProcess(kernel, mean=0, fit_mean=False)

    def get_kernel(self, lib, kernel_name, coeffs, k, c=0):
        """Get individual kernels.

        Parameters
        ----------
        lib : module
            The library to use (np or jnp).
        kernel_name : str
            The name of the kernel to get. Currently unused since only
            celerite's Matern32 is supported.
        coeffs : list
            The kernel coefficients (e.g., amplitude and lengthscale).
        k : int
            The kernel number.
        c : int; optional
            The channel index, by default 0.

        Returns
        -------
        kernel
            The requested kernel.
        """
        # get metric and amplitude for the current kernel and channel
        amp = lib.exp(coeffs[k][0])
        metric = lib.exp(coeffs[k][1])

        # Currently only the Matern32 kernel is supported
        kernel = celerite2.terms.Matern32Term(sigma=1, rho=metric)

        # Setting the amplitude
        kernel *= celerite2.terms.RealTerm(a=amp, c=0)

        return kernel

    def loglikelihood(self, fit_lc, channel=None):
        """Compute log likelihood of GP

        Parameters
        ----------
        fit_lc : ndarray
            The fitted model.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.

        Returns
        -------
        float
            log likelihood of the GP evaluated by celerite2
        """
        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel, ]

        logL = 0
        for c in np.arange(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
                # get flux and uncertainties for current channel
                flux, unc_fit = split([self.flux, self.unc_fit],
                                      self.nints, chan)
                if channel is None:
                    fit_temp = split([fit_lc, ], self.nints, chan)[0]
                else:
                    # If only a specific channel is being evaluated, then only
                    # that channel's fitted model will be passed in
                    fit_temp = fit_lc
            else:
                chan = 0
                # get flux and uncertainties for current channel
                flux = self.flux
                fit_temp = fit_lc
                unc_fit = self.unc_fit
            residuals = flux-fit_temp
            if self.multwhite:
                time = split([self.time, ], self.nints, chan)[0]
            else:
                time = self.time

            # Remove poorly handled invalid values
            good = np.isfinite(time)
            unc_fit = unc_fit[good]
            residuals = residuals[good]

            # set up GP with current parameters
            gp = self.setup_GP(chan, eval=True)

            kernel_inputs = self.kernel_inputs[chan][0][good]
            gp.compute(kernel_inputs, yerr=unc_fit)
            logL_temp = gp.log_likelihood(residuals)

            logL += logL_temp

        return logL
