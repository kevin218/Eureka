import numpy as np
import george
from george import kernels
import celerite2

from .Model import Model
from ..likelihood import update_uncertainty
from ...lib.split_channels import split

# tinygp is not supported yet
try:
    import tinygp
except ModuleNotFoundError:
    # tinygp isn't supported yet, so don't throw an exception if it
    # isn't installed
    pass


class GPModel(Model):
    """Model for Gaussian Process (GP)"""
    def __init__(self, kernel_types, kernel_input_names, lc,
                 gp_code_name='celerite', normalize=False,
                 useHODLR=False, **kwargs):
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
            Type GP package to use from ('george', 'celerite'),
            by default 'celerite'.
        normalize : bool; optional
            If True, normalize the covariate by mean subtracting it and
            dividing by the standard deviation. By default, False.
        useHODLR : bool; optional
            If True, use george's HODLRSolver instead of the default solver.
            Only relevant if gp_code is 'george'. By default, False.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from Model class
        super().__init__(kernel_types=kernel_types,
                         nkernels=len(kernel_types),
                         kernel_input_names=kernel_input_names,
                         kernel_inputs=None,
                         gp_code_name=gp_code_name, normalize=normalize,
                         useHODLR=useHODLR, fit_lc=np.ma.ones(lc.flux.shape),
                         flux=lc.flux, unc=lc.unc, unc_fit=lc.unc_fit,
                         **kwargs)
        self.name = 'GP'

        # Define model type (physical, systematic, other)
        self.modeltype = 'GP'

        # Do some initial sanity checks and raise errors if needed
        if self.gp_code_name == 'celerite':
            if self.nkernels > 1:
                raise AssertionError('Celerite2 cannot compute multi-'
                                     'dimensional GPs. Please choose a '
                                     'different GP code')
            elif self.kernel_types[0] != 'Matern32':
                raise AssertionError('Our celerite2 implementation currently '
                                     'only supports a Matern32 kernel.')

        # Setup coefficients
        self.coeffs = np.zeros((self.nchannel_fitted, self.nkernels, 2))
        self._parse_coeffs()

    def _parse_coeffs(self):
        """Convert dict of coefficients into an array."""
        # Parse parameters as coefficients
        for c in range(self.nchannel_fitted):
            if self.nchannel_fitted > 1:
                chan = self.fitted_channels[c]
            else:
                chan = 0

            if chan == 0:
                chankey = ''
            else:
                chankey = f'_ch{chan}'

            for i, par in enumerate(['A', 'm']):
                for k in range(self.nkernels):
                    if k == 0:
                        kernelkey = ''
                    else:
                        kernelkey = str(k)

                    try:
                        index = f'{par}{kernelkey}{chankey}'
                        self.coeffs[c, k, i] = self.parameters.dict[index][0]
                    except KeyError:
                        pass

    def update(self, newparams, **kwargs):
        # Inherit from Model class
        super().update(newparams, **kwargs)

        self.unc_fit = update_uncertainty(newparams, self.nints, self.unc,
                                          self.freenames, self.nchannel_fitted)

    def eval(self, fit_lc, channel=None, gp=None, **kwargs):
        """Compute GP with the given parameters

        Parameters
        ----------
        fit_lc : ndarray
            The rest of the current model evaluated.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        gp : celerite2.GP, george.GP, or tinygp.GaussianProcess; optional
            The current GP object.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            Predicted systematics model
        """
        input_gp = gp

        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel, ]

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        lcfinal = np.ma.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
                # get flux and uncertainties for current channel
                flux, unc_fit = split([self.flux, self.unc_fit],
                                      self.nints, chan)
                if nchan > 1:
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
            residuals = np.ma.masked_invalid(flux-fit_temp)
            if self.multwhite:
                time = split([self.time, ], self.nints, chan)[0]
            else:
                time = self.time
            residuals = np.ma.masked_where(time.mask, residuals)

            # Remove poorly handled masked values
            good = ~np.ma.getmaskarray(residuals)
            unc_fit = unc_fit[good]
            residuals = residuals[good]

            # Create the GP object with current parameters
            if input_gp is None:
                gp = self.setup_GP(chan)
            else:
                gp = input_gp

            if self.gp_code_name == 'george':
                gp.compute(self.kernel_inputs[chan][:, good].T, unc_fit)
                mu = gp.predict(residuals, self.kernel_inputs[chan][:, good].T,
                                return_cov=False)
            elif self.gp_code_name == 'celerite':
                kernel_inputs = self.kernel_inputs[chan][0][good]
                gp.compute(kernel_inputs, yerr=unc_fit)
                mu = gp.predict(residuals)
            elif self.gp_code_name == 'tinygp':
                cond_gp = gp.condition(residuals, noise=unc_fit).gp
                mu = cond_gp.loc

            # Re-insert and mask bad values
            mu_full = np.ma.zeros(len(time))
            mu_full[good] = mu
            mu_full = np.ma.masked_where(~good, mu_full)

            # Append to the full list
            lcfinal = np.ma.append(lcfinal, mu_full)

        return lcfinal

    def setup_inputs(self):
        """Setting up kernel inputs as array and standardizing them if asked.

        For details on the benefits of normalization, see e.g.
        Evans et al. 2017.
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

            kernel_inputs_channel = np.ma.zeros((0, time.size))
            for name in self.kernel_input_names:
                if name == 'time':
                    x = np.ma.copy(self.time)
                else:
                    # add more input options here
                    raise ValueError('Currently, only GPs as a function of '
                                     'time are supported, but you have '
                                     'specified a GP as a function of '
                                     f'{name}.')

                if self.multwhite:
                    x = split([x, ], self.nints, chan)[0]

                if self.normalize:
                    x = (x-np.ma.mean(x))/np.ma.std(x)

                kernel_inputs_channel = np.ma.append(kernel_inputs_channel,
                                                     x[np.newaxis], axis=0)

            self.kernel_inputs.append(kernel_inputs_channel)

    def setup_GP(self, c=0):
        """Set up GP kernels and GP object.

        Parameters
        ----------
        c : int; optional
            The current channel index. Defaults to 0.

        Returns
        -------
        celerite2.GP, george.GP, or tinygp.GaussianProcess
            The GP object to use for this fit.
        """
        # Parse parameters as coefficients
        self._parse_coeffs()

        if self.kernel_inputs is None:
            self.setup_inputs()

        # get the kernel which is the sum of the individual kernel functions
        kernel = self.get_kernel(self.kernel_types[0], 0, c)
        for k in range(1, self.nkernels):
            kernel += self.get_kernel(self.kernel_types[k], k, c)

        # Make the gp object
        if self.gp_code_name == 'george':
            if self.useHODLR:
                solver = george.solvers.HODLRSolver
            else:
                solver = None
            gp = george.GP(kernel, mean=0, fit_mean=False, solver=solver)
        elif self.gp_code_name == 'celerite':
            gp = celerite2.GaussianProcess(kernel, mean=0, fit_mean=False)
        elif self.gp_code_name == 'tinygp':
            if self.nchannel_fitted > 1:
                chan = self.fitted_channels[c]
            else:
                chan = 0
            gp = tinygp.GaussianProcess(kernel, self.kernel_inputs[chan].T,
                                        mean=0)

        return gp

    def get_kernel(self, kernel_name, k, c=0):
        """Get individual kernels.

        Parameters
        ----------
        kernel_name : str
            The name of the kernel to get.
        k : int
            The kernel number.
        c : int; optional
            The channel index, by default 0.

        Returns
        -------
        kernel
            The requested kernel.

        Raises
        ------
        AssertionError
            george and tinygp currently only support the Matern32, ExpSquared,
            RationalQuadratic, and Exp kernels.
        """
        if self.gp_code_name == 'george':
            # get metric and amplitude for the current kernel and channel
            amp = np.exp(self.coeffs[c, k, 0])
            metric = np.exp(self.coeffs[c, k, 1]*2)

            if kernel_name == 'Matern32':
                kernel = amp*kernels.Matern32Kernel(
                    metric, ndim=self.nkernels, axes=k)
            elif kernel_name == 'ExpSquared':
                kernel = amp*kernels.ExpSquaredKernel(
                    metric, ndim=self.nkernels, axes=k)
            elif kernel_name == 'RationalQuadratic':
                kernel = amp*kernels.RationalQuadraticKernel(
                    log_alpha=1, metric=metric, ndim=self.nkernels, axes=k)
            elif kernel_name == 'Exp':
                kernel = amp*kernels.ExpKernel(
                    metric, ndim=self.nkernels, axes=k)
            else:
                raise AssertionError(f'The kernel {kernel_name} is not in the '
                                     'currently supported list of kernels for '
                                     'george which includes:\nMatern32, '
                                     'ExpSquared, RationalQuadratic, Exp.')
        elif self.gp_code_name == 'celerite':
            # get metric and amplitude for the current kernel and channel
            amp = np.exp(self.coeffs[c, k, 0])
            metric = np.exp(self.coeffs[c, k, 1])

            kernel = celerite2.terms.Matern32Term(sigma=1, rho=metric)
            # Setting the amplitude
            kernel *= celerite2.terms.RealTerm(a=amp, c=0)
        elif self.gp_code_name == 'tinygp':
            # get metric and amplitude for the current kernel and channel
            amp = np.exp(self.coeffs[c, k, 0])
            metric = np.exp(self.coeffs[c, k, 1]*2)

            if kernel_name == 'Matern32':
                kernel = amp*tinygp.kernels.Matern32(metric)
            elif kernel_name == 'ExpSquared':
                kernel = amp*tinygp.kernels.ExpSquared(metric)
            elif kernel_name == 'RationalQuadratic':
                kernel = amp*tinygp.kernels.RationalQuadratic(alpha=1,
                                                              scale=metric)
            elif kernel_name == 'Exp':
                kernel = amp*tinygp.kernels.Exp(metric)
            else:
                raise AssertionError(f'The kernel {kernel_name} is not in the '
                                     'currently supported list of kernels for '
                                     'tinygp which includes:\nMatern32, '
                                     'ExpSquared, RationalQuadratic, Exp.')

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
            log likelihood of the GP evaluated by george/tinygp/celerite2
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
            residuals = np.ma.masked_invalid(flux-fit_temp)
            if self.multwhite:
                time = split([self.time, ], self.nints, chan)[0]
            else:
                time = self.time
            residuals = np.ma.masked_where(time.mask, residuals)

            # Remove poorly handled masked values
            good = ~np.ma.getmaskarray(residuals)
            unc_fit = unc_fit[good]
            residuals = residuals[good]

            # set up GP with current parameters
            gp = self.setup_GP(chan)

            if self.gp_code_name == 'george':
                gp.compute(self.kernel_inputs[chan][:, good].T, unc_fit)
                logL_temp = gp.lnlikelihood(residuals, quiet=True)
            elif self.gp_code_name == 'celerite':
                kernel_inputs = self.kernel_inputs[chan][0][good]
                gp.compute(kernel_inputs, yerr=unc_fit)
                logL_temp = gp.log_likelihood(residuals)
            elif self.gp_code_name == 'tinygp':
                cond = gp.condition(residuals, diag=unc_fit)
                logL_temp = cond.log_probability
            logL += logL_temp

        return logL
