import numpy as np
import george
from george import kernels
import celerite

from .Model import Model
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
    def __init__(self, kernel_classes, kernel_inputs, lc, gp_code='george',
                 normalize=False, **kwargs):
        """Initialize the GP model.

        Parameters
        ----------
        kernel_classes : list
            The types of GP kernels to use.
        kernel_inputs : list
            The names of the GP kernel inputs.
        lc : eureka.S5_lightcurve_fitting.lightcurve
            The current lightcurve object.
        gp_code : str; optional
            Type GP package to use from ('george', 'celerite'),
            by default 'george'.
        normalize : bool; optional
            If True, normalize the covariate by mean subtracting it and
            dividing by the standard deviation. By default, False.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from Model class
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'GP'

        # Get GP parameters
        self.gp_code_name = gp_code
        self.normalize = normalize
        self.kernel_types = kernel_classes
        self.kernel_input_names = kernel_inputs
        self.kernel_inputs = []
        self.nkernels = len(kernel_classes)
        self.flux = lc.flux
        self.unc_fit = lc.unc_fit
        self.time = lc.time

        if self.nchannel_fitted > 1:
            raise AssertionError('The GP model cannot currently be used '
                                 'when fitting multiple channels '
                                 'simultaneously!')

        if self.nkernels > 1 and self.gp_code_name == 'celerite':
            raise AssertionError('Celerite cannot compute multi-dimensional '
                                 'GPs, please choose a different GP code')

        # Update coefficients
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
                chankey = f'_{chan}'

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

    def eval(self, fit, channel=None, gp=None, **kwargs):
        """Compute GP with the given parameters

        Parameters
        ----------
        fit : eureka.S5_lightcurve_fitting.models.Model
            Current model (i.e. transit model)
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        gp : celerite.GP, george.GP, or tinygp.GaussianProcess; optional
            The current GP object.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            Predicted systematics model
        """
        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel, ]

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        lcfinal = np.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
                # get flux and uncertainties for current channel
                flux = split([self.flux, ], self.nints, chan)[0]
                fit = split([fit, ], self.nints, chan)[0]
                unc_fit = split([self.unc_fit, ], self.nints, chan)[0]
            else:
                chan = 0
                # get flux and uncertainties for current channel
                flux = self.flux
                fit = fit
                unc_fit = self.unc_fit
            residuals = flux-fit

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            # Create the GP object with current parameters
            if gp is None:
                gp = self.setup_GP(chan)

            if self.gp_code_name == 'george':
                gp.compute(self.kernel_inputs.T, unc_fit)
                mu = gp.predict(residuals, self.kernel_inputs.T,
                                return_cov=False)
            elif self.gp_code_name == 'celerite':
                gp.compute(self.kernel_inputs[0], unc_fit)
                mu = gp.predict(residuals, self.kernel_inputs[0])
            elif self.gp_code_name == 'tinygp':
                cond_gp = gp.condition(residuals, noise=unc_fit).gp
                mu = cond_gp.loc
            lcfinal = np.append(lcfinal, mu)

        return lcfinal

    def setup_inputs(self):
        """Setting up kernel inputs as array and standardizing them if asked.
        
        For details on the benefits of normalization, see e.g.
        Evans et al. 2017.
        """
        kernel_inputs = []
        for name in self.kernel_input_names:
            if name == 'time':
                x = self.time
                kernel_inputs.append(x)
            else:
                # add more input options here
                raise ValueError('Currently only GPs as a function of time '
                                 'are supported, but you have specified a GP '
                                 f'as a function of {name}.')

        if self.normalize:
            kernel_inputs = [(input-input.mean())/input.std()
                             for input in kernel_inputs]
        self.kernel_inputs = np.array(kernel_inputs)
        if len(self.kernel_inputs.shape) == 1:
            self.kernel_inputs = self.kernel_inputs[np.newaxis]

    def setup_GP(self, c=0):
        """Set up GP kernels and GP object.

        Parameters
        ----------
        c : int; optional
            The current channel index. Defaults to 0.

        Returns
        -------
        celerite.GP, george.GP, or tinygp.GaussianProcess
            The GP object to use for this fit.
        """
        if len(self.kernel_inputs) == 0:
            self.setup_inputs()

        # get the kernel which is the sum of the individual kernel functions
        kernel = self.get_kernel(self.kernel_types[0], 0, c)
        for k in range(self.nkernels):
            kernel += self.get_kernel(self.kernel_types[k], k, c)

        # Make the gp object
        if self.gp_code_name == 'george':
            gp = george.GP(kernel, mean=0, fit_mean=False)
            #                solver=george.solvers.HODLRSolver)
        elif self.gp_code_name == 'celerite':
            gp = celerite.GP(kernel, mean=0, fit_mean=False)
        elif self.gp_code_name == 'tinygp':
            gp = tinygp.GaussianProcess(kernel, self.kernel_inputs.T,
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
            The requested george, celerite, or tinygp kernel.

        Raises
        ------
        AssertionError
            Celerite currently only supports a Matern32 kernel.
        """
        # get metric and amplitude for the current kernel and channel
        amp = self.coeffs[c, k, 0]
        metric = (1./np.exp(self.coeffs[c, k, 1]))**2

        if self.gp_code_name == 'george':
            if kernel_name == 'Matern32':
                kernel = kernels.Matern32Kernel(metric, ndim=self.nkernels,
                                                axes=k)
            elif kernel_name == 'ExpSquared':
                kernel = kernels.ExpSquaredKernel(metric, ndim=self.nkernels,
                                                  axes=k)
            elif kernel_name == 'RationalQuadratic':
                kernel = kernels.RationalQuadraticKernel(log_alpha=1,
                                                         metric=metric,
                                                         ndim=self.nkernels,
                                                         axes=k)
            elif kernel_name == 'Exp':
                kernel = kernels.ExpKernel(metric, ndim=self.nkernels, axes=k)
            else:
                raise AssertionError(f'The kernel {kernel_name} is not in the '
                                     'currently supported list of kernels for '
                                     'george which includes:\nMatern32, '
                                     'ExpSquared, RationalQuadratic, Exp.')

            # Setting the amplitude
            kernel *= kernels.ConstantKernel(amp, ndim=self.nkernels, axes=k)
        elif self.gp_code_name == 'celerite':
            if kernel_name == 'Matern32':
                kernel = celerite.terms.Matern32Term(log_sigma=1,
                                                     log_rho=metric)
            else:
                raise AssertionError('Celerite currently only supports a '
                                     'Matern32 kernel')

            # Setting the amplitude
            kernel *= celerite.terms.RealTerm(log_a=amp, log_c=0)
        elif self.gp_code_name == 'tinygp':
            if kernel_name == 'Matern32':
                kernel = tinygp.kernels.Matern32(metric)
            elif kernel_name == 'ExpSquared':
                kernel = tinygp.kernels.ExpSquared(metric)
            elif kernel_name == 'RationalQuadratic':
                kernel = tinygp.kernels.RationalQuadratic(alpha=1,
                                                          scale=metric)
            elif kernel_name == 'Exp':
                kernel = tinygp.kernels.Exp(metric)
            else:
                raise AssertionError(f'The kernel {kernel_name} is not in the '
                                     'currently supported list of kernels for '
                                     'tinygp which includes:\nMatern32, '
                                     'ExpSquared, RationalQuadratic, Exp.')

            # Setting the amplitude
            kernel *= tinygp.kernels.Constant(amp)

        return kernel

    def loglikelihood(self, fit, unc_fit, channel=None):
        """Compute log likelihood of GP

        Parameters
        ----------
        fit : ndarray
            The fitted model.
        unc_fit : ndarray
            The fitted uncertainty.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.

        Returns
        -------
        float
            log likelihood of the GP evaluated by george/tinygp/celerite
        """
        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel, ]

        # update uncertainty
        self.unc_fit = unc_fit

        logL = []

        for c in np.arange(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
                # get flux and uncertainties for current channel
                flux = split([self.flux, ], self.nints, chan)[0]
                fit = split([fit, ], self.nints, chan)[0]
                unc_fit = split([self.unc_fit, ], self.nints, chan)[0]
            else:
                chan = 0
                # get flux and uncertainties for current channel
                flux = self.flux
                fit = fit
                unc_fit = self.unc_fit
            residuals = flux-fit

            # set up GP with current parameters
            gp = self.setup_GP(chan)

            if self.gp_code_name == 'george':
                gp.compute(self.kernel_inputs.T, unc_fit)
                logL_temp = gp.lnlikelihood(residuals, quiet=True)
            elif self.gp_code_name == 'celerite':
                gp.compute(self.kernel_inputs[0], unc_fit)
                logL_temp = gp.log_likelihood(residuals)
            elif self.gp_code_name == 'tinygp':
                cond = gp.condition(residuals, diag=unc_fit)
                logL_temp = cond.log_probability
            logL.append(logL_temp)

        return sum(logL)

