import numpy as np
import george
from george import kernels
import celerite
from .Model import Model
from ...lib.readEPF import Parameters

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
                 **kwargs):
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
        self.kernel_types = kernel_classes
        self.kernel_input_names = kernel_inputs
        self.kernel_input_arrays = []
        self.nkernels = len(kernel_classes)
        self.flux = lc.flux
        self.unc_fit = lc.unc
        self.time = lc.time

        # Check for Parameters instance
        self.parameters = kwargs.get('parameters')

        # Generate parameters from kwargs if necessary
        if self.parameters is None:
            self.parameters = Parameters(**kwargs)

        # Set parameters for multi-channel fits
        self.longparamlist = kwargs.get('longparamlist')
        self.nchan = kwargs.get('nchan')
        self.paramtitles = kwargs.get('paramtitles')

        # Update coefficients
        self._parse_coeffs()

    def _parse_coeffs(self):
        """Convert dict of coefficients into a list
        of coefficients in increasing order.
        """
        # Parse keyword arguments as coefficients
        self.coeffs = {}
        for i in range(self.nkernels):
            self.coeffs[self.kernel_types[i]] = []
        for k, v in self.parameters.dict.items():
            if k.startswith('A'):
                remvisnum = k.split('_')
                if len(remvisnum) > 1:
                    self.coeffs['A_%i' % int(remvisnum[1])] = v[0]
                elif self.nchan > 1:
                    self.coeffs['A_0'] = v[0]
                else:
                    self.coeffs['A'] = v[0]
            if k.lower().startswith('m'):
                remvisnum = k.split('_')
                if len(remvisnum) > 1 or self.nchan > 1:
                    no = int(remvisnum[0][1])-1
                    if no < 0:
                        raise AssertionError('Please start your metric '
                                             'enumeration with m1.')
                    self.coeffs[self.kernel_types[no]].append(v[0])
                else:
                    no = int(remvisnum[0][1])-1
                    self.coeffs[self.kernel_types[no]].append(v[0])
            if k.startswith('WN'):
                remvisnum = k.split('_')
                if len(remvisnum) > 1:
                    self.coeffs['WN_%i' % int(remvisnum[1])] = v[0]
                elif self.nchan > 1:
                    self.coeffs['WN_0'] = v[0]
                else:
                    self.coeffs['WN'] = v[0]
                if 'fixed' in v:
                    self.fit_white_noise = False
                else:
                    self.fit_white_noise = True

    def eval(self, fit, gp=None, **kwargs):
        """Compute GP with the given parameters

        Parameters
        ----------
        fit : eureka.S5_lightcurve_fitting.models.Model
            Current model (i.e. transit model)
        gp : celerite.GP, george.GP, or tinygp.GaussianProcess; optional
            The current GP object.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            Predicted systematics model
        """
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        lcfinal = np.array([])

        for c in np.arange(self.nchan):

            # get flux and uncertainties for current channel
            c_flux = self.flux[len(self.time)*c:len(self.time)*(c+1)]
            c_fit = fit[len(self.time)*c:len(self.time)*(c+1)]
            c_unc_fit = self.unc_fit[len(self.time)*c:len(self.time)*(c+1)]

            # Create the GP object with current parameters
            if gp is None:
                gp = self.setup_GP(c)

            if self.nkernels > 1:
                if self.gp_code_name == 'george':
                    gp.compute(self.kernel_input_arrays.T, c_unc_fit)
                    mu, cov = gp.predict(c_flux-c_fit,
                                         self.kernel_input_arrays.T)

                if self.gp_code_name == 'tinygp':
                    cond_gp = gp.condition(c_flux-c_fit, noise=c_unc_fit).gp
                    mu, cov = cond_gp.loc, cond_gp.variance

                if self.gp_code_name == 'celerite':
                    raise AssertionError('Celerite cannot compute '
                                         'multi-dimensional GPs, please choose'
                                         ' a different GP code')
            else:
                if self.gp_code_name == 'george':
                    gp.compute(self.kernel_input_arrays[0], c_unc_fit)
                    mu, cov = gp.predict(c_flux-c_fit,
                                         self.kernel_input_arrays[0])

                if self.gp_code_name == 'tinygp':
                    cond_gp = gp.condition(c_flux-c_fit, noise=c_unc_fit).gp
                    mu, cov = cond_gp.loc, cond_gp.variance

                if self.gp_code_name == 'celerite':
                    gp.compute(self.kernel_input_arrays[0], c_unc_fit)
                    mu, cov = gp.predict(c_flux-c_fit,
                                         self.kernel_input_arrays[0],
                                         return_var=True)
            lcfinal = np.append(lcfinal, mu)

        return lcfinal  # , cov

    def set_inputs(self, normalise=False):
        """Setting up kernel inputs as array and standardizing them
        see e.g. Evans et al. 2017.

        Parameters
        ----------
        normalise : bool; optional
            Standardize kernels following Evans+2017. Defaults to False.
        """
        kernel_inputs = []
        for i in self.kernel_input_names:
            if i == 'time':
                x = self.time
                kernel_inputs.append(x)
            # add more input options here

        if normalise:
            norm_kernel_inputs = [(i-i.mean())/i.std() for i in kernel_inputs]
            self.kernel_input_arrays = np.array(norm_kernel_inputs)
        else:
            self.kernel_input_arrays = np.array(kernel_inputs)

    def setup_GP(self, channel):
        """Set up GP kernels and GP object.

        Parameters
        ----------
        channel : int
            The current channel number.

        Returns
        -------
        celerite.GP, george.GP, or tinygp.GaussianProcess
            The GP object to use for this fit.
        """
        if len(self.kernel_input_arrays) == 0:
            self.set_inputs()

        if self.nchan > 1:
            # get the kernel which is the sum of the individual
            # kernel functions
            for i in range(self.nkernels):
                if i == 0:
                    kernel = self.get_kernel(self.kernel_types[i], i, channel)
                else:
                    kernel += self.get_kernel(self.kernel_types[i], i, channel)

            if self.gp_code_name == 'celerite':
                # adding the amplitude
                amps = self.coeffs['A_%i' % channel]
                kernel *= celerite.terms.RealTerm(log_a=amps,
                                                  log_c=0)

                # adding the jitter/white noise term
                wn = self.coeffs['WN_%i' % channel]
                kernel += celerite.terms.JitterTerm(log_sigma=wn)

                # make gp object
                gp = celerite.GP(kernel, mean=0, fit_mean=False)

            if self.gp_code_name == 'george':
                # adding the amplitude
                amps = self.coeffs['A_%i' % channel]
                kernel *= kernels.ConstantKernel(amps, ndim=self.nkernels,
                                                 axes=np.arange(self.nkernels))

                # make gp object
                wn = self.coeffs['WN_%i' % channel]
                gp = george.GP(kernel, white_noise=wn,
                               fit_white_noise=self.fit_white_noise, mean=0,
                               fit_mean=False)
                #                solver=george.solvers.HODLRSolver)

            if self.gp_code_name == 'tinygp':
                # adding the amplitude
                amps = self.coeffs['A_%i' % channel]
                kernel *= tinygp.kernels.Constant(amps)

                # make gp object
                wn2 = self.coeffs['WN_%i' % channel]**2
                gp = tinygp.GaussianProcess(kernel,
                                            self.kernel_input_arrays.T,
                                            diag=wn2, mean=0)
        else:
            # get the kernel which is the sum of the individual
            # kernel fucntions
            for i in range(self.nkernels):
                if i == 0:
                    kernel = self.get_kernel(self.kernel_types[i], i)
                else:
                    kernel += self.get_kernel(self.kernel_types[i], i)

            if self.gp_code_name == 'celerite':
                # adding the amplitude
                amp = self.coeffs['A']
                kernel *= celerite.terms.RealTerm(log_a=amp, log_c=0)

                # adding the jitter/white noise term
                wn = self.coeffs['WN']
                kernel += celerite.terms.JitterTerm(log_sigma=wn)

                # make gp object
                gp = celerite.GP(kernel, mean=0, fit_mean=False)

            if self.gp_code_name == 'george':
                # adding the amplitude
                amp = self.coeffs['A']
                kernel *= kernels.ConstantKernel(amp,
                                                 ndim=self.nkernels,
                                                 axes=np.arange(self.nkernels))

                # make gp object
                wn = self.coeffs['WN']
                gp = george.GP(kernel, white_noise=wn,
                               fit_white_noise=self.fit_white_noise, mean=0,
                               fit_mean=False)
                #                solver=george.solvers.HODLRSolver)

            if self.gp_code_name == 'tinygp':
                # adding the amplitude
                amp = self.coeffs['A']
                kernel *= tinygp.kernels.Constant(amp)

                # make gp object
                wn2 = self.coeffs['WN']**2
                gp = tinygp.GaussianProcess(kernel,
                                            self.kernel_input_arrays.T,
                                            diag=wn2, mean=0)

        return gp

    def loglikelihood(self, fit, unc_fit):
        """Compute log likelihood of GP

        Parameters
        ----------
        fit : ndarray
            The fitted model.
        unc_fit : ndarray
            The fitted uncertainty.

        Returns
        -------
        float
            log likelihood of the GP evaluated by george/tinygp/celerite
        """
        # update uncertainty
        self.unc_fit = unc_fit

        logL = []

        for c in np.arange(self.nchan):
            # set up GP with current parameters
            gp = self.setup_GP(c)

            # get fluxes, current fit and uncertainties for channel
            c_flux = self.flux[len(self.time)*c:len(self.time)*(c+1)]
            c_fit = fit[len(self.time)*c:len(self.time)*(c+1)]
            c_unc_fit = self.unc_fit[len(self.time)*c:len(self.time)*(c+1)]

            if self.gp_code_name == 'celerite':
                if self.nkernels > 1:
                    raise AssertionError('Celerite cannot compute '
                                         'multi-dimensional GPs, please choose'
                                         ' a different GP code')
                else:
                    gp.compute(self.kernel_input_arrays[0], c_unc_fit)
                logL.append(gp.log_likelihood(c_flux - c_fit))

            if self.gp_code_name == 'george':
                if self.nkernels > 1:
                    gp.compute(self.kernel_input_arrays.T, c_unc_fit)
                else:
                    gp.compute(self.kernel_input_arrays[0], c_unc_fit)
                logL.append(gp.lnlikelihood(c_flux - c_fit, quiet=True))

            if self.gp_code_name == 'tinygp':
                cond = gp.condition(c_flux - c_fit)
                logL.append(cond.log_probability)

        return sum(logL)

    def get_kernel(self, kernel_name, i, channel=0):
        """Get individual kernels.

        Parameters
        ----------
        kernel_name : str
            The name of the kernel to get.
        i : int
            The kernel number.
        channel : int; optional
            The channel number, by default 0.

        Returns
        -------
        kernel
            The requested george, celerite, or tinygp kernel.

        Raises
        ------
        AssertionError
            Celerite currently only supports a Matern32 kernel.
        """
        # get metric for the individual kernel for the current channel
        metric = (1./np.exp(self.coeffs[kernel_name][channel]))**2

        if self.gp_code_name == 'george':
            if kernel_name == 'Matern32':
                kernel = kernels.Matern32Kernel(metric, ndim=self.nkernels,
                                                axes=i)
            if kernel_name == 'ExpSquared':
                kernel = kernels.ExpSquaredKernel(metric, ndim=self.nkernels,
                                                  axes=i)
            if kernel_name == 'RationalQuadratic':
                kernel = kernels.RationalQuadraticKernel(log_alpha=1,
                                                         metric=metric,
                                                         ndim=self.nkernels,
                                                         axes=i)
            if kernel_name == 'Exp':
                kernel = kernels.ExpKernel(metric, ndim=self.nkernels, axes=i)

        if self.gp_code_name == 'tinygp':
            if kernel_name == 'Matern32':
                kernel = tinygp.kernels.Matern32(metric)
            if kernel_name == 'ExpSquared':
                kernel = tinygp.kernels.ExpSquared(metric)
            if kernel_name == 'RationalQuadratic':
                kernel = tinygp.kernels.RationalQuadratic(alpha=1,
                                                          scale=metric)
            if kernel_name == 'Exp':
                kernel = tinygp.kernels.Exp(metric)

        if self.gp_code_name == 'celerite':
            if kernel_name == 'Matern32':
                kernel = celerite.terms.Matern32Term(log_sigma=1,
                                                     log_rho=metric)
            else:
                raise AssertionError('Celerite currently only supports a '
                                     'Matern32 kernel')

        return kernel
