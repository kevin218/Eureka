import numpy as np
import george
from george import kernels
import celerite2

from .Model import Model
from ..likelihood import update_uncertainty
from ...lib.split_channels import split

try:
    import tinygp
except ModuleNotFoundError:
    # tinygp is optional and isn't supported yet, so don't throw an exception
    # if it isn't installed
    tinygp = None


class GPModel(Model):
    """Model for Gaussian Process (GP)"""
    def __init__(self, kernel_types, kernel_input_names, lc,
                 gp_code_name='celerite', normalize=False,
                 useHODLR=False, **kwargs):
        """Initialize the GP model.

        Parameters
        ----------
        kernel_types : list[str]
            The types of GP kernels to use (e.g., ['Matern32']).
        kernel_input_names : list[str]
            Names of GP inputs (currently only 'time' is supported).
        lc : eureka.S5_lightcurve_fitting.lightcurve
            The current lightcurve object.
        gp_code_name : {'george','celerite','tinygp'}; optional
            GP backend. Default is 'celerite'.
        normalize : bool; optional
            If True, standardize inputs (mean 0, std 1). Default False.
            For details on the benefits of normalization, see e.g.
            Evans et al. 2017.
        useHODLR : bool; optional
            If True and gp_code_name == 'george', use george's HODLRSolver.
            Default is False.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
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
                raise AssertionError(
                    'celerite2 cannot compute multi-dimensional GPs. '
                    'Use a single kernel or a different GP backend.'
                )
            if self.kernel_types[0] != 'Matern32':
                raise AssertionError(
                    'Our celerite2 implementation currently supports only '
                    'a Matern32 kernel.'
                )

    def update(self, newparams, **kwargs):
        """Update parameters and refresh uncertainty fit array."""
        # Inherit from Model class
        super().update(newparams, **kwargs)
        self.unc_fit = update_uncertainty(newparams, self.nints, self.unc,
                                          self.freenames, self.nchannel_fitted)

    def eval(self, fit_lc, channel=None, gp=None, **kwargs):
        """Compute the GP predictive mean of residuals.

        Parameters
        ----------
        fit_lc : ndarray
            The current (non-GP) model evaluation.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        gp : celerite2.GP, george.GP, or tinygp.GaussianProcess; optional
            Pre-built GP object to reuse; if None, a new one is created.
        **kwargs : dict
            Must include 'time' if self.time is None.

        Returns
        -------
        lcfinal : np.ma.MaskedArray
            Predicted GP systematics (same shape as time).
        """
        input_gp = gp
        nchan, channels = self._channels(channel)

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        lcfinal = np.ma.array([])
        for chan in channels:
            if self.nchannel_fitted > 1:
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
            unc_good = unc_fit[good]
            res_good = residuals[good]

            # Build or reuse the GP object
            if input_gp is None:
                gp = self.setup_GP(chan)
            else:
                gp = input_gp
                # If caller passed a pre-built GP, we may still need
                # to the compute the inputs.
                if self.kernel_inputs is None:
                    self.setup_inputs()

            if self.gp_code_name == 'george':
                kin = self.kernel_inputs[chan][:, good].T
                gp.compute(kin, unc_good)
                mu = gp.predict(res_good, kin, return_cov=False)
            elif self.gp_code_name == 'celerite':
                kin = self.kernel_inputs[chan][0][good]
                gp.compute(kin, yerr=unc_good)
                mu = gp.predict(res_good)
            elif self.gp_code_name == 'tinygp':
                if tinygp is None:
                    raise RuntimeError('tinygp is not available.')
                cond_gp = gp.condition(res_good, noise=unc_good).gp
                mu = cond_gp.loc
            else:
                raise ValueError(f'Unknown gp_code_name: {self.gp_code_name}')

            # Re-insert and mask bad values
            mu_full = np.ma.zeros(len(time))
            mu_full[good] = mu
            mu_full = np.ma.masked_where(~good, mu_full)

            # Append to the full list
            lcfinal = np.ma.append(lcfinal, mu_full)

        return lcfinal

    def setup_inputs(self):
        """Build kernel input arrays; standardize if requested.

        Currently supports only 'time' as an input dimension. When
        normalize=True, inputs are standardized to zero mean and unit std.
        """
        # Store by real channel id to avoid index mismatches.
        self.kernel_inputs = {}
        for chan in self.fitted_channels:
            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            kin_chan = np.ma.zeros((0, time.size))
            for name in self.kernel_input_names:
                if name == 'time':
                    x = np.ma.copy(self.time)
                else:
                    # add more input options here
                    raise ValueError(
                        'Only time-based GPs are currently supported; '
                        f"received '{name}'."
                    )

                if self.multwhite:
                    x = split([x, ], self.nints, chan)[0]

                if self.normalize:
                    x = (x-np.ma.mean(x))/np.ma.std(x)

                kin_chan = np.ma.append(kin_chan, x[np.newaxis], axis=0)

            self.kernel_inputs[chan] = kin_chan

    def setup_GP(self, chan=0):
        """Construct the GP object for channel index c.

        Parameters
        ----------
        chan : int; optional
            The current channel index. Defaults to 0.

        Returns
        -------
        celerite2.GaussianProcess, george.GP, or tinygp.GaussianProcess
            The GP instance for the requested backend.
        """
        if self.kernel_inputs is None:
            self.setup_inputs()

        # Build kernel as a sum over per-kernel components
        kernel = self.get_kernel(self.kernel_types[0], 0, chan)
        for k in range(1, self.nkernels):
            kernel += self.get_kernel(self.kernel_types[k], k, chan)

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
            if tinygp is None:
                raise RuntimeError('tinygp is not available.')
            gp = tinygp.GaussianProcess(kernel, self.kernel_inputs[chan].T,
                                        mean=0)
        else:
            raise ValueError(f'Unknown gp_code_name: {self.gp_code_name}')

        return gp

    def get_kernel(self, kernel_name, k, chan=0):
        """Return a backend-specific kernel instance.

        Parameters
        ----------
        kernel_name : str
            Kernel type ('Matern32', 'ExpSquared', 'RationalQuadratic',
            'Exp').
        k : int
            Kernel index (0-based).
        chan : int; optional
            The current channel index. Defaults to 0.

        Returns
        -------
        kernel
            The requested kernel.

        Raises
        ------
        AssertionError
            When an unsupported kernel/backend combination is requested.
        """
        # Read per-kernel, per-channel params on demand using suffix rules.
        # A{ki}, m{ki} where ki = '' for k==0 else '1','2',...
        ki = '' if k == 0 else str(k)
        amp_log = self._get_param_value(f'A{ki}', chan=chan)
        metric_log = self._get_param_value(f'm{ki}', chan=chan)

        if self.gp_code_name == 'george':
            amp = np.exp(amp_log)
            metric = np.exp(metric_log*2)

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
                raise AssertionError(
                    f'Unsupported kernel for george: {kernel_name}. '
                    'Supported: Matern32, ExpSquared, RationalQuadratic, Exp.'
                )
        elif self.gp_code_name == 'celerite':
            # celerite2: Matern32 term with sigma, rho
            sigma = np.sqrt(np.exp(amp_log))
            rho = np.exp(metric_log)
            if kernel_name != 'Matern32':
                raise AssertionError('celerite2 path only supports Matern32,'
                                     f' got {kernel_name}.')
            kernel = celerite2.terms.Matern32Term(sigma=sigma, rho=rho)
        elif self.gp_code_name == 'tinygp':
            if tinygp is None:
                raise RuntimeError('tinygp is not available.')
            amp = np.exp(amp_log)
            metric = np.exp(metric_log*2)

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
                raise AssertionError(
                    f'Unsupported kernel for tinygp: {kernel_name}. '
                    'Supported: Matern32, ExpSquared, '
                    'RationalQuadratic, Exp.')
        else:
            raise ValueError(f'Unknown gp_code_name: {self.gp_code_name}')

        return kernel

    def loglikelihood(self, fit_lc, channel=None):
        """Compute the GP log-likelihood.

        Parameters
        ----------
        fit_lc : ndarray
            The fitted (non-GP) model.
        channel : int; optional
            If provided, evaluate only that channel. Defaults to None.

        Returns
        -------
        float
            Log-likelihood from the selected GP backend.
        """
        nchan, channels = self._channels(channel)

        logL = 0.
        for chan in channels:
            if self.nchannel_fitted > 1:
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
            unc_good = unc_fit[good]
            res_good = residuals[good]

            # set up GP with current parameters
            gp = self.setup_GP(chan)

            if self.gp_code_name == 'george':
                kin = self.kernel_inputs[chan][:, good].T
                gp.compute(kin, unc_good)
                logL += gp.lnlikelihood(res_good, quiet=True)
            elif self.gp_code_name == 'celerite':
                kin = self.kernel_inputs[chan][0][good]
                gp.compute(kin, yerr=unc_good)
                logL += gp.log_likelihood(res_good)
            elif self.gp_code_name == 'tinygp':
                if tinygp is None:
                    raise RuntimeError('tinygp is not available.')
                cond = gp.condition(res_good, diag=unc_good)
                logL += cond.log_probability
            else:
                raise ValueError(f'Unknown gp_code_name: {self.gp_code_name}')

        return logL
