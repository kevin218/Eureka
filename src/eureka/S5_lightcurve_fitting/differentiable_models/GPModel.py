import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt
import celerite2.pymc3 as celerite2

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from . import PyMC3Model
from ..likelihood import update_uncertainty
from ...lib.split_channels import split


class GPModel(PyMC3Model):
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
            eureka.S5_lightcurve_fitting.differentiable_models.PyMC3Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """  # noqa: E501
        # Inherit from PyMC3Model class
        super().__init__(kernel_types=kernel_types,
                         nkernels=len(kernel_types),
                         kernel_input_names=kernel_input_names,
                         kernel_inputs=None,
                         gp_code_name=gp_code_name, normalize=normalize,
                         fit_lc=np.ma.ones(self.flux.shape),
                         flux=lc.flux, unc=lc.unc, unc_fit=lc.unc_fit,
                         **kwargs)
        self.name = 'GP'

        # Define model type (physical, systematic, other)
        self.modeltype = 'GP'

        # Do some initial sanity checks and raise errors if needed
        if self.gp_code_name != 'celerite':
            raise AssertionError('Currently celerite2 is the only GP package '
                                 'that can be used with the exoplanet and '
                                 'nuts fitting methods.')
        elif self.nkernels > 1:
            raise AssertionError('Our celerite2 implementation cannot compute '
                                 'multi-dimensional GPs, please choose a '
                                 'different GP code.')
        elif self.kernel_types[0] != 'Matern32':
            raise AssertionError('Our celerite2 implementation currently only '
                                 'supports a Matern32 kernel.')

    def setup(self):
        """Setup a model for evaluation and fitting.
        """
        if eval:
            coeffs = self.fit_coeffs
            model = self.fit
        else:
            coeffs = self.coeffs
            model = self.model

        # Parse parameters as coefficients
        coeffs = np.zeros((self.nchannel_fitted, self.nkernels, 2)).tolist()

        self.gps = []
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
                        coeffs[c][k][i] = getattr(model, index)
                    except KeyError:
                        pass

    def update(self, newparams, **kwargs):
        # Inherit from Model class
        super().update(newparams, **kwargs)

        self.unc_fit = update_uncertainty(newparams, self.nints, self.unc,
                                          self.freenames)

    def setup(self):
        """Setup a model for evaluation and fitting.
        """
        self.gps = []
        for c in range(self.nchannel_fitted):
            gp = self.setup_GP(c=c, eval=False)
            self.gps.append(gp)

    def eval(self, fit_lc, channel=None, gp=None, **kwargs):
        """Compute GP with the given parameters

        Parameters
        ----------
        fit_lc : ndarray
            The rest of the current model evaluated.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        gp : celerite2.GP; optional
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
        self.fit_lc = fit_lc

        lcfinal = np.ma.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
                # get flux and uncertainties for current channel
                flux, unc_fit = split([self.flux, self.unc_fit],
                                      self.nints, chan)
                if channel is None:
                    fit_lc = split([self.fit_lc, ], self.nints, chan)[0]
                else:
                    # If only a specific channel is being evaluated, then only
                    # that channel's fitted model will be passed in
                    fit_lc = self.fit_lc
            else:
                chan = 0
                # get flux and uncertainties for current channel
                flux = self.flux
                fit_lc = self.fit_lc
                unc_fit = self.unc_fit
            residuals = np.ma.masked_invalid(flux-fit)
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
            if gp is None:
                gp = self.setup_GP(c=chan, eval=True)

            if self.gp_code_name == 'celerite':
                gp.compute(self.kernel_inputs[chan][0], yerr=unc_fit)
                mu = gp.predict(residuals).eval()

            # Re-insert and mask bad values
            mu_full = np.ma.zeros(len(time))
            mu_full[good] = mu
            mu_full = np.ma.masked_where(~good, mu_full)

            # Append this channel to the outputs
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
        # Parse parameters as coefficients
        self._parse_coeffs(eval=eval)

        if self.kernel_inputs is None:
            self.setup_inputs()

        # get the kernel which is the sum of the individual kernel functions
        kernel = self.get_kernel(self.kernel_types[0], 0, c, eval=eval)
        for k in range(1, self.nkernels):
            kernel += self.get_kernel(self.kernel_types[k], k, c, eval=eval)

        # Make the gp object
        gp = celerite2.GaussianProcess(kernel, mean=0, fit_mean=False)

        return gp

    def get_kernel(self, kernel_name, k, c=0, eval=True):
        """Get individual kernels.

        Parameters
        ----------
        kernel_name : str
            The name of the kernel to get. Currently unused since only
            celerite's Matern32 is supported.
        k : int
            The kernel number.
        c : int; optional
            The channel index, by default 0.
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.

        Returns
        -------
        kernel
            The requested kernel.
        """
        if eval:
            lib = np.ma
            coeffs = self.fit_coeffs
        else:
            lib = tt
            coeffs = self.coeffs

        # get metric and amplitude for the current kernel and channel
        amp = lib.exp(coeffs[c][k][0])
        metric = lib.exp(coeffs[c][k][1])

        # Currently only the Matern32 kernel is supported
        kernel = celerite2.terms.Matern32Term(sigma=1, rho=metric)

        # Setting the amplitude
        kernel *= celerite2.terms.RealTerm(a=amp, c=0)

        return kernel
