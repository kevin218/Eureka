import numpy as np
import jax.numpy as jnp

from . import JaxModel
from ...lib.split_channels import split


class Params():
    """
    Define asymetric Lorentzian parameters.
    """
    def __init__(self, model):
        """
        Set attributes to Params object.

        Parameters
        ----------
        model : object
            The model.eval object that contains a dictionary of parameter names
            and their current values.
        """
        # Set parameters
        self.lor_amp = None
        self.lor_amp_lhs = None
        self.lor_amp_rhs = None
        self.lor_hwhm = None
        self.lor_hwhm_lhs = None
        self.lor_hwhm_rhs = None
        self.lor_t0 = None
        self.lor_power = 2
        for item in self.__dict__.keys():
            try:
                setattr(self, item, getattr(model, item))
            except AttributeError:
                pass


class LorentzianModel(JaxModel):
    """An asymetric lorentzian model"""
    def __init__(self, **kwargs):
        """Initialize the asymetric lorentzian model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.jax_models.JaxModel.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from Model class
        super().__init__(**kwargs)
        self.name = 'lorentzian'

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

    def eval(self, eval=True, channel=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            The value of the model at the times self.time.
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

        if eval:
            lib = np
            model = self.fit
        else:
            lib = jnp
            model = self.model

        # Set all parameters
        lcfinal = lib.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            # Temporarily emove poorly handled NaN values
            full_time = np.copy(time)
            good = np.isfinite(time)
            time = time[good]

            # Initialize model
            params = Params(model)

            if (params.lor_hwhm is None) and (params.lor_amp is None):
                # Compute asymmetric Lorentzian with baseline offset
                baseline = 1 + params.lor_amp_lhs - params.lor_amp_rhs
                ut = lib.where(
                    time <= params.lor_t0,
                    (params.lor_t0 - time) / params.lor_hwhm_lhs,
                    (time - params.lor_t0) / params.lor_hwhm_rhs
                )
                lorentzian = lib.where(
                    time <= params.lor_t0,
                    1 + params.lor_amp_lhs/(1 + ut**params.lor_power),
                    baseline+params.lor_amp_rhs/(1+ut**params.lor_power)
                )
            elif (params.lor_hwhm is None) and \
                    (params.lor_amp_lhs is None) and \
                    (params.lor_amp_rhs is None):
                # Compute asymmetric Lorentzian with constant baseline
                ut = lib.where(
                    time <= params.lor_t0,
                    (params.lor_t0 - time) / params.lor_hwhm_lhs,
                    (time - params.lor_t0) / params.lor_hwhm_rhs
                )
                lorentzian = 1 + params.lor_amp_lhs/(1 + ut**params.lor_power)
            elif (params.lor_hwhm_lhs is None) and \
                    (params.lor_hwhm_rhs is None) and \
                    (params.lor_amp_lhs is None) and \
                    (params.lor_amp_rhs is None):
                # Compute symmetric Lorentzian
                ut = 2*(time-params.lor_t0)/params.lor_hwhm
                lorentzian = 1 + params.lor_amp/(1 + ut**params.lor_power)
            else:
                # Unresolvable situation
                raise Exception("Cannot determine the type of Lorentzian model"
                                " to fit.  Use one of the following options: "
                                "1. lor_amp, lor_hwhm; "
                                "2. lor_amp, lor_hwhm_lhs, lor_hwhm_rhs; "
                                "3. lor_amp_lhs, lor_amp_rhs, lor_hwhm_lhs, "
                                "lor_hwhm_rhs.")

            # Re-insert and mask bad values
            lorentzian_full = np.nan*lib.ones(len(full_time))
            if eval:
                lorentzian_full[good] = lorentzian
            else:
                # Jax arrays are immutable, so we need to update the array
                # in a different way
                lorentzian_full = lorentzian_full.at[good].set(lorentzian)

            lcfinal = lib.concatenate([lcfinal, lorentzian_full])

        return lcfinal
