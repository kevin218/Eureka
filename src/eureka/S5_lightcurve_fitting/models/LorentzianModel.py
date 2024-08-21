import numpy as np
from .Model import Model
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
                setattr(self, item, model.parameters.dict[item][0])
            except KeyError:
                pass


class LorentzianModel(Model):
    """An asymetric lorentzian model"""
    def __init__(self, **kwargs):
        """Initialize the asymetric lorentzian model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from Model class
        super().__init__(**kwargs)
        self.name = 'lorentzian'

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

    def eval(self, channel=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
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

        # Set all parameters
        lcfinal = np.ma.masked_array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            # Initialize model
            params = Params(self)
            p = params.lor_power
            t0 = params.lor_t0

            if (params.lor_hwhm is None) and (params.lor_amp is None):
                # Determine left and right halves of asymmetric Lorentzian
                lhs = np.ma.where(time <= t0)
                rhs = np.ma.where(time > t0)
                # Compute asymmetric Lorentzian with baseline offset
                ut = np.ma.zeros(time.shape)
                lorentzian = np.ma.zeros(time.shape)
                ut[lhs] = (t0-time[lhs])/params.lor_hwhm_lhs
                ut[rhs] = (time[rhs]-t0)/params.lor_hwhm_rhs
                lorentzian[lhs] = 1 + params.lor_amp_lhs/(1 + ut[lhs]**p)
                baseline = 1 + params.lor_amp_lhs - params.lor_amp_rhs
                lorentzian[rhs] = baseline+params.lor_amp_rhs/(1+ut[rhs]**p)
            elif (params.lor_hwhm is None) and \
                    (params.lor_amp_lhs is None) and \
                    (params.lor_amp_rhs is None):
                # Determine left and right halves of asymmetric Lorentzian
                lhs = np.ma.where(time <= t0)
                rhs = np.ma.where(time > t0)
                # Compute asymmetric Lorentzian with constant baseline
                ut = np.ma.zeros(time.shape)
                ut[lhs] = (time[lhs]-t0)/params.lor_hwhm_lhs
                ut[rhs] = (time[rhs]-t0)/params.lor_hwhm_rhs
                lorentzian = 1 + params.lor_amp_lhs/(1 + ut**p)
            elif (params.lor_hwhm_lhs is None) and \
                    (params.lor_hwhm_rhs is None) and \
                    (params.lor_amp_lhs is None) and \
                    (params.lor_amp_rhs is None):
                # Compute symmetric Lorentzian
                ut = 2*(time-t0)/params.lor_hwhm
                lorentzian = 1 + params.lor_amp/(1 + ut**p)
            else:
                # Unresolvable situation
                raise Exception("Cannot determine the type of Lorentzian model"
                                " to fit.  Use one of the following options: "
                                "1. lor_amp, lor_hwhm; "
                                "2. lor_amp, lor_hwhm_lhs, lor_hwhm_rhs; "
                                "3. lor_amp_lhs, lor_amp_rhs, lor_hwhm_lhs, "
                                "lor_hwhm_rhs.")

            lcfinal = np.ma.append(lcfinal, lorentzian)

        return lcfinal
