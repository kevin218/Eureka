import numpy as np
from .Model import Model
from ...lib.split_channels import split


class Params():
    """
    Define damped oscillator parameters.
    """
    def __init__(self, model):      
        # Set parameters
        self.osc_amp = 0.
        self.osc_amp_decay = 0.
        self.osc_per = 1
        self.osc_per_decay = 0.
        self.osc_t0 = 0.
        self.osc_t1 = 0.
        for item in self.__dict__.keys():
            try:
                setattr(self, item, model.parameters.dict[item][0])
            except:
                pass


class DampedOscillatorModel(Model):
    """A sinusoidal phase curve model"""
    def __init__(self, **kwargs):
        """Initialize the phase curve model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from Model calss
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

    @property
    def time(self):
        """A getter for the time."""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time."""
        self._time = time_array

    # def update(self, newparams, **kwargs):
    #     """Update the model with new parameter values.

    #     Parameters
    #     ----------
    #     newparams : ndarray
    #         New parameter values.
    #     **kwargs : dict
    #         Additional parameters to pass to
    #         eureka.S5_lightcurve_fitting.models.Model.update().
    #     """
    #     super().update(newparams, **kwargs)
    #     if self.transit_model is not None:
    #         self.transit_model.update(newparams, **kwargs)
    #     if self.eclipse_model is not None:
    #         self.eclipse_model.update(newparams, **kwargs)

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
        lcfinal = np.array([])
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

            # Compute damped oscillator
            amp = params.osc_amp * np.exp(-params.osc_amp_decay * 
                                          (time - params.osc_t0) )
            per = params.osc_per * np.exp(-params.osc_per_decay * 
                                          (time - params.osc_t0) )
            osc = 1 + amp * np.sin(2 * np.pi * (time - params.osc_t1) / per)
            osc[time < params.osc_t0] = 1

            lcfinal = np.append(lcfinal, osc)

        return lcfinal
