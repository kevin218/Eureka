import numpy as np
import jax
import jax.numpy as jnp

from . import JaxModel
from ...lib.split_channels import split

jax.config.update("jax_enable_x64", True)


class Params():
    """
    Define damped oscillator parameters.
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
        self.osc_amp = None
        self.osc_amp_decay = 0.
        self.osc_per = None
        self.osc_per_decay = 0.
        self.osc_t0 = 0.
        self.osc_t1 = None
        for item in self.__dict__.keys():
            try:
                setattr(self, item, getattr(model, item))
            except AttributeError:
                pass


class DampedOscillatorModel(JaxModel):
    """A damped oscillator model"""
    def __init__(self, **kwargs):
        """Initialize the damped oscillator model.

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
        self.name = 'damped oscillator'

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

            # Initialize model
            params = Params(model)

            # Compute damped oscillator
            amp = params.osc_amp * lib.exp(-params.osc_amp_decay *
                                           (time - params.osc_t0))
            per = params.osc_per * lib.exp(-params.osc_per_decay *
                                           (time - params.osc_t0))
            osc = 1 + amp * lib.sin(2 * np.pi * (time - params.osc_t1) / per)

            # Jax arrays are immutable, so if not eval we need to update the
            # array in a different way
            if eval:
                osc[time < params.osc_t0] = 1
            else:
                osc = osc.at[time < params.osc_t0].set(1)

            lcfinal = lib.concatenate([lcfinal, osc])

        return lcfinal
