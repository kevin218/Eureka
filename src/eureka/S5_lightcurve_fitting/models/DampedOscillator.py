import numpy as np

from .Model import Model
from ...lib.split_channels import split


class DampedOscillatorModel(Model):
    """A damped oscillator model"""
    def __init__(self, **kwargs):
        """Initialize the damped oscillator model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
        super().__init__(**kwargs)
        self.name = 'damped oscillator'

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

    def eval(self, channel=None, **kwargs):
        """Evaluate the model at the current (or provided) times.

        Parameters
        ----------
        channel : int; optional
            If not None, evaluate only this channel. Defaults to None.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : np.ma.MaskedArray
            The model value at self.time.
        """
        nchan, channels = self._channels(channel)

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        pieces = []
        for chan_id in channels:
            t = self.time
            if self.multwhite:
                t = split([t], self.nints, chan_id)[0]

            # Get the coefficients for this channel
            amp0 = self._get_param_value('osc_amp', chan=chan_id)
            amp_decay = self._get_param_value('osc_amp_decay', chan=chan_id)
            per0 = self._get_param_value('osc_per', chan=chan_id)
            per_decay = self._get_param_value('osc_per_decay', chan=chan_id)
            t0 = self._get_param_value('osc_t0', chan=chan_id)
            t1 = self._get_param_value('osc_t1', chan=chan_id)

            amp = amp0 * np.exp(-amp_decay * (t - t0))
            per = per0 * np.exp(-per_decay * (t - t0))
            lcpiece = 1. + amp * np.sin(2 * np.pi * (t - t1) / per)
            # Force pre-t0 region to unity.
            lcpiece[t < t0] = 1.
            pieces.append(lcpiece)

        if len(pieces) == 1:
            return pieces[0]
        else:
            return np.ma.concatenate(pieces)
