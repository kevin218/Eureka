import numpy as np

from .Model import Model
from ...lib.split_channels import split, get_trim


class ExpRampModel(Model):
    """Model for single or double exponential ramps"""
    def __init__(self, **kwargs):
        """Initialize the exponential ramp model.

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
        self.name = 'exp. ramp'

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

    @property
    def time(self):
        """A getter for the time."""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time."""
        if time_array is None:
            self._time = None
            self.time_local = None
            return

        self._time = np.ma.masked_invalid(time_array)
        # Convert to local time
        if self.multwhite:
            self.time_local = np.ma.zeros(self._time.shape)
            for chan in self.fitted_channels:
                # Split the arrays that have lengths
                # of the original time axis
                trim1, trim2 = get_trim(self.nints, chan)
                piece = self._time[trim1:trim2]
                # Use .data[0] to be robust to masks
                self.time_local[trim1:trim2] = piece - piece.data[0]
        else:
            # Use .data[0] to be robust to masks
            self.time_local = self._time - self._time.data[0]

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
        lcfinal : np.ma.MaskedArray
            The ramp model evaluated at self.time (or provided time).
        """
        nchan, channels = self._channels(channel)

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Create the ramp from the coeffs
        pieces = []
        for chan in channels:
            t = self.time_local
            if self.multwhite:
                # Split arrays that have lengths of the original time axis
                t = split([t], self.nints, chan)[0]

            # Get the coefficients for this channel
            r0 = self._get_param_value('r0', chan=chan)
            r1 = self._get_param_value('r1', chan=chan)
            r2 = self._get_param_value('r2', chan=chan)
            r3 = self._get_param_value('r3', chan=chan)
            lcpiece = 1. + r0*np.exp(-r1*t) + r2*np.exp(-r3*t)
            pieces.append(lcpiece)

        if len(pieces) == 1:
            return pieces[0]
        else:
            return np.ma.concatenate(pieces)
