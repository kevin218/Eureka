import numpy as np

from .Model import Model
from ...lib.split_channels import split, get_trim


class HSTRampModel(Model):
    """Model for HST orbit-long exponential plus quadratic ramps"""
    def __init__(self, **kwargs):
        """Initialize the HST ramp model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
        # Inherit from Model class
        super().__init__(**kwargs)
        self.name = 'hst ramp'

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
                trim1, trim2 = get_trim(self.nints, chan)
                piece = self._time[trim1:trim2]
                self.time_local[trim1:trim2] = piece - piece.data[0]
        else:
            self.time_local = self._time - self._time.data[0]

    def eval(self, channel=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : np.ma.MaskedArray
            The value of the model at self.time.
        """
        nchan, channels = self._channels(channel)

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        pieces = []
        for chan in channels:
            t = self.time_local
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                t = split([t, ], self.nints, chan)[0]

            # Get the coefficients for this channel
            h0 = self._get_param_value('h0', 0.0, chan=chan)
            h1 = self._get_param_value('h1', 0.0, chan=chan)
            h2 = self._get_param_value('h2', 0.0, chan=chan)
            h3 = self._get_param_value('h3', 0.0, chan=chan)
            h4 = self._get_param_value('h4', 0.0, chan=chan)
            h5 = self._get_param_value('h5', 0.0, chan=chan)

            # Batch time is relative to the start of each HST orbit
            # h4 = orbital period (~96 min), h5 = phase offset.
            t_batch = (t - h5) % h4
            lcpiece = (1. +
                       h0*np.exp(-h1*t_batch) +
                       h2*t_batch +
                       h3*t_batch**2)
            pieces.append(lcpiece)

        if len(pieces) == 1:
            return pieces[0]
        else:
            return np.ma.concatenate(pieces)
