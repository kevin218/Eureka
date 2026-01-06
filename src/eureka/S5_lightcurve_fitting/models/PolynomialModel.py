import numpy as np

from .Model import Model
from ...lib.split_channels import split, get_trim


class PolynomialModel(Model):
    """Polynomial Model"""
    def __init__(self, **kwargs):
        """Initialize the polynomial model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
        # Inherit from Model class
        super().__init__(**kwargs)
        self.name = 'polynomial'

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
                self.time_local[trim1:trim2] = piece - piece.mean()
        else:
            self.time_local = self._time - self._time.mean()

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
                t = split([t], self.nints, chan)[0]

            # Get the coefficients for this channel
            vals = np.array([
                self._get_param_value(f'c{i}', 0.0, chan=chan)
                for i in range(10)
            ])
            # Trim high-degree trailing zeros.
            nz = np.nonzero(vals)[0]
            if nz.size == 0:
                coeffs_desc = np.array([0.0], dtype=float)
            else:
                max_idx = int(nz[-1])
                trimmed = vals[:max_idx + 1]
                # Descending order for np.polyval: [cN, ..., c0]
                coeffs_desc = trimmed[::-1]

            lcpiece = np.polyval(coeffs_desc, t)
            lcpiece = np.ma.masked_where(np.ma.getmaskarray(t), lcpiece)
            pieces.append(lcpiece)

        if len(pieces) == 1:
            return pieces[0]
        else:
            return np.ma.concatenate(pieces)
