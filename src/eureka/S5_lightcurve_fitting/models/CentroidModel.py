import numpy as np

from .Model import Model
from ...lib.split_channels import split, get_trim


class CentroidModel(Model):
    """Centroid Model

    Linear decorrelation against x position (axis='xpos'), y position
    (axis='ypos'), x width (axis='xwidth'), or y width (axis='ywidth').
    """
    def __init__(self, axis, **kwargs):
        """Initialize the centroid model.

        Parameters
        ----------
        axis : str
            The axis to use for the centroid model. Must be one of
            'xpos', 'ypos', 'xwidth', or 'ywidth'.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan,
            paramtitles, axis, and centroid arguments here.
        """
        # Validate axis kwarg early
        if axis is None or axis not in ['xpos', 'ypos', 'xwidth', 'ywidth']:
            raise ValueError("CentroidModel requires an 'axis' argument set "
                             "to 'xpos', 'ypos', 'xwidth', or 'ywidth'.")
        super().__init__(axis=axis, **kwargs)
        self.name = self.axis

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

    @property
    def centroid(self):
        """A getter for the centroid."""
        return self._centroid

    @centroid.setter
    def centroid(self, centroid_array):
        """A setter for the centroid."""
        if centroid_array is None:
            self._centroid = None
            self.centroid_local = None
            return

        self._centroid = np.ma.masked_invalid(centroid_array)
        # Convert to local (mean-centered) centroid
        if self.multwhite:
            self.centroid_local = np.ma.zeros(self._centroid.shape)
            for chan in self.fitted_channels:
                # Split the arrays that have lengths
                # of the original time axis
                trim1, trim2 = get_trim(self.nints, chan)
                piece = self._centroid[trim1:trim2]
                # Use .mean() to be robust to masks
                self.centroid_local[trim1:trim2] = piece - piece.mean()
        else:
            # Use .mean() to be robust to masks
            self.centroid_local = self._centroid - self._centroid.mean()

    def eval(self, channel=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        **kwargs : dict
            Must pass in the centroid array here if not already set.

        Returns
        -------
        np.ma.MaskedArray
            The value of the model at self.centroid.
        """
        nchan, channels = self._channels(channel)

        # Get the centroids
        if self.centroid is None:
            self.centroid = kwargs.get('centroid')

        # Create the centroid model for each wavelength
        pieces = []
        for chan in channels:
            centroid = self.centroid_local
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                centroid = split([centroid, ], self.nints, chan)[0]

            # Get the coefficient for this channel
            coeff = self._get_param_value(self.axis, 0.0, chan=chan)
            lcpiece = 1. + centroid*coeff
            pieces.append(lcpiece)

        if len(pieces) == 1:
            return pieces[0]
        else:
            return np.ma.concatenate(pieces)
