import numpy as np

from .Model import Model
from ...lib.split_channels import split


class CentroidModel(Model):
    """Centroid Model

    This can be used to do a linear decorrelation against the x position
    (axis='xpos'), y position (axis='ypos'), x width (axis='xwidth'),
    or y width (axis='ywidth').

    """
    def __init__(self, **kwargs):
        """Initialize the centroid model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan,
            paramtitles, axis, and centroid arguments here.
        """
        # Needed before setting centroid
        self.multwhite = kwargs.get('multwhite')
        self.nints = kwargs.get('nints')

        # Inherit from Model class
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

        # Figure out if using xpos, ypos, xwidth, ywidth
        self.axis = kwargs.get('axis')
        self.centroid = kwargs.get('centroid')

        self.coeff_keys = [f'{self.axis}_{c}' if c > 0 else self.axis
                           for c in range(self.nchannel_fitted)]

    @property
    def centroid(self):
        """A getter for the centroid."""
        return self._centroid

    @centroid.setter
    def centroid(self, centroid_array):
        """A setter for the centroid."""
        self._centroid = np.ma.masked_invalid(centroid_array)
        if self.centroid is not None:
            # Convert to local centroid
            if self.multwhite:
                self.centroid_local = np.ma.zeros(0)
                for chan in self.fitted_channels:
                    # Split the arrays that have lengths
                    # of the original time axis
                    centroid = split([self.centroid, ], self.nints, chan)[0]
                    self.centroid_local = np.ma.append(
                        self.centroid_local, centroid-np.ma.mean(centroid))
            else:
                self.centroid_local = self.centroid - np.ma.mean(self.centroid)

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
        lcfinal : ndarray
            The value of the model at the centroid self.centroid.
        """
        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel, ]

        # Get the centroids
        if self.centroid is None:
            self.centroid = kwargs.get('centroid')

        # Create the centroid model for each wavelength
        lcfinal = np.ma.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            centroid = self.centroid_local
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                centroid = split([centroid, ], self.nints, chan)[0]

            coeff = getattr(self.parameters, self.coeff_keys[chan]).value
            lcpiece = 1 + centroid*coeff
            lcfinal = np.ma.append(lcfinal, lcpiece)
        return lcfinal
