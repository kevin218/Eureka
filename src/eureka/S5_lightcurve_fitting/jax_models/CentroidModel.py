import numpy as np
import jax.numpy as jnp

from . import JaxModel
from ...lib.split_channels import split, get_trim


class CentroidModel(JaxModel):
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
            eureka.S5_lightcurve_fitting.jax_models.JaxModel.__init__().
            Can pass in the parameters, longparamlist, nchan,
            paramtitles, axis, and centroid arguments here.
        """
        # Inherit from JaxModel class
        super().__init__(**kwargs)
        self.name = self.axis

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

        self.coeff_keys = [f'{self.axis}_ch{c}' if c > 0 else self.axis
                           for c in range(self.nchannel_fitted)]

    @property
    def centroid(self):
        """A getter for the centroid."""
        return self._centroid

    @centroid.setter
    def centroid(self, centroid_array):
        """A setter for the centroid."""
        if isinstance(centroid_array, np.ma.core.MaskedArray):
            # Convert to a numpy array with NaN masking
            centroid_array = centroid_array.filled(np.nan)

        self._centroid = centroid_array

        if self.centroid is not None:
            # Convert to local centroid
            if self.multwhite:
                self.centroid_local = np.zeros(self.centroid.shape)
                for chan in self.fitted_channels:
                    # Split the arrays that have lengths
                    # of the original time axis
                    trim1, trim2 = get_trim(self.nints, chan)
                    centroid = self.centroid[trim1:trim2]
                    self.centroid_local[trim1:trim2] = \
                        centroid-np.nanmean(centroid)
            else:
                self.centroid_local = self.centroid-np.nanmean(self.centroid)

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

        if eval:
            lib = np
            model = self.fit
        else:
            lib = jnp
            model = self.model

        lcfinal = lib.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            centroid = np.copy(self.centroid_local)
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                centroid = split([centroid, ], self.nints, chan)[0]

            coeff = getattr(model, self.coeff_keys[chan])
            lcpiece = 1 + centroid*coeff
            lcfinal = lib.concatenate([lcfinal, lcpiece])
        return lcfinal
