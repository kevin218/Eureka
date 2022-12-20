import numpy as np

from .Model import Model
from ...lib.readEPF import Parameters


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
        self.mwhites_nexp = kwargs.get('mwhites_nexp')

        # Inherit from Model class
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

        # Check for Parameters instance
        self.parameters = kwargs.get('parameters')

        # Generate parameters from kwargs if necessary
        if self.parameters is None:
            self.parameters = Parameters(**kwargs)

        # Set parameters for multi-channel fits
        self.longparamlist = kwargs.get('longparamlist')
        self.nchan = kwargs.get('nchan')
        self.paramtitles = kwargs.get('paramtitles')

        # Figure out if using xpos, ypos, xwidth, ywidth
        self.axis = kwargs.get('axis')
        self.centroid = kwargs.get('centroid')

        if self.nchan == 1:
            self.coeff_keys = [self.axis]
        else:
            self.coeff_keys = [f'{self.axis}_{i}' for i in range(self.nchan)]

    @property
    def centroid(self):
        """A getter for the centroid."""
        return self._centroid

    @centroid.setter
    def centroid(self, centroid_array):
        """A setter for the time."""
        self._centroid = np.ma.masked_invalid(centroid_array)
        if self.centroid is not None:
            # Convert to local centroid
            if self.multwhite:
                self.time_local = []
                for c in np.arange(self.nchan):
                    trim1 = np.nansum(self.mwhites_nexp[:c])
                    trim2 = trim1 + self.mwhites_nexp[c]
                    centroid = self.centroid[trim1:trim2]
                    self.centroid_local.append(centroid - centroid.mean())
            else:
                self.centroid_local = self.centroid - self.centroid.mean()

    def eval(self, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        **kwargs : dict
            Must pass in the centroid array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            The value of the model at the centroid self.centroid.
        """
        # Get the centroids
        if self.centroid is None:
            self.centroid = kwargs.get('centroid')

        # Create the centroid model for each wavelength
        lcfinal = np.array([])
        for c in np.arange(self.nchan):
            if self.multwhite:
                trim1 = np.nansum(self.mwhites_nexp[:c])
                trim2 = trim1 + self.mwhites_nexp[c]
                centroid = self.centroid_local[trim1:trim2]
            else:
                centroid = self.centroid_local
            coeff = getattr(self.parameters, self.coeff_keys[c]).value
            lcpiece = 1 + centroid*coeff
            lcfinal = np.append(lcfinal, lcpiece)
        return lcfinal
