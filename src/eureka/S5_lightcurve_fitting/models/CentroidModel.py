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
        if self.centroid is not None:
            # Convert to local centroid
            self.centroid_local = self.centroid - self.centroid.mean()

        if self.nchan == 1:
            self.coeff_keys = [self.axis]
        else:
            self.coeff_keys = [f'{self.axis}_{i}' for i in range(self.nchan)]

        # Update coefficients
        self._parse_coeffs()

    def eval(self, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            The value of the model at the times self.time.
        """
        # Get the centroids
        if self.centroid is None:
            self.centroid = kwargs.get('centroid')
            # Convert to local centroid
            self.centroid_local = self.centroid - self.centroid.mean()

        # Create the centroid model for each wavelength
        lcfinal = np.array([])
        for c in np.arange(self.nchan):
            coeff = getattr(self.parameters, self.coeff_keys[c]).value
            lcpiece = 1 + self.centroid_local*coeff
            lcfinal = np.append(lcfinal, lcpiece)
        return lcfinal
