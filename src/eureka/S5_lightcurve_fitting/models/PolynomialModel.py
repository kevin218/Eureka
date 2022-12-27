import numpy as np

from .Model import Model
from ...lib.readEPF import Parameters


class PolynomialModel(Model):
    """Polynomial Model"""
    def __init__(self, **kwargs):
        """Initialize the polynomial model.

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

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

        # Generate parameters from kwargs if necessary
        if self.parameters is None:
            coeff_dict = kwargs.get('coeff_dict')
            params = {cN: coeff for cN, coeff in coeff_dict.items()
                      if cN.startswith('c') and cN[1:].isdigit()}
            self.parameters = Parameters(**params)

        # Update coefficients
        self._parse_coeffs()

    @property
    def time(self):
        """A getter for the time."""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time."""
        self._time = time_array
        if self.time is not None:
            # Convert to local time
            self.time_local = self.time - self.time.mean()

    def _parse_coeffs(self):
        """Convert dict of 'c#' coefficients into a list
        of coefficients in decreasing order, i.e. ['c2','c1','c0'].

        Returns
        -------
        np.ndarray
            The sequence of coefficient values
        """
        # Parse 'c#' keyword arguments as coefficients
        self.coeffs = np.zeros((self.nchan, 10))
        for j in range(self.nchan):
            for i in range(9, -1, -1):
                try:
                    if j == 0:
                        self.coeffs[j, 9-i] = \
                            self.parameters.dict[f'c{i}'][0]
                    else:
                        self.coeffs[j, 9-i] = \
                            self.parameters.dict[f'c{i}_j'][0]
                except KeyError:
                    pass

        # Trim zeros
        self.coeffs = self.coeffs[:, ~np.all(self.coeffs == 0, axis=0)]

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
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Create the polynomial from the coeffs
        lcfinal = np.array([])
        for c in np.arange(self.nchan):
            poly = np.poly1d(self.coeffs[c])
            lcpiece = np.polyval(poly, self.time_local)
            lcfinal = np.append(lcfinal, lcpiece)
        return lcfinal
