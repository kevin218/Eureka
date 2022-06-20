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

        # Check for Parameters instance
        self.parameters = kwargs.get('parameters')

        # Generate parameters from kwargs if necessary
        if self.parameters is None:
            coeff_dict = kwargs.get('coeff_dict')
            params = {cN: coeff for cN, coeff in coeff_dict.items()
                      if cN.startswith('c') and cN[1:].isdigit()}
            self.parameters = Parameters(**params)

        # Set parameters for multi-channel fits
        self.longparamlist = kwargs.get('longparamlist')
        self.nchan = kwargs.get('nchan')
        self.paramtitles = kwargs.get('paramtitles')

        # Update coefficients
        self._parse_coeffs()

    def _parse_coeffs(self):
        """Convert dict of 'c#' coefficients into a list
        of coefficients in decreasing order, i.e. ['c2','c1','c0'].

        Returns
        -------
        np.ndarray
            The sequence of coefficient values
        """
        # Parse 'c#' keyword arguments as coefficients
        coeffs = np.zeros((self.nchan, 10))
        for k, v in self.parameters.dict.items():
            remvisnum = k.split('_')
            if k.lower().startswith('c') and k[1:].isdigit():
                coeffs[0, int(k[1:])] = v[0]
            elif (len(remvisnum) > 1 and self.nchan > 1 and
                  remvisnum[0].lower().startswith('c') and
                  remvisnum[0][1:].isdigit() and remvisnum[1].isdigit()):
                coeffs[int(remvisnum[1]), int(remvisnum[0][1:])] = v[0]

        # Trim zeros and reverse
        coeffs = coeffs[:, ~np.all(coeffs == 0, axis=0)]
        coeffs = np.flip(coeffs, axis=1)
        self.coeffs = coeffs

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

        # Convert to local time
        time_local = self.time - self.time.mean()

        # Create the polynomial from the coeffs
        lcfinal = np.array([])
        for c in np.arange(self.nchan):
            poly = np.poly1d(self.coeffs[c])
            lcpiece = np.polyval(poly, time_local)
            lcfinal = np.append(lcfinal, lcpiece)
        return lcfinal
