import numpy as np

from .Model import Model
from ..parameters import Parameters

class PolynomialModel(Model):
    """Polynomial Model"""
    def __init__(self, **kwargs):
        """Initialize the polynomial model
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

        # Update coefficients
        self._parse_coeffs()

    def _parse_coeffs(self, **kwargs):
        """Convert dict of 'c#' coefficients into a list
        of coefficients in decreasing order, i.e. ['c2','c1','c0']

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            The sequence of coefficient values
        """

        # Parse 'c#' keyword arguments as coefficients
        coeffs = np.zeros(10)
        for k, v in self.parameters.dict.items():
            if k.lower().startswith('c') and k[1:].isdigit():
                coeffs[int(k[1:])] = v[0]

        # Trim zeros and reverse
        self.coeffs = np.trim_zeros(coeffs)[::-1]

    def eval(self, **kwargs):
        """Evaluate the function with the given values"""
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Create the polynomial from the coeffs
        poly = np.poly1d(self.coeffs)

        # Convert to local time
        time_local = self.time - self.time.mean()

        # Evaluate the polynomial
        return np.polyval(poly, time_local)

    def update(self, newparams, names, **kwargs):
        """Update parameter values"""
        for ii,arg in enumerate(names):
            if hasattr(self.parameters,arg):
                val = getattr(self.parameters,arg).values[1:]
                val[0] = newparams[ii]
                setattr(self.parameters, arg, val)
        self._parse_coeffs()
        return
