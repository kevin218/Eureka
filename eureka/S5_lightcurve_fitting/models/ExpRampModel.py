import numpy as np

from .Model import Model
from ..parameters import Parameters

class ExpRampModel(Model):
    """Model for single or double exponential ramps"""
    def __init__(self, **kwargs):
        """Initialize the exponential ramp model
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
            params = {rN: coeff for rN, coeff in coeff_dict.items()
                      if rN.startswith('r') and rN[1:].isdigit()}
            self.parameters = Parameters(**params)

        # Update coefficients
        self._parse_coeffs()

    def _parse_coeffs(self):
        """Convert dict of 'r#' coefficients into a list
        of coefficients in increasing order, i.e. ['r0','r1','r2']

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            The sequence of coefficient values
        """
        # Parse 'c#' keyword arguments as coefficients
        self.coeffs = np.zeros(6)
        for k, v in self.parameters.dict.items():
            if k.lower().startswith('r') and k[1:].isdigit():
                self.coeffs[int(k[1:])] = v[0]

    def eval(self, **kwargs):
        """Evaluate the function with the given values"""
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Create the individual coeffs
        r0, r1, r2, r3, r4, r5 = self.coeffs
        # if len(self.coeffs) == 3:
        #     r0, r1, r2 = self.coeffs
        #     r3, r4, r5 = 0, 0, 0
        # elif len(self.coeffs) == 6:
        #     r0, r1, r2, r3, r4, r5 = self.coeffs
        # else:
        #     raise IndexError('Exponential ramp requires 3 or 6 parameters labelled r#.')

        # Convert to local time
        time_local = self.time - self.time[0]

        # Evaluate the polynomial
        return r0*np.exp(-r1*time_local + r2) + r3*np.exp(-r4*time_local + r5) + 1

    def update(self, newparams, names, **kwargs):
        """Update parameter values"""
        for ii,arg in enumerate(names):
            if hasattr(self.parameters,arg):
                val = getattr(self.parameters,arg).values[1:]
                val[0] = newparams[ii]
                setattr(self.parameters, arg, val)
        self._parse_coeffs()
        return