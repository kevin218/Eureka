import numpy as np

from .Model import Model
from ...lib.readEPF import Parameters


class HSTRampModel(Model):
    """Model for HST orbit-long exponential plus quadratic ramps"""
    def __init__(self, **kwargs):
        """Initialize the HST ramp model.

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
            params = {hN: coeff for hN, coeff in coeff_dict.items()
                      if hN.startswith('h') and hN[1:].isdigit()}
            self.parameters = Parameters(**params)

        # Set parameters for multi-channel fits
        self.longparamlist = kwargs.get('longparamlist')
        self.nchan = kwargs.get('nchan')
        self.paramtitles = kwargs.get('paramtitles')

        # Update coefficients
        self.coeffs = np.zeros((self.nchan, 6))
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
            self.time_local = self.time - self.time[0]

    def _parse_coeffs(self):
        """Convert dict of 'h#' coefficients into a list
        of coefficients in increasing order, i.e. ['h0','h1','h2'].

        Returns
        -------
        np.ndarray
            The sequence of coefficient values.
        """
        # Parse 'h#' keyword arguments as coefficients
        for j in range(self.nchan):
            for i in range(6):
                try:
                    if j == 0:
                        self.coeffs[j, i] = self.parameters.dict[f'h{i}'][0]
                    else:
                        self.coeffs[j, i] = self.parameters.dict[f'h{i}_j'][0]
                except KeyError:
                    pass

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

        # Create the ramp from the coeffs
        lcfinal = np.array([])
        for c in np.arange(self.nchan):
            h0, h1, h2, h3, h4, h5 = self.coeffs[c]
            # Batch time is relative to the start of each HST orbit
            # h5 is the orbital period of HST (~96 minutes)
            self.time_batch = self.time_local % h5
            lcpiece = (1+h0*np.exp(-h1*self.time_batch + h2)
                       + h3*self.time_batch + h4*self.time_batch**2)
            lcfinal = np.append(lcfinal, lcpiece)
        return lcfinal
