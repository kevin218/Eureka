import numpy as np

from .Model import Model
from ...lib.readEPF import Parameters


class ExpRampModel(Model):
    """Model for single or double exponential ramps"""
    def __init__(self, **kwargs):
        """Initialize the exponential ramp model.

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
            params = {rN: coeff for rN, coeff in coeff_dict.items()
                      if rN.startswith('r') and rN[1:].isdigit()}
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
            if self.multwhite:
                self.time_local = []
                for c in np.arange(self.nchan):
                    trim1 = np.nansum(self.mwhites_nexp[:c])
                    trim2 = trim1 + self.mwhites_nexp[c]
                    time = self.time[trim1:trim2]
                    self.time_local.extend(time - time[0])
                self.time_local = np.array(self.time_local)
            else:
                self.time_local = self.time - self.time[0]

    def _parse_coeffs(self):
        """Convert dict of 'r#' coefficients into a list
        of coefficients in increasing order, i.e. ['r0','r1','r2'].

        Returns
        -------
        np.ndarray
            The sequence of coefficient values.
        """
        # Parse 'r#' keyword arguments as coefficients
        for j in range(self.nchan):
            for i in range(6):
                try:
                    if j == 0:
                        self.coeffs[j, i] = self.parameters.dict[f'r{i}'][0]
                    else:
                        self.coeffs[j, i] = self.parameters.dict[f'r{i}_j'][0]
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
            if self.multwhite:
                trim1 = np.nansum(self.mwhites_nexp[:c])
                trim2 = trim1 + self.mwhites_nexp[c]
                time = self.time_local[trim1:trim2]
            else:
                time = self.time_local
            r0, r1, r2, r3, r4, r5 = self.coeffs[c]
            lcpiece = (1+r0*np.exp(-r1*time + r2)
                       + r3*np.exp(-r4*time + r5))
            lcfinal = np.append(lcfinal, lcpiece)
        return lcfinal
