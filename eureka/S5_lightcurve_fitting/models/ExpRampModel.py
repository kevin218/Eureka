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
        self._parse_coeffs()

    def _parse_coeffs(self):
        """Convert dict of 'r#' coefficients into a list
        of coefficients in increasing order, i.e. ['r0','r1','r2'].

        Returns
        -------
        np.ndarray
            The sequence of coefficient values.
        """
        # Parse 'r#' keyword arguments as coefficients
        self.coeffs = np.zeros((self.nchan, 6))
        for k, v in self.parameters.dict.items():
            remvisnum = k.split('_')
            if k.lower().startswith('r') and k[1:].isdigit():
                self.coeffs[0, int(k[1:])] = v[0]
            elif (len(remvisnum) > 1 and self.nchan > 1 and
                  remvisnum[0].lower().startswith('r') and
                  remvisnum[0][1:].isdigit() and
                  remvisnum[1].isdigit()):
                self.coeffs[int(remvisnum[1]),
                            int(remvisnum[0][1:])] = v[0]

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
        time_local = self.time - self.time[0]

        # Create the ramp from the coeffs
        lcfinal = np.array([])
        for c in np.arange(self.nchan):
            r0, r1, r2, r3, r4, r5 = self.coeffs[c]
            lcpiece = (1+r0*np.exp(-r1*time_local + r2)
                       + r3*np.exp(-r4*time_local + r5))
            lcfinal = np.append(lcfinal, lcpiece)
        return lcfinal
