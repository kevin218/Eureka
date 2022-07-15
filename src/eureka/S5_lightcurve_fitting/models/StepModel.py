import numpy as np

from .Model import Model
from ...lib.readEPF import Parameters


class StepModel(Model):
    """Model for step-functions in time"""
    def __init__(self, **kwargs):
        """Initialize the step-function model.

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
            steps_dict = kwargs.get('steps_dict')
            steptimes_dict = kwargs.get('steptimes_dict')
            params = {key: coeff for key, coeff in steps_dict.items()
                      if key.startswith('step') and key[4:].isdigit()}
            params.update({key: coeff for key, coeff in steptimes_dict.items()
                           if (key.startswith('steptime') and 
                               key[9:].isdigit())})
            self.parameters = Parameters(**params)

        # Set parameters for multi-channel fits
        self.longparamlist = kwargs.get('longparamlist')
        self.nchan = kwargs.get('nchan')
        self.paramtitles = kwargs.get('paramtitles')

        # Update coefficients
        self._parse_coeffs()

    def _parse_coeffs(self):
        """Convert dictionary of parameters into an array.
        
        Converts dict of 'step#' coefficients into an array
        of coefficients in increasing order, i.e. ['step0', 'step1'].
        Also converts dict of 'steptime#' coefficients into an array
        of times in increasing order, i.e. ['steptime0', 'steptime1'].
        
        Returns
        -------
        np.ndarray
            The sequence of coefficient values.

        Notes
        -----
        History:

        - 2022 July 14, Taylor J Bell
            Initial version.
        """
        self.steps = np.zeros((self.nchan, 10))
        self.steptimes = np.zeros((self.nchan, 10))
        for k, v in self.parameters.dict.items():
            remvisnum = k.split('_')

            # Setup keys to look for
            key1 = 'step'
            key1_len = len(key1)
            key2 = 'steptime'
            key2_len = len(key2)

            # Parse 'step#' keyword arguments
            if (len(k) > key1_len and k.lower().startswith(key1) 
                    and k[key1_len:].isdigit()):
                self.steps[0, int(k[key1_len:])] = v[0]
            elif (len(remvisnum) > 1 and self.nchan > 1 and
                  remvisnum[0].lower().startswith(key1) and
                  remvisnum[0][key1_len:].isdigit() and
                  remvisnum[1].isdigit()):
                self.steps[int(remvisnum[1]),
                           int(remvisnum[0][key1_len:])] = v[0]
            # Parse 'steptime#' keyword arguments
            elif (len(k) > key1_len and k.lower().startswith(key2)
                  and k[key2_len:].isdigit()):
                self.steptimes[0, int(k[key2_len:])] = v[0]
            elif (len(remvisnum) > 1 and self.nchan > 1 and
                  remvisnum[0].lower().startswith(key2) and
                  remvisnum[0][key2_len:].isdigit() and
                  remvisnum[1].isdigit()):
                self.steptimes[int(remvisnum[1]),
                               int(remvisnum[0][key2_len:])] = v[0]

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
        lcfinal = np.ones((self.nchan, len(self.time)))
        for c in range(self.nchan):
            for s in np.where(self.steps[c] != 0)[0]:
                lcfinal[c, self.time >= self.steptimes[c, s]] += \
                    self.steps[c, s]
        return lcfinal.flatten()
