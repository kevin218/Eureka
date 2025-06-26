import numpy as np

from .Model import Model
from ...lib.readEPF import Parameters
from ...lib.split_channels import split, get_trim


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
        self.name = 'step'

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

        # Update coefficients
        self.steps = np.zeros((self.nchannel_fitted, 10))
        self.steptimes = np.zeros((self.nchannel_fitted, 10))
        self.keys = list(self.parameters.dict.keys())
        self.keys = [key for key in self.keys if key.startswith('step')]
        self._parse_coeffs()

    @property
    def time(self):
        """A getter for the time."""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time."""
        if time_array is not None:
            self._time = np.ma.masked_invalid(time_array)
            # Convert to local time
            if self.multwhite:
                self.time_local = np.ma.zeros(self.time.shape)
                for chan in self.fitted_channels:
                    # Split the arrays that have lengths
                    # of the original time axis
                    trim1, trim2 = get_trim(self.nints, chan)
                    time = self.time[trim1:trim2]
                    self.time_local[trim1:trim2] = time-time.data[0]
            else:
                self.time_local = self.time - self.time.data[0]


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
        for key in self.keys:
            split_key = key.split('_')
            if len(split_key) == 1:
                chan = 0
            else:
                chan = int(split_key[1])
            if len(split_key[0]) < 9:
                # Get the step number and update self.steps
                self.steps[chan, int(split_key[0][4:])] = \
                    self.parameters.dict[key][0]
            else:
                # Get the steptime number and update self.steptimes
                self.steptimes[chan, int(split_key[0][8:])] = \
                    self.parameters.dict[key][0]

    def eval(self, channel=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            The value of the model at the times self.time.
        """
        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel, ]

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        lcfinal = np.array([])

        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            lcpiece = np.ma.ones(len(time))
            for s in np.where(self.steps[c] != 0)[0]:
                lcpiece[time >= self.steptimes[c, s]] += \
                    self.steps[c, s]
            lcfinal = np.append(lcfinal, lcpiece)
        return lcfinal
