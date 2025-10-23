import numpy as np

from .Model import Model
from ...lib.readEPF import Parameters
from ...lib.split_channels import split
from ...lib import astropytable


class CommonModeModel(Model):
    """Common Mode Model"""
    def __init__(self, **kwargs):
        """Initialize the common mode model.

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
        self.name = 'common_mode'

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

        # Check for Parameters instance
        self.parameters = kwargs.get('parameters')
        # Generate parameters from kwargs if necessary
        if self.parameters is None:
            coeff_dict = kwargs.get('coeff_dict')
            params = {cN: coeff for cN, coeff in coeff_dict.items()
                      if cN.startswith('cm') and cN[1:].isdigit()}
            self.parameters = Parameters(**params)

        # Update coefficients
        self.coeffs = np.zeros((self.nchannel_fitted, 2))
        self._parse_coeffs()

        # Read common-mode systematics file and store corresponding values.
        # This is usually the ECSV generated from a Stage 5 white LC fit.
        log = kwargs.get('log')
        meta = kwargs.get('meta')
        log.writelog(f'Reading {meta.common_mode_name} values from' +
                     ' common-mode systematics file:' +
                     f'{meta.common_mode_file}.', mute=(not meta.verbose))
        lc_table = astropytable.readtable(meta.common_mode_file)
        self.cm_flux = lc_table[meta.common_mode_name] - \
            np.mean(lc_table[meta.common_mode_name])

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
                self.time_local = np.ma.zeros(0)
                for chan in self.fitted_channels:
                    # Split the arrays that have lengths
                    # of the original time axis
                    time = split([self.time, ], self.nints, chan)[0]
                    self.time_local = np.ma.append(
                        self.time_local, time-np.ma.mean(time))
            else:
                self.time_local = self.time - np.ma.mean(self.time)

    def _parse_coeffs(self):
        """Convert dict of 'cm#' coefficients into a list
        of coefficients in increasing order, i.e. ['cm1','cm2'].

        Returns
        -------
        np.ndarray
            The sequence of coefficient values
        """
        # Parse 'cm#' keyword arguments as coefficients
        for c in range(self.nchannel_fitted):
            if self.nchannel_fitted > 1:
                chan = self.fitted_channels[c]
            else:
                chan = 0

            for i in range(2):
                try:
                    if chan == 0:
                        self.coeffs[c, i] = \
                            self.parameters.dict[f'cm{i+1}'][0]
                    else:
                        self.coeffs[c, i] = \
                            self.parameters.dict[f'cm{i+1}_ch{chan}'][0]
                except KeyError:
                    pass

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

        # Create the polynomial from the coeffs
        lcfinal = np.ma.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time_local
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            cm1, cm2 = self.coeffs[chan]
            lcpiece = 1 + cm1*self.cm_flux + cm2*self.cm_flux**2
            lcpiece = np.ma.masked_where(np.ma.getmaskarray(time), lcpiece)
            lcfinal = np.ma.append(lcfinal, lcpiece)
        return lcfinal
