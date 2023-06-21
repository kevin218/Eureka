import numpy as np

from .Model import Model
from ...lib.readEPF import Parameters
from ...lib.split_channels import split


class SineScanModel(Model):
    """
        Sinusoid Scan Model

        Used to fit the spatial scan systematic effect in HST/WFC3 spatially
        scanned lightcurves, according to the model:

        M(t) = T(t)*Sys(t) + S(t_orb)

        where T(t) is the transit model, Sys(t) are the non-spatial scan 
        systematics models (e.g. polynomial trends, orbit-level ramps), 
        and S(t_orb) = sine_amp*cos(pi*t_orb / tau)

        sine_amp is the orbit-dependent, wavelength dependent amplitude, 
        tau is the exposure cadence, and t_orb is the local time in each HST
        orbit.
    
    
    """
    def __init__(self, **kwargs):
        """Initialize the spatial scan model for WFC data.

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
            try:
                params = {
                    "sine_amp": coeff_dict["sine_amp"],
                    "t_orb": coeff_dict["t_orb"],
                    "tau": coeff_dict["tau"],
                }
                self.parameters = Parameters(**params)
            except KeyError:
                # log error somehow?
                pass
            
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
            if self.multwhite:
                self.time_local = []
                for chan in self.fitted_channels:
                    # Split the arrays that have lengths
                    # of the original time axis
                    time = split([self.time, ], self.nints, chan)[0]
                    self.time_local.extend(time - time.mean())
                self.time_local = np.array(self.time_local)
            else:
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
        self.coeffs = np.zeros((self.nchannel_fitted, 10))
        for c in range(self.nchannel_fitted):
            if self.nchannel_fitted > 1:
                chan = self.fitted_channels[c]
            else:
                chan = 0

            for i in range(9, -1, -1):
                try:
                    if chan == 0:
                        self.coeffs[c, 9-i] = \
                            self.parameters.dict[f'c{i}'][0]
                    else:
                        self.coeffs[c, 9-i] = \
                            self.parameters.dict[f'c{i}_{chan}'][0]
                except KeyError:
                    pass

        # Trim zeros
        self.coeffs = self.coeffs[:, ~np.all(self.coeffs == 0, axis=0)]

        """Convert dict of coefficients into a list
        of coefficients in increasing order.
        """
        # Parse keyword arguments as coefficients
        self.coeffs = {}
        for i in range(self.nkernels):
            self.coeffs[self.kernel_types[i]] = []
        for k, v in self.parameters.dict.items():
            if k.startswith('A'):
                remvisnum = k.split('_')
                if len(remvisnum) > 1:
                    self.coeffs['A_%i' % int(remvisnum[1])] = v[0]
                elif self.nchan > 1:
                    self.coeffs['A_0'] = v[0]
                else:
                    self.coeffs['A'] = v[0]
            if k.lower().startswith('m'):
                remvisnum = k.split('_')
                if len(remvisnum) > 1 or self.nchan > 1:
                    no = int(remvisnum[0][1])-1
                    if no < 0:
                        raise AssertionError('Please start your metric '
                                             'enumeration with m1.')
                    self.coeffs[self.kernel_types[no]].append(v[0])
                else:
                    no = int(remvisnum[0][1])-1
                    self.coeffs[self.kernel_types[no]].append(v[0])
            if k.startswith('WN'):
                remvisnum = k.split('_')
                if len(remvisnum) > 1:
                    self.coeffs['WN_%i' % int(remvisnum[1])] = v[0]
                elif self.nchan > 1:
                    self.coeffs['WN_0'] = v[0]
                else:
                    self.coeffs['WN'] = v[0]
                if 'fixed' in v:
                    self.fit_white_noise = False
                else:
                    self.fit_white_noise = True

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
        lcfinal = np.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time_local
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]
            
            poly = np.poly1d(self.coeffs[chan])
            lcpiece = np.polyval(poly, time)
            lcfinal = np.append(lcfinal, lcpiece)
        return lcfinal
