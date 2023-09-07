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
        and S(t_orb) = scan_amp*cos(pi*t_orb / tau)

        scan_amp is the orbit-dependent, wavelength dependent amplitude, 
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
                params = {ampN: amp for ampN, amp in coeff_dict.items()
                          if ampN.startswith('amp') and ampN[3:].isdigit()}
                self.parameters = Parameters(**params)
            except KeyError:
                # log error somehow?
                pass

        # Get orbit numbers
        self.orbits = kwargs.get('orbits')

        num_orbits = np.max(np.unique(self.orbits))

        # Update coefficients
        self.amps = np.zeros((self.nchannel_fitted, num_orbits))
        self._parse_amps()

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
                    self.time_local.extend(time - time[0])
                self.time_local = np.array(self.time_local)
            else:
                self.time_local = self.time - self.time[0]

    def _parse_amps(self):
        """Convert dict of scan_amp into a list.

        Returns
        -------
        np.ndarray
            The sequence of scan parameter values.
        """
        # Parse keyword arguments
        for c in range(self.nchannel_fitted):
            if self.nchannel_fitted > 1:
                chan = self.fitted_channels[c]
            else:
                chan = 0

            try:
                if chan == 0:
                    self.amps[c] = self.parameters.dict['scan_amp'][0]
                else:
                    self.amps[c] = \
                        self.parameters.dict[f'scan_amp_{chan}'][0]
            except KeyError:
                pass
    
    def eval(self, channel=None, **kwargs):
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
        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel, ]
        
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Get exposure cadence tau
        tau = np.mean(np.diff(self.time))  # self.time[1] - self.time[0]

        # Get orbit-level times
        orbits = kwargs.get('orbits')
        
        for i in rang


        # Create the scan from the coeffs
        lcfinal = np.array([])
        for c in np.arange(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time_local
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]
            
            h0, h1, h2, h3, h4, h5 = self.coeffs[c]
            # Batch time is relative to the start of each HST orbit
            # h5 is the orbital period of HST (~96 minutes)
            self.time_batch = self.time_local % h5
            lcpiece = (1+h0*np.exp(-h1*self.time_batch + h2)
                       + h3*self.time_batch + h4*self.time_batch**2)
            lcfinal = np.append(lcfinal, lcpiece)
        return lcfinal