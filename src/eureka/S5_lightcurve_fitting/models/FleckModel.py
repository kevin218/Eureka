import numpy as np
import astropy.units as unit
import inspect

import fleck
import batman

from .Model import Model
from ..limb_darkening_fit import ld_profile
from ...lib.split_channels import split


class FleckTransitModel(Model):
    """Transit Model with Star Spots"""
    def __init__(self, **kwargs):
        """Initialize the fleck model

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
        self.modeltype = 'physical'

        log = kwargs.get('log')

        # Store the ld_profile
        self.ld_from_S4 = kwargs.get('ld_from_S4')
        ld_func = ld_profile(self.parameters.limb_dark.value, 
                             use_gen_ld=self.ld_from_S4)
        len_params = len(inspect.signature(ld_func).parameters)
        self.coeffs = ['u{}'.format(n) for n in range(len_params)[1:]]

        self.ld_from_file = kwargs.get('ld_from_file')
        
        # Replace u parameters with generated limb-darkening values
        if self.ld_from_S4 or self.ld_from_file:
            log.writelog("Using the following limb-darkening values:")
            self.ld_array = kwargs.get('ld_coeffs')
            for c in range(self.nchannel_fitted):
                chan = self.fitted_channels[c]
                if self.ld_from_S4:
                    ld_array = self.ld_array[len_params-2]
                else:
                    ld_array = self.ld_array
                for u in self.coeffs:
                    index = np.where(np.array(self.paramtitles) == u)[0]
                    if len(index) != 0:
                        item = self.longparamlist[c][index[0]]
                        param = int(item.split('_')[0][-1])
                        ld_val = ld_array[chan][param-1]
                        log.writelog(f"{item}, {ld_val}")
                        # Use the file value as the starting guess
                        self.parameters.dict[item][0] = ld_val
                        # In a normal prior, center at the file value
                        if (self.parameters.dict[item][-1] == 'N' and
                                self.recenter_ld_prior):
                            self.parameters.dict[item][-3] = ld_val
                        # Update the non-dictionary form as well
                        setattr(self.parameters, item,
                                self.parameters.dict[item])
                                
        # Get relevant spot parameters
        self.spotrad = np.zeros((self.nchannel_fitted, 10))
        self.spotlat = np.zeros((self.nchannel_fitted, 10))
        self.spotlon = np.zeros((self.nchannel_fitted, 10))
        self.keys = list(self.parameters.dict.keys())
        self.keys = [key for key in self.keys if key.startswith('spot')]

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

        # Initialize model
        bm_params = batman.TransitParams()

        # Set all parameters
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

            # Set all batman parameters
            for index, item in enumerate(self.longparamlist[chan]):
                setattr(bm_params, self.paramtitles[index],
                        self.parameters.dict[item][0])
                        
            # Set spot/fleck parameters
            nspots = 0
            # set a star rotation for the star object
            # not actually used in fast mode. 
            # overwritten if user supplies and runs slow mode
            star_rotation = 100 
            fleck_fast = True
            for key in self.keys:
                split_key = key.split('_')
                if len(split_key) == 1:
                    chan = 0
                else:
                    chan = int(split_key[1])
                if 'rad' in split_key[0]:
                    # Get the spot number and update self.spotrad
                    self.spotrad[chan, int(split_key[0][7:])] = \
                        np.array([self.parameters.dict[key][0]])
                    nspots += 1
                elif 'lat' in split_key[0]:
                    # Get the spot lat and update self.spotlat
                    self.spotlat[chan, int(split_key[0][7:])] = \
                        np.array([self.parameters.dict[key][0]])
                elif 'lon' in split_key[0]:
                    # Get the spot lon and update self.spotlon
                    self.spotlon[chan, int(split_key[0][7:])] = \
                        np.array([self.parameters.dict[key][0]])
                elif 'con' in split_key[0]:
                    # Get the spot constrast and assign
                    spot_contrast = self.parameters.dict[key][0]
                elif 'rot' in split_key[0]:
                    # Get the stellar rotation and assign
                    star_rotation = self.parameters.dict[key][0]
                    fleck_fast = False
                elif 'stari' in split_key[0]:
                    # Get the stellar inclination and assign
                    star_inc = self.parameters.dict[key][0]
                elif 'npts' in split_key[0]:
                    # it's the number of points to evaluate
                    npoints = self.parameters.dict[key][0]
            
            # Set limb darkening parameters
            uarray = []
            for u in self.coeffs:
                index = np.where(np.array(self.paramtitles) == u)[0]
                if len(index) != 0:
                    item = self.longparamlist[chan][index[0]]
                    uarray.append(self.parameters.dict[item][0])
            bm_params.u = uarray

            # Enforce physicality to avoid crashes from batman by returning
            # something that should be a horrible fit
            if not ((0 < bm_params.per) and (0 < bm_params.inc < 90) and
                    (1 < bm_params.a) and (0 <= bm_params.ecc < 1) and
                    (0 <= bm_params.w <= 360)):
                # Returning nans or infs breaks the fits, so this was the
                # best I could think of
                lcfinal = np.append(lcfinal, 1e12*np.ones_like(time))
                continue

            # Use batman ld_profile name
            if self.parameters.limb_dark.value == '4-parameter':
                bm_params.limb_dark = 'nonlinear'
            elif self.parameters.limb_dark.value == 'kipping2013':
                # Enforce physicality to avoid crashes from batman by
                # returning something that should be a horrible fit
                if bm_params.u[0] <= 0:
                    # Returning nans or infs breaks the fits, so this was
                    # the best I could think of
                    lcfinal = np.append(lcfinal, 1e12*np.ones_like(time))
                    continue
                bm_params.limb_dark = 'quadratic'
                u1 = 2*np.sqrt(bm_params.u[0])*bm_params.u[1]
                u2 = np.sqrt(bm_params.u[0])*(1-2*bm_params.u[1])
                bm_params.u = np.array([u1, u2])

            # Make the star object
            star = fleck.Star(spot_contrast=spot_contrast, 
                              u_ld=bm_params.u, 
                              rotation_period=star_rotation)
            # Make the transit model
            fleck_times = np.linspace(time[0], time[-1], npoints)
            fleck_transit = star.light_curve(
                self.spotlon[chan][:nspots, None]*unit.deg, 
                self.spotlat[chan][:nspots, None]*unit.deg, 
                self.spotrad[chan][:nspots, None],
                star_inc*unit.deg, 
                planet=bm_params, times=fleck_times).flatten()
            m_transit = np.interp(time, fleck_times, fleck_transit)

            lcfinal = np.append(lcfinal, m_transit/m_transit[0])

        return lcfinal


