import numpy as np
import astropy.units as unit
import inspect

import fleck

from .Model import Model
from ..models.AstroModel import PlanetParams
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
        self.name = 'fleck transit'

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

    def eval(self, channel=None, pid=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        pid : int; optional
            Planet ID, default is None which combines the eclipse models from
            all planets.
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
            
        if pid is None or pid == 0:
            pid_iter = range(self.num_planets)
            pid_id = ''
        else:
            pid_iter = [pid,]
            pid_id = f'_pl{pid}'

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # find number of spots
        spotinds = [s for s in self.parameters.dict.keys() if 'spotrad' in s]   
        counter = [s for s in spotinds if '_' in s]
        nspots = len(spotinds) - len(counter)
        
        # Get relevant spot parameters
        spotrad = np.zeros((self.nchannel_fitted, nspots))
        spotlat = np.zeros((self.nchannel_fitted, nspots))
        spotlon = np.zeros((self.nchannel_fitted, nspots))
        spotcon = np.zeros((self.nchannel_fitted))
        starrot = np.zeros((self.nchannel_fitted))
        starinc = np.zeros((self.nchannel_fitted))

        # Set all parameters
        lcfinal = np.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            # channel ID
            if chan == 0:
                channel_id = ''
            else:
                channel_id = f'_{chan}'

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]
                
            light_curve = np.ma.ones(time.shape)
            for pid in pid_iter:
                # Initialize planet
                pl_params = PlanetParams(self, pid, chan)

                # Set spot/fleck parameters
                # set a star rotation for the star object
                # not actually used in fast mode. 
                # overwritten if user supplies and runs slow mode
                starrot[:] = 100
                fleck_fast = True

                for n in range(nspots):
                    # read radii
                    item0 = 'spotrad' + str(n) + pid_id
                    if self.parameters.dict[item0][1] == 'free':
                        item0 += channel_id
                    value = self.parameters.dict[item0][0]
                    spotrad[chan, n] = value
                    # read latitudes
                    item0 = 'spotlat' + str(n) + pid_id
                    if self.parameters.dict[item0][1] == 'free':
                        item0 += channel_id
                    value = self.parameters.dict[item0][0]
                    spotlat[chan, n] = value
                    # read longitudes
                    item0 = 'spotlon' + str(n) + pid_id
                    if self.parameters.dict[item0][1] == 'free':
                        item0 += channel_id
                    value = self.parameters.dict[item0][0]
                    spotlon[chan, n] = value
                # read contrasts (same for all spots)
                item0 = 'spotcon0' + pid_id
                if self.parameters.dict[item0][1] == 'free':
                    item0 += channel_id
                value = self.parameters.dict[item0][0]
                spotcon[chan] = value
                # read stellar inclination
                item0 = 'spotstari' + pid_id
                if self.parameters.dict[item0][1] == 'free':
                    item0 += channel_id
                value = self.parameters.dict[item0][0]
                starinc[chan] = value
                # read number of points
                npoints = self.parameters.dict['spotnpts'][0]
                # read stellar rotation (if provided)
                if 'spotrot' in self.parameters.dict.keys():
                    starrot[chan] = self.parameters.dict['spotrot'][0]
                    fleck_fast = False
                            
                # Set limb darkening parameters
                uarray = []
                for u in self.coeffs:
                    index = np.where(np.array(self.paramtitles) == u)[0]
                    if len(index) != 0:
                        item = self.longparamlist[chan][index[0]]
                        uarray.append(self.parameters.dict[item][0])
                pl_params.u = uarray
                pl_params.limb_dark = self.parameters.dict['limb_dark'][0]

                # Enforce physicality to avoid crashes from batman by returning
                # something that should be a horrible fit
                if not ((0 < pl_params.per) and (0 < pl_params.inc < 90) and
                        (1 < pl_params.a) and (0 <= pl_params.ecc < 1)):
                    # Returning nans or infs breaks the fits, so this was the
                    # best I could think of
                    light_curve = 1e12*np.ma.ones(time.shape)
                    continue

                # Use batman ld_profile name
                if self.parameters.limb_dark.value == '4-parameter':
                    pl_params.limb_dark = 'nonlinear'
                elif self.parameters.limb_dark.value == 'kipping2013':
                    # Enforce physicality to avoid crashes from batman by
                    # returning something that should be a horrible fit
                    if pl_params.u[0] <= 0:
                        # Returning nans or infs breaks the fits, so this was
                        # the best I could think of
                        lcfinal = 1e12*np.ma.ones(time.shape)
                        continue
                    pl_params.limb_dark = 'quadratic'
                    u1 = 2*np.sqrt(pl_params.u[0])*pl_params.u[1]
                    u2 = np.sqrt(pl_params.u[0])*(1-2*pl_params.u[1])
                    pl_params.u = np.array([u1, u2])

                # Make the star object
                star = fleck.Star(spot_contrast=spotcon[chan], 
                                  u_ld=pl_params.u, 
                                  rotation_period=starrot[chan])
                # Make the transit model
                fleck_times = np.linspace(time[0], time[-1], npoints)
                fleck_transit = star.light_curve(
                    spotlon[chan][:, None]*unit.deg, 
                    spotlat[chan][:, None]*unit.deg, 
                    spotrad[chan][:, None],
                    starinc[chan]*unit.deg, 
                    planet=pl_params, times=fleck_times,
                    fast=fleck_fast).flatten()
                m_transit = np.interp(time, fleck_times, fleck_transit)
                
                light_curve *= m_transit/m_transit[0]

            lcfinal = np.append(lcfinal, light_curve)

        return lcfinal


