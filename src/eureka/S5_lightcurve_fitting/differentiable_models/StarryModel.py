import numpy as np
import inspect
import astropy.constants as const

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

import starry
starry.config.quiet = True
starry.config.lazy = True

from . import PyMC3Model
from ..limb_darkening_fit import ld_profile
from ...lib.split_channels import split


class temp_class:
    def __init__(self):
        pass


class StarryModel(PyMC3Model):
    def __init__(self, **kwargs):
        """Initialize the model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.differentiable_models.PyMC3Model.__init__().
        """
        # Inherit from PyMC3Model class
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        # Set default to turn light-travel correction on if not specified
        if not hasattr(self, 'compute_ltt') or self.compute_ltt is None:
            self.compute_ltt = True

        required = np.array(['Ms', 'Rs'])
        missing = np.array([name not in self.paramtitles for name in required])
        if np.any(missing):
            message = (f'Missing required params {required[missing]} in your '
                       f'EPF.')
            raise AssertionError(message)

        if 'u2' in self.paramtitles:
            self.udeg = 2
        elif 'u1' in self.paramtitles:
            self.udeg = 1
        else:
            self.udeg = 0

        # Find the Ylm value with the largest l value to set ydeg
        ylm_params = np.where(['Y' == par[0] and par[1].isnumeric()
                               for par in self.paramtitles])[0]
        if len(ylm_params) > 0:
            l_vals = [int(self.paramtitles[ind][1])
                      for ind in ylm_params]
            self.ydeg = max(l_vals)
        else:
            self.ydeg = 0
        
        # Store the ld_profile
        self.ld_from_S4 = kwargs.get('ld_from_S4')
        if hasattr(self.parameters, 'limb_dark'):
            ld_func = ld_profile(self.parameters.limb_dark.value, 
                                 use_gen_ld=self.ld_from_S4)
            len_params = len(inspect.signature(ld_func).parameters)
            self.coeffs = ['u{}'.format(n) for n in range(len_params)[1:]]

        self.ld_from_file = kwargs.get('ld_from_file')

        self.recenter_ld_prior = kwargs.get('recenter_ld_prior')

        # Replace u parameters with generated limb-darkening values
        if self.ld_from_S4 or self.ld_from_file:
            self.ld_array = kwargs.get('ld_coeffs')
            if self.ld_from_S4:
                self.ld_array = self.ld_array[len_params-2]
            for c in range(self.nchannel_fitted):
                chan = self.fitted_channels[c]
                for u in self.coeffs:
                    index = np.where(np.array(self.paramtitles) == u)[0]
                    if len(index) != 0:
                        item = self.longparamlist[c][index[0]]
                        param = int(item.split('_')[0][-1])
                        ld_val = self.ld_array[chan][param-1]
                        # Use the file value as the starting guess
                        self.parameters.dict[item][0] = ld_val
                        # In a normal prior, center at the file value
                        if (self.parameters.dict[item][-1] == 'N' and
                                self.recenter_ld_prior):
                            self.parameters.dict[item][-3] = ld_val
                        # Update the non-dictionary form as well
                        setattr(self.parameters, item,
                                self.parameters.dict[item])

    def setup(self):
        """Setup a model for evaluation and fitting.
        """
        self.systems = []
        self.rps = []
        self.rps_2 = []
        self.rps_3 = []
        for c in range(self.nchannel_fitted):
            # To save ourselves from tonnes of getattr lines, let's make a
            # new object without the _c parts of the parnames
            # For example, this way we can do `temp.u1` rather than
            # `getattr(self.model, 'u1_'+c)`.
            temp = temp_class()
            for key in self.paramtitles:
                ptype = getattr(self.parameters, key).ptype
                if (ptype not in ['fixed', 'independent']
                        and c > 0):
                    # Remove the _c part of the parname but leave any
                    # other underscores intact
                    setattr(temp, key, getattr(self.model,
                                               key+'_'+str(c)))
                else:
                    setattr(temp, key, getattr(self.model, key))

            # Initialize star object
            star = starry.Primary(starry.Map(udeg=self.udeg),
                                  m=temp.Ms, r=temp.Rs)

            if hasattr(self.parameters, 'limb_dark'):
                if self.parameters.limb_dark.value == 'kipping2013':
                    # Transform stellar variables to uniform used by starry
                    star.map[1] = 2*tt.sqrt(temp.u1)*temp.u2
                    star.map[2] = tt.sqrt(temp.u1)*(1-2*temp.u2)
                elif self.parameters.limb_dark.value == 'quadratic':
                    star.map[1] = temp.u1
                    star.map[2] = temp.u2
                elif self.parameters.limb_dark.value == 'linear':
                    star.map[1] = temp.u1
                elif self.parameters.limb_dark.value != 'uniform':
                    message = (f'ERROR: starryModel is not yet able to '
                               f'handle {self.parameters.limb_dark.value} '
                               f'limb darkening.\n'
                               f'       limb_dark must be one of uniform, '
                               f'linear, quadratic, or kipping2013.')
                    raise ValueError(message)

            # Solve Keplerian orbital period equation for Mp
            # (otherwise starry is going to mess with P or a...)
            a = temp.a*temp.Rs*const.R_sun.value
            p = temp.per*(24.*3600.)
            Mp = (((2.*np.pi*a**(3./2.))/p)**2/const.G.value/const.M_sun.value
                  - temp.Ms)

            a_2 = temp.a2*temp.Rs*const.R_sun.value
            p_2 = temp.per2*(24.*3600.)
            Mp_2 = (((2.*np.pi*a_2**(3./2.))/p_2)**2
                    / const.G.value/const.M_sun.value
                    - temp.Ms)
            
            a_3 = temp.a3*temp.Rs*const.R_sun.value
            p_3 = temp.per3*(24.*3600.)
            Mp_3 = (((2.*np.pi*a_3**(3./2.))/p_3)**2
                    / const.G.value/const.M_sun.value
                    - temp.Ms)

            if not hasattr(temp, 'fp'):
                planet_map = starry.Map(ydeg=self.ydeg, amp=0)
            else:
                planet_map = starry.Map(ydeg=self.ydeg)
                planet_map2 = starry.Map(ydeg=self.ydeg)
                for ell in range(1, self.ydeg+1):
                    for m in range(-ell, ell+1):
                        if hasattr(temp, f'Y{ell}{m}'):
                            planet_map[ell, m] = getattr(temp, f'Y{ell}{m}')
                            planet_map2[ell, m] = getattr(temp, f'Y{ell}{m}')
                amp = temp.fp/tt.abs_(planet_map2.flux(theta=0)[0])
                planet_map.amp = amp
            self.rps.append(temp.rp)

            if not hasattr(temp, 'fp2'):
                planet_map_2 = starry.Map(ydeg=0, amp=0)
            else:
                planet_map_2 = starry.Map(ydeg=self.ydeg)
                planet_map2_2 = starry.Map(ydeg=self.ydeg)
                for ell in range(1, self.ydeg+1):
                    for m in range(-ell, ell+1):
                        if hasattr(temp, f'Y{ell}{m}'):
                            planet_map_2[ell, m] = getattr(temp,
                                                           f'Y{ell}{m}2')
                            planet_map2_2[ell, m] = getattr(temp,
                                                            f'Y{ell}{m}2')
                amp_2 = temp.fp2/tt.abs_(planet_map2_2.flux(theta=0)[0])
                planet_map_2.amp = amp_2
            self.rps_2.append(temp.rp2)

            planet_map_3 = starry.Map(ydeg=0, amp=0)
            self.rps_3.append(temp.rp3)

            # The following code should work but doesn't see to work well
            # self.model.ecc = tt.sqrt(temp.ecosw**2 + temp.esinw**2)
            # longitude of periastron needs to be in degrees for batman!
            # self.model.w = tt.arctan2(temp.esinw, temp.ecosw)*180./np.pi

            # Initialize planet object
            planet = starry.Secondary(
                planet_map,
                m=Mp,
                # Convert radius to R_star units
                r=tt.abs_(temp.rp)*temp.Rs,
                # Setting porb here overwrites a
                a=temp.a,
                # Another option to set inclination using impact parameter
                # inc=tt.arccos(b/a)*180/np.pi
                inc=temp.inc,
                ecc=temp.ecc,
                w=temp.w
            )
            # Setting porb here may not override a
            planet.porb = temp.per
            # Setting prot here may not override a
            planet.prot = temp.per
            # Offset is controlled by Y11
            planet.theta0 = 180.0
            planet.t0 = temp.t0

            # The following code should work but doesn't see to work well
            # self.model.ecc2 = tt.sqrt(temp.ecosw2**2 + temp.esinw2**2)
            # longitude of periastron needs to be in degrees for batman!
            # self.model.w2 = tt.arctan2(temp.esinw2, temp.ecosw2)*180./np.pi

            # Initialize planet object
            planet_2 = starry.Secondary(
                planet_map_2,
                m=Mp_2,
                # Convert radius to R_star units
                r=tt.abs_(temp.rp2)*temp.Rs,
                # Setting porb here overwrites a
                a=temp.a2,
                # Another option to set inclination using impact parameter
                # inc=tt.arccos(b/a)*180/np.pi
                inc=temp.inc2,
                ecc=temp.ecc2,
                w=temp.w2
            )
            # Setting porb here may not override a
            planet_2.porb = temp.per2
            # Setting prot here may not override a
            planet_2.prot = temp.per2
            # Offset is controlled by Y11
            planet_2.theta0 = 180.0
            planet_2.t0 = temp.t02

            # The following code should work but doesn't see to work well
            # self.model.ecc3 = tt.sqrt(temp.ecosw3**2 + temp.esinw3**2)
            # longitude of periastron needs to be in degrees for batman!
            # self.model.w3 = tt.arctan2(temp.esinw3, temp.ecosw3)*180./np.pi

            # Initialize planet object
            planet_3 = starry.Secondary(
                planet_map_3,
                m=Mp_3,
                # Convert radius to R_star units
                r=tt.abs_(temp.rp3)*temp.Rs,
                # Setting porb here overwrites a
                a=temp.a3,
                # Another option to set inclination using impact parameter
                # inc=tt.arccos(b/a)*180/np.pi
                inc=temp.inc3,
                ecc=temp.ecc3,
                w=temp.w3
            )
            # Setting porb here may not override a
            planet_3.porb = temp.per3
            # Setting prot here may not override a
            planet_3.prot = temp.per3
            # Offset is controlled by Y11
            planet_3.theta0 = 180.0
            planet_3.t0 = temp.t03

            # Instantiate the system
            system = starry.System(star, planet, planet_2, planet_3,
                                   light_delay=self.compute_ltt)
            self.systems.append(system)

    def eval(self, eval=True, channel=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        ndarray
            The value of the model at the times self.time.
        """
        if channel is None:
            nchan = self.nchannel_fitted
            channels = self.fitted_channels
        else:
            nchan = 1
            channels = [channel, ]

        if eval:
            lib = np
            systems = self.fit.systems
        else:
            lib = tt
            systems = self.systems

        phys_flux = lib.zeros(0)
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            # Combine the planet and stellar flux
            lcpiece = systems[chan].flux(time)

            if eval:
                lcpiece = lcpiece.eval()
            phys_flux = lib.concatenate([phys_flux, lcpiece])

        return phys_flux

    def compute_fp(self, theta=0, planet=0):
        """Compute the planetary flux at an arbitrary orbital position.

        Parameters
        ----------
        theta : int, ndarray; optional
            The orbital angle(s) in degrees with respect to mid-eclipse.
            Defaults to 0.

        Returns
        -------
        ndarray
            The disk-integrated planetary flux for each value of theta.
        """
        with self.model:
            fps = []
            for system in self.fit.systems:
                planet_map = system.secondaries[planet].map
                fps.append(planet_map.flux(theta=theta).eval())
            return np.array(fps)

    def update(self, newparams, **kwargs):
        """Update parameters and update the self.fit.systems list.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.differentiable_models.PyMC3Model.update().
        """
        super().update(newparams, **kwargs)

        self.fit.systems = []
        self.fit_rps = []
        self.fit_rps_2 = []
        self.fit_rps_3 = []
        for c in range(self.nchannel_fitted):
            # To save ourselves from tonnes of getattr lines, let's make a
            # new object without the _c parts of the parnames
            # For example, this way we can do `temp.u1` rather than
            # `getattr(self.model, 'u1_'+c)`.
            temp = temp_class()
            for key in self.paramtitles:
                ptype = getattr(self.parameters, key).ptype
                if (ptype not in ['fixed', 'independent', 'shared']
                        and c > 0):
                    # Remove the _c part of the parname but leave any
                    # other underscores intact
                    setattr(temp, key, getattr(self.fit, key+'_'+str(c)))
                else:
                    setattr(temp, key, getattr(self.fit, key))

            # Initialize star object
            star = starry.Primary(starry.Map(udeg=self.udeg),
                                  m=temp.Ms, r=temp.Rs)

            if hasattr(self.parameters, 'limb_dark'):
                if self.parameters.limb_dark.value == 'kipping2013':
                    # Transform stellar variables to uniform used by starry
                    star.map[1] = 2*np.sqrt(temp.u1)*temp.u2
                    star.map[2] = np.sqrt(temp.u1)*(1-2*temp.u2)
                elif self.parameters.limb_dark.value == 'quadratic':
                    star.map[1] = temp.u1
                    star.map[2] = temp.u2
                elif self.parameters.limb_dark.value == 'linear':
                    star.map[1] = temp.u1
                elif self.parameters.limb_dark.value != 'uniform':
                    message = (f'ERROR: starryModel is not yet able to handle '
                               f'{self.parameters.limb_dark.value} '
                               f'limb_dark.\n'
                               f'       limb_dark must be one of uniform, '
                               f'linear, quadratic, or kipping2013.')
                    raise ValueError(message)
            
            # Solve Keplerian orbital period equation for Mp
            # (otherwise starry is going to mess with P or a...)
            a = temp.a*temp.Rs*const.R_sun.value
            p = temp.per*(24.*3600.)
            Mp = (((2.*np.pi*a**(3./2.))/p)**2/const.G.value/const.M_sun.value
                  - temp.Ms)
            
            # Solve Keplerian orbital period equation for Mp
            # (otherwise starry is going to mess with P or a...)
            a_2 = temp.a2*temp.Rs*const.R_sun.value
            p_2 = temp.per2*(24.*3600.)
            Mp_2 = (((2.*np.pi*a_2**(3./2.))/p_2)**2
                    / const.G.value/const.M_sun.value
                    - temp.Ms)
            
            # Solve Keplerian orbital period equation for Mp
            # (otherwise starry is going to mess with P or a...)
            a_3 = temp.a3*temp.Rs*const.R_sun.value
            p_3 = temp.per3*(24.*3600.)
            Mp_3 = (((2.*np.pi*a_3**(3./2.))/p_3)**2
                    / const.G.value/const.M_sun.value
                    - temp.Ms)

            if not hasattr(temp, 'fp'):
                planet_map = starry.Map(ydeg=self.ydeg, amp=0)
            else:
                planet_map = starry.Map(ydeg=self.ydeg)
                planet_map2 = starry.Map(ydeg=self.ydeg)
                for ell in range(1, self.ydeg+1):
                    for m in range(-ell, ell+1):
                        if hasattr(temp, f'Y{ell}{m}'):
                            planet_map[ell, m] = getattr(temp, f'Y{ell}{m}')
                            planet_map2[ell, m] = getattr(temp, f'Y{ell}{m}')
                amp = temp.fp/np.abs(planet_map2.flux(theta=0)[0])
                planet_map.amp = amp
            self.fit_rps.append(temp.rp)

            if not hasattr(temp, 'fp2'):
                planet_map_2 = starry.Map(ydeg=self.ydeg, amp=0)
            else:
                planet_map_2 = starry.Map(ydeg=self.ydeg)
                planet_map2_2 = starry.Map(ydeg=self.ydeg)
                for ell in range(1, self.ydeg+1):
                    for m in range(-ell, ell+1):
                        if hasattr(temp, f'Y{ell}{m}'):
                            planet_map_2[ell, m] = getattr(temp,
                                                           f'Y{ell}{m}2')
                            planet_map2_2[ell, m] = getattr(temp,
                                                            f'Y{ell}{m}2')
                amp_2 = temp.fp2/np.abs(planet_map2_2.flux(theta=0)[0])
                planet_map_2.amp = amp_2
            self.fit_rps_2.append(temp.rp2)

            planet_map_3 = starry.Map(ydeg=0, amp=0)
            self.fit_rps_3.append(temp.rp3)

            # The following code should work but doesn't see to work well
            # ecc = np.sqrt(temp.ecosw**2 + temp.esinw**2)
            # longitude of periastron needs to be in degrees for batman!
            # w = np.arctan2(temp.esinw, temp.ecosw)*180./np.pi

            # Initialize planet object
            planet = starry.Secondary(
                planet_map,
                m=Mp,
                # Convert radius to R_star units
                r=np.abs(temp.rp)*temp.Rs,
                # Setting porb here overwrites a
                a=temp.a,
                # Another option to set inclination using impact parameter
                # inc=tt.arccos(b/a)*180/np.pi
                inc=temp.inc,
                ecc=temp.ecc,
                w=temp.w
            )
            # Setting porb here may not override a
            planet.porb = temp.per
            # Setting prot here may not override a
            planet.prot = temp.per
            # Offset is controlled by Y11
            planet.theta0 = 180.0
            planet.t0 = temp.t0

            # The following code should work but doesn't see to work well
            # ecc2 = np.sqrt(temp.ecosw2**2 + temp.esinw2**2)
            # longitude of periastron needs to be in degrees for batman!
            # w2 = np.arctan2(temp.esinw2, temp.ecosw2)*180./np.pi

            # Initialize planet object
            planet_2 = starry.Secondary(
                planet_map_2,
                m=Mp_2,
                # Convert radius to R_star units
                r=np.abs(temp.rp2)*temp.Rs,
                # Setting porb here overwrites a
                a=temp.a2,
                # Another option to set inclination using impact parameter
                # inc=tt.arccos(b/a)*180/np.pi
                inc=temp.inc2,
                ecc=temp.ecc2,
                w=temp.w2
            )
            # Setting porb here may not override a
            planet_2.porb = temp.per2
            # Setting prot here may not override a
            planet_2.prot = temp.per2
            # Offset is controlled by Y11
            planet_2.theta0 = 180.0
            planet_2.t0 = temp.t02

            # The following code should work but doesn't see to work well
            # ecc3 = np.sqrt(temp.ecosw3**2 + temp.esinw3**2)
            # longitude of periastron needs to be in degrees for batman!
            # w3 = np.arctan2(temp.esinw3, temp.ecosw3)*180./np.pi

            # Initialize planet object
            planet_3 = starry.Secondary(
                planet_map_3,
                m=Mp_3,
                # Convert radius to R_star units
                r=np.abs(temp.rp3)*temp.Rs,
                # Setting porb here overwrites a
                a=temp.a3,
                # Another option to set inclination using impact parameter
                # inc=tt.arccos(b/a)*180/np.pi
                inc=temp.inc3,
                ecc=temp.ecc3,
                w=temp.w3
            )
            # Setting porb here may not override a
            planet_3.porb = temp.per3
            # Setting prot here may not override a
            planet_3.prot = temp.per3
            # Offset is controlled by Y11
            planet_3.theta0 = 180.0
            planet_3.t0 = temp.t03

            # Instantiate the system
            sys = starry.System(star, planet, planet_2, planet_3,
                                light_delay=self.compute_ltt)
            self.fit.systems.append(sys)
