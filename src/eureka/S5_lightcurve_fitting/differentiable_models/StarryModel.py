import numpy as np
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


class temp_class:
    def __init__(self):
        pass


class StarryModel(PyMC3Model):
    def __init__(self, **kwargs):
        # Inherit from PyMC3Model class
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

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
        if np.any(np.array(['Y2' == par[:2] for par in self.paramtitles])):
            self.ydeg = 2
        elif np.any(np.array(['Y1' == par[:2] in par
                              for par in self.paramtitles])):
            self.ydeg = 1
        else:
            self.ydeg = 0

    def setup(self, full_model):
        self.full_model = full_model

        self.systems = []
        for c in range(self.nchan):
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
                    setattr(temp, key, getattr(self.model, key+'_'+str(c)))
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
                amp = temp.fp/planet_map2.flux(theta=0)[0]
                planet_map.amp = amp

            # Initialize planet object
            planet = starry.Secondary(
                planet_map,
                # Convert mass to M_sun units
                # m=temp.Mp*const.M_jup.value/const.M_sun.value,
                m=Mp,
                # Convert radius to R_star units
                r=temp.rp*temp.Rs,
                # Setting porb here overwrites a
                a=temp.a,
                # porb = temp.per,
                # prot = temp.per,
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

            # Instantiate the system
            system = starry.System(star, planet)
            self.systems.append(system)

    def eval(self, eval=True, channel=None, **kwargs):
        if channel is None:
            nchan = self.nchan
            channels = np.arange(nchan)
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
        for c in channels:
            lcpiece = systems[c].flux(self.time)
            if eval:
                lcpiece = lcpiece.eval()
        phys_flux = lib.concatenate([phys_flux, lcpiece])

        return phys_flux

    def compute_fp(self, theta=0):
        with self.full_model.model:
            fps = []
            for c in range(self.nchan):
                planet_map = self.fit.systems[c].secondaries[0].map
                fps.append(planet_map.flux(theta=theta).eval())
            return np.array(fps)

    def update(self, newparams):
        super().update(newparams)

        self.fit.systems = []
        for c in range(self.nchan):
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
                amp = temp.fp/planet_map2.flux(theta=0)[0]
                planet_map.amp = amp

            # Initialize planet object
            planet = starry.Secondary(
                planet_map,
                # Convert mass to M_sun units
                # m=temp.Mp*const.M_jup.value/const.M_sun.value,
                m=Mp,
                # Convert radius to R_star units
                r=temp.rp*temp.Rs,
                # Setting porb here overwrites a
                a=temp.a,
                # porb = temp.per,
                # prot = temp.per,
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

            # Instantiate the system
            sys = starry.System(star, planet)
            self.fit.systems.append(sys)

    @property
    def time(self):
        """A getter for the time"""
        return self._time

    @time.setter
    def time(self, time_array, time_units='BJD'):
        """A setter for the time

        Parameters
        ----------
        time_array: sequence, astropy.units.quantity.Quantity
            The time array
        time_units: str
            The units of the input time_array, ['MJD', 'BJD', 'phase']
        """
        # Check the type
        if not isinstance(time_array, (np.ndarray, tuple, list)):
            raise TypeError("Time axis must be a tuple, list, or numpy array.")

        # Set the units
        self.time_units = time_units

        # Set the array
        # self._time = np.array(time_array)
        self._time = np.ma.masked_array(time_array)