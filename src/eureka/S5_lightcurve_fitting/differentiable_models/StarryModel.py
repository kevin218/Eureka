import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import starry
import theano.tensor as tt
import astropy.constants as const

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from . import PyMC3Model

starry.config.quiet = True
starry.config.lazy = True


class fit_class:
    def __init__(self):
        pass


class StarryModel(PyMC3Model):
    def __init__(self, model, **kwargs):
        # Inherit from Model class
        super().__init__(**kwargs)

        self.model = model

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        # Check for Parameters instance
        self.parameters = kwargs.get('parameters')
        # Set parameters for multi-channel fits
        self.longparamlist = kwargs.get('longparamlist')
        self.nchan = kwargs.get('nchan')
        self.paramtitles = kwargs.get('paramtitles')
        self.uniqueparams = np.unique(self.longparamlist)

        self.components = [self]

        required = np.array(['Ms', 'Mp', 'Rs'])
        missing = np.array([name not in self.paramtitles for name in required])
        if np.any(missing):
            message = (f'Missing required params {required[missing]} in your '
                       f'EPF.')
            raise AssertionError(message)

        if hasattr(self, 'u2'):
            self.udeg = 2
        elif hasattr(self, 'u1'):
            self.udeg = 1
        else:
            self.udeg = 0
        if hasattr(self, 'AmpCos2') or hasattr(self, 'AmpSin2'):
            self.ydeg = 2
        elif hasattr(self, 'AmpCos1') or hasattr(self, 'AmpSin1'):
            self.ydeg = 1
        else:
            self.ydeg = 0
        
        self.systems = []
        for c in range(self.nchan):
            # Initialize star object
            star = starry.Primary(starry.Map(ydeg=0, udeg=self.udeg,
                                             amp=1.0),
                                  m=self.Ms, r=self.Rs, prot=1.0)

            # To save ourselves from tonnes of getattr lines, let's make a
            # new object without the _c parts of the parnames
            # This way we can do `temp.u1` rather than
            # `getattr(self, 'u1_'+c)`
            temp = fit_class()
            for key in self.paramtitles:
                if getattr(self.parameters, key).ptype in ['free',
                                                           'shared',
                                                           'fixed',
                                                           'white_free',
                                                           'white_fixed']:
                    if (getattr(self.parameters, key).ptype != 'fixed'
                            and c > 0):
                        # Remove the _c part of the parname but leave any
                        # other underscores intact
                        setattr(temp, key, getattr(self, key+'_'+str(c)))
                    else:
                        setattr(temp, key, getattr(self, key))

            # FINDME: non-uniform limb darkening does not currently work
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
            
            if hasattr(self, 'fp'):
                amp = temp.fp
            else:
                amp = 0
            # Initialize planet object
            planet = starry.Secondary(
                starry.Map(ydeg=self.ydeg, udeg=0, amp=amp, inc=90.0,
                           obl=0.0),
                # Convert mass to M_sun units
                m=self.Mp*const.M_jup.value/const.M_sun.value,
                # Convert radius to R_star units
                r=temp.rp*self.Rs,
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
            if hasattr(self, 'AmpCos1'):
                planet.map[1, 0] = temp.AmpCos1
            if hasattr(self, 'AmpSin1'):
                planet.map[1, 1] = temp.AmpSin1
            if self.ydeg == 2:
                if hasattr(self, 'AmpCos2'):
                    planet.map[2, 0] = temp.AmpCos2
                if hasattr(self, 'AmpSin2'):
                    planet.map[2, 1] = temp.AmpSin2
            # Offset is controlled by AmpSin1
            planet.theta0 = 180.0
            planet.tref = 0

            # Instantiate the system
            sys = starry.System(star, planet)
            self.systems.append(sys)

    def eval(self, interp=False, eval=True, channel=None):
        if channel is None:
            nchan = self.nchan
            channels = np.arange(nchan)
        else:
            nchan = 1
            channels = [channel, ]

        if interp:
            dt = self.time[1]-self.time[0]
            steps = int(np.round((self.time[-1]-self.time[0])/dt+1))
            new_time = np.linspace(self.time[0], self.time[-1], steps,
                                   endpoint=True)
        else:
            new_time = self.time

        if eval:
            phys_flux = np.zeros(0)
            for chan in range(nchan):
                c = channels[chan]
                lcpiece = self.fit.systems[c].flux(new_time-self.fit.t0).eval()
                phys_flux = np.concatenate([phys_flux, lcpiece])

            return phys_flux, new_time
        else:
            phys_flux = tt.zeros(0)
            for chan in range(nchan):
                c = channels[chan]
                lcpiece = self.systems[c].flux(new_time-self.t0)
                phys_flux = tt.concatenate([phys_flux, lcpiece])

            return phys_flux, new_time

    @property
    def fit_dict(self):
        return self._fit_dict

    @fit_dict.setter
    def fit_dict(self, input_fit_dict):
        self._fit_dict = input_fit_dict

        fit = fit_class()
        for key in self.fit_dict.keys():
            setattr(fit, key, self.fit_dict[key])

        for parname in self.uniqueparams:
            param = getattr(self.parameters, parname)
            if param.ptype == 'independent':
                continue
            elif param.ptype == 'fixed':
                setattr(fit, parname, param.value)
            elif param.ptype == 'shared':
                for c in range(1, self.nchan):
                    parname_temp = parname+'_'+str(c)
                    setattr(fit, parname_temp, getattr(fit, parname))
                    self._fit_dict[parname_temp] = getattr(fit, parname)

        if hasattr(self, 'u2'):
            fit.udeg = 2
        elif hasattr(self, 'u1'):
            fit.udeg = 1
        else:
            fit.udeg = 0
        if hasattr(self, 'AmpCos2') or hasattr(self, 'AmpSin2'):
            fit.ydeg = 2
        elif hasattr(self, 'AmpCos1') or hasattr(self, 'AmpSin1'):
            fit.ydeg = 1
        else:
            fit.ydeg = 0

        fit.systems = []
        for c in range(self.nchan):
            # Initialize star object
            star = starry.Primary(starry.Map(ydeg=0, udeg=fit.udeg, amp=1.0),
                                  m=self.Ms, r=self.Rs, prot=1.0)

            # To save ourselves from tonnes of getattr lines, let's make a new
            # object without the _c parts of the parnames
            # This way we can do `temp.u1` rather than
            # `getattr(self, 'u1_'+c)`
            temp = fit_class()
            for key in self.paramtitles:
                if getattr(self.parameters, key).ptype in ['free', 'shared',
                                                           'fixed']:
                    if (c > 0 and
                            getattr(self.parameters, key).ptype != 'fixed'):
                        # Remove the _c part of the parname but leave any
                        # other underscores intact
                        setattr(temp, key, getattr(fit, key+'_'+str(c)))
                    else:
                        setattr(temp, key, getattr(fit, key))

            # FINDME: non-uniform limb darkening does not currently work
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
            
            if hasattr(self, 'fp'):
                amp = temp.fp
            else:
                amp = 0
            # Initialize planet object
            planet = starry.Secondary(
                starry.Map(ydeg=fit.ydeg, udeg=0, amp=amp, inc=90.0, obl=0.0),
                # Convert mass to M_sun units
                m=self.Mp*const.M_jup.value/const.M_sun.value,
                # Convert radius to R_star units
                r=temp.rp*self.Rs,
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
            if hasattr(self, 'AmpCos1'):
                planet.map[1, 0] = temp.AmpCos1
            if hasattr(self, 'AmpSin1'):
                planet.map[1, 1] = temp.AmpSin1
            if self.ydeg == 2:
                if hasattr(self, 'AmpCos2'):
                    planet.map[2, 0] = temp.AmpCos2
                if hasattr(self, 'AmpSin2'):
                    planet.map[2, 1] = temp.AmpSin2
            # Offset is controlled by AmpSin1
            planet.theta0 = 180.0
            planet.tref = 0

            # Instantiate the system
            sys = starry.System(star, planet)
            fit.systems.append(sys)

        self.fit = fit

        return

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
