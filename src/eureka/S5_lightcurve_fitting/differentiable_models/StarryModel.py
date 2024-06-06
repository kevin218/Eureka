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

        required = np.array(['Rs'])
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

            # Solve Keplerian orbital period equation for system mass
            # (otherwise starry is going to mess with P or a...)
            a = temp.a*temp.Rs*const.R_sun.value
            p = temp.per*(24.*3600.)
            Ms = ((2.*np.pi*a**(3./2.))/p)**2/const.G.value/const.M_sun.value

            # Initialize star object
            star = starry.Primary(starry.Map(udeg=self.udeg),
                                  m=Ms, r=temp.Rs)

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

            # Initialize planet object
            planet = starry.Secondary(
                planet_map,
                m=0,
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

            # Instantiate the system
            system = starry.System(star, planet, light_delay=self.compute_ltt)
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
            rps = self.fit_rps
        else:
            lib = tt
            systems = self.systems
            rps = self.rps

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

            # Combine the planet and stellar flux (allowing negative rp)
            fstar, fp = systems[chan].flux(time, total=False)
            # Do some annoying math to allow theano functions to compile
            # (correctly defined for -1 < rp < 1)
            sign = (lib.ceil(rps[chan])+lib.floor(rps[chan]))
            fstar = (fstar-1)*sign + 1
            fp = fp*sign
            lcpiece = fstar+fp

            if eval:
                lcpiece = lcpiece.eval()
            phys_flux = lib.concatenate([phys_flux, lcpiece])

        return phys_flux

    def compute_fp(self, theta=0):
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
                planet_map = system.secondaries[0].map
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

            # Solve Keplerian orbital period equation for system mass
            # (otherwise starry is going to mess with P or a...)
            a = temp.a*temp.Rs*const.R_sun.value
            p = temp.per*(24.*3600.)
            Ms = ((2.*np.pi*a**(3./2.))/p)**2/const.G.value/const.M_sun.value

            # Initialize star object
            star = starry.Primary(starry.Map(udeg=self.udeg),
                                  m=Ms, r=temp.Rs)

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

            # Initialize planet object
            planet = starry.Secondary(
                planet_map,
                m=0,
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

            # Instantiate the system
            sys = starry.System(star, planet, light_delay=self.compute_ltt)
            self.fit.systems.append(sys)
