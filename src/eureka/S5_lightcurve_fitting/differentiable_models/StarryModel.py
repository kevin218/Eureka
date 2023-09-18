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
import pymc3 as pm

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
        elif 'pixel_ydeg' in self.paramtitles:
            # read l order used for pixel sampling
            self.ydeg = self.parameters.pixel_ydeg.value
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

            if not hasattr(temp, 'fp'):
                planet_map = starry.Map(ydeg=self.ydeg, amp=0)
            else:
                planet_map = starry.Map(ydeg=self.ydeg)
                planet_map2 = starry.Map(ydeg=self.ydeg)
                for ell in range(1, self.ydeg+1):
                    for m in range(-ell, ell+1):
                        if hasattr(temp, f'Y{ell}{m}'):
                            planet_map[ell, m] = getattr(temp,
                                                         f'Y{ell}{m}')
                            planet_map2[ell, m] = getattr(temp,
                                                         f'Y{ell}{m}')
                amp = temp.fp/planet_map2.flux(theta=0)[0]
                planet_map.amp = amp

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

            # Pixel sampling setup
            if 'pixel_ydeg' in self.paramtitles:
                # Always set this to a factor 3 to ensure proper sampling (~ 4*L^2)
                self.oversample = 3

                # Get pixel transform matrix and number of pixels
                A = planet.map.get_pixel_transforms(
                    oversample=self.oversample)[3]
                self.npix = A.shape[1]

                # Set prior to either be log normal, or normal around zero
                pixel_prior_mean = self.parameters.pixel_prior_mean.value
                pixel_prior_width = self.parameters.pixel_prior_width.value
                if self.force_positivity:
                    p = pm.LogNormal("p", mu=np.log(pixel_prior_mean/np.pi), sigma = pixel_prior_width/pixel_prior_mean,
                                     shape=(self.npix,))
                else:
                    p = pm.Normal("p", mu=pixel_prior_mean/np.pi, sd=pixel_prior_width/np.pi,
                                  shape=(self.npix, ))

                # Transform pixels to spherical harmonics
                self.starry_x = tt.dot(A, p)
                # Record spherical harmonics
                pm.Deterministic("y", self.starry_x)

            # Instantiate the system
            system = starry.System(star, planet)

            if 'pixel_ydeg' in self.paramtitles:
                # Calculate light curve by multiplying spherical harmonics by
                # design matrix, then record
                self.starry_X = system.design_matrix(self.time)
                lcpiece = self.starry_X[:, 0] + tt.dot(self.starry_X[:, 1:],
                                                       self.starry_x)

                # Calculate and record map
                map_plot = starry.Map(ydeg=self.ydeg)
                map_plot.amp = self.starry_x[0]
                map_plot[1:, :] = self.starry_x[1:]/self.starry_x[0]

                pm.Deterministic("flux_model", lcpiece)
                pm.Deterministic("map_grid",
                                 np.pi*map_plot.render(projection="rect",
                                                       res=100))

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
            lessthan = np.less
            systems = self.fit.systems
            rps = [systems[chan].secondaries[0].r.eval()
                   for chan in range(nchan)]
        else:
            lib = tt
            lessthan = tt.lt
            systems = self.systems
            rps = [systems[chan].secondaries[0].r for chan in range(nchan)]

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
            if 'pixel_ydeg' in self.paramtitles:
                lcpiece = self.starry_X[:, 0] + tt.dot(self.starry_X[:, 1:],
                                                       self.starry_x)
            else:
                fstar, fp = systems[chan].flux(time, total=False)
                if lessthan(rps[chan], 0):
                    fstar = 2-fstar
                    fp *= -1
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

            if not hasattr(temp, 'fp'):
                planet_map = starry.Map(ydeg=self.ydeg, amp=0)
            else:
                planet_map = starry.Map(ydeg=self.ydeg)
                planet_map2 = starry.Map(ydeg=self.ydeg)
                for ell in range(1, self.ydeg+1):
                    for m in range(-ell, ell+1):
                        if hasattr(temp, f'Y{ell}{m}'):
                            planet_map[ell, m] = getattr(temp,
                                                        f'Y{ell}{m}')
                            planet_map2[ell, m] = getattr(temp,
                                                         f'Y{ell}{m}')
                amp = temp.fp/planet_map2.flux(theta=0)[0]
                planet_map.amp = amp

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

            if 'pixel_ydeg' in self.paramtitles:
                # import pixel values and convert to spherical harmonics
                A = planet.map.get_pixel_transforms(
                    oversample=self.oversample)[3]
                p_fit = newparams[-self.npix:]
                self.starry_x = tt.dot(A, p_fit)
                # Instantiate the system
                sys = starry.System(star, planet)
                self.starry_X = sys.design_matrix(self.time)
            else:
                sys = starry.System(star, planet)

            self.fit.systems.append(sys)
