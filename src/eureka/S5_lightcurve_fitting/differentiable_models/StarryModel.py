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
from .AstroModel import PlanetParams
from ..limb_darkening_fit import ld_profile
from ...lib.split_channels import split


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
        super().__init__(name='starry', modeltype='physical', **kwargs)

        # Set default to turn light-travel correction on if not specified
        if self.compute_ltt is None:
            self.compute_ltt = True

        log = kwargs.get('log')

        required = np.array(['Rs',])
        missing = np.array([name not in self.paramtitles for name in required])
        if np.any(missing):
            message = (f'Missing required params {required[missing]} in your '
                       'EPF. Make sure it is not set to \'independent\' as '
                       'this is no longer a supported option; you can set '
                       'these parameters to fixed if you want to maintain the '
                       'old \'independent\' behavior.')
            raise AssertionError(message)

        if 'u2' in self.paramtitles:
            self.udeg = 2
        elif 'u1' in self.paramtitles:
            self.udeg = 1
        else:
            self.udeg = 0

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
                        log.writelog(f"{item}: {ld_val}")
                        # Use the file value as the starting guess
                        self.parameters.dict[item][0] = ld_val
                        # In a normal prior, center at the file value
                        if (self.parameters.dict[item][-1] == 'N' and
                                self.recenter_ld_prior):
                            self.parameters.dict[item][-3] = ld_val
                        # Update the non-dictionary form as well
                        setattr(self.parameters, item,
                                self.parameters.dict[item])

        self.spotcon_file = kwargs.get('spotcon_file')
        if self.spotcon_file:
            # Load spot contrast coefficients from a custom file
            try:
                spot_coeffs = np.genfromtxt(self.spotcon_file)
            except FileNotFoundError:
                raise Exception(f"The spot contrast file {self.spotcon_file}"
                                " could not be found.")

            nspots = len([s for s in self.parameters.dict.keys()
                          if 'spotrad' in s and '_' not in s])

            # Fix array shaping if only one contrast specified for all spots
            if len(spot_coeffs.shape) == 1:
                spot_coeffs = np.repeat(spot_coeffs[np.newaxis, :],
                                        nspots, axis=0)

            # Load all spot contrasts into the parameters object
            log.writelog("Using the following spot contrast values:")
            for c in range(self.nchannel_fitted):
                chan = self.fitted_channels[c]
                if c == 0 or self.nchannel_fitted == 1:
                    chankey = ''
                else:
                    chankey = f'_ch{chan}'
                for n in range(nspots):
                    item = f'spotcon{n}{chankey}'
                    if item in self.paramtitles:
                        contrast_val = spot_coeffs[n, chan]
                        log.writelog(f"{item}: {contrast_val}")
                        # Use the file value as the starting guess
                        self.parameters.dict[item][0] = contrast_val
                        # In a normal prior, center at the file value
                        if (self.parameters.dict[item][-1] == 'N' and
                                self.recenter_spotcon_prior):
                            self.parameters.dict[item][-3] = contrast_val
                        # Update the non-dictionary form as well
                        setattr(self.parameters, item,
                                self.parameters.dict[item])

    def setup(self, newparams):
        """Setup a model for evaluation and fitting.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        """
        self.systems = []
        self.rps = []
        for chan in range(self.nchannel_fitted):
            if chan == 0:
                chankey = ''
            else:
                chankey = f'_ch{chan}'

            # Initialize PlanetParams object
            pl_params = PlanetParams(self, 0, chan, eval=False)

            if pl_params.nspots > 0:
                # Check for spots and set spot parameters if needed
                # create arrays to hold values
                spotrad = tt.zeros(0)
                spotlat = tt.zeros(0)
                spotlon = tt.zeros(0)
                spotcon = tt.zeros(0)

                for n in range(pl_params.nspots):
                    # read radii, latitudes, longitudes, and contrasts
                    if n > 0:
                        spot_id = f'{n}'
                    else:
                        spot_id = ''
                    spotrad = tt.concatenate([
                        spotrad, [getattr(pl_params, f'spotrad{spot_id}'),]])
                    spotlat = tt.concatenate([
                        spotlat, [getattr(pl_params, f'spotlat{spot_id}'),]])
                    spotlon = tt.concatenate([
                        spotlon, [getattr(pl_params, f'spotlon{spot_id}'),]])
                    spotcon = tt.concatenate([
                        spotcon, [getattr(pl_params, f'spotcon{spot_id}'),]])

                # Apply some conversions since inputs are in fleck units
                spotrad *= 90
                spotcon = 1-spotcon

                if pl_params.spotnpts is None:
                    # Have a default spotnpts for starry
                    pl_params.spotnpts = 30

                # Initialize map object and add spots
                map = starry.Map(ydeg=pl_params.spotnpts, udeg=self.udeg,
                                 inc=pl_params.spotstari)
                for n in range(pl_params.nspots):
                    map.spot(contrast=spotcon[n], radius=spotrad[n],
                             lat=spotlat[n], lon=spotlon[n])

                # Initialize star object
                star = starry.Primary(map, m=0, r=pl_params.Rs,
                                      prot=pl_params.spotrot)
            else:
                # Initialize star object without any spots
                star = starry.Primary(starry.Map(udeg=self.udeg),
                                      m=0, r=pl_params.Rs)

            if pl_params.limb_dark == 'quadratic':
                # PlanetParams takes care of doing kipping2013->quadratic
                star.map[1] = pl_params.u1
                star.map[2] = pl_params.u2
            elif pl_params.limb_dark == 'linear':
                star.map[1] = pl_params.u1
            elif pl_params.limb_dark != 'uniform':
                message = (f'ERROR: Our StarryModel is not yet able to '
                           f'handle {self.parameters.limb_dark.value} '
                           f'limb darkening.\n'
                           f'       limb_dark must be one of uniform, '
                           f'linear, quadratic, or kipping2013.')
                raise ValueError(message)

            # Setup each planet
            planets = []
            for pid in range(self.num_planets):
                # Initialize PlanetParams object for this planet
                pl_params = PlanetParams(self, pid, chan, eval=False)

                # Pixel sampling setup
                if self.pixelsampling:
                    planet_map = starry.Map(ydeg=pl_params.ydeg)
                    planet_map_temp = starry.Map(ydeg=pl_params.ydeg)

                    # Get pixel values
                    p = tt.zeros(0)
                    for pix in range(self.npix):
                        pixname = 'pixel'
                        if pix > 0:
                            pixname += f'{pix}'
                        p = tt.concatenate([p, [getattr(pl_params, pixname),]])

                    # Get pixel transform matrix
                    P2Y = planet_map.get_pixel_transforms(
                        oversample=self.oversample)[3]

                    # Transform pixels to spherical harmonics
                    ylms = tt.dot(P2Y, p)
                    planet_map_temp.amp = ylms[0]
                    planet_map_temp[1:, :] = ylms[1:]/ylms[0]

                    amp = pl_params.fp/tt.abs_(
                        planet_map_temp.flux(theta=0)[0])
                    planet_map.amp = amp*ylms[0]
                    planet_map[1:, :] = ylms[1:]/ylms[0]

                    # Store the fp, Ylm, and map for convenient access later
                    if f'fp{chankey}' not in self.freenames:
                        setattr(self.model, f'fp{chankey}', pm.Deterministic(
                            f'fp{chankey}', pl_params.fp))
                    for ell in range(1, pl_params.ydeg+1):
                        for m in range(-ell, ell+1):
                            setattr(self.model, f'Y{ell}{m}{chankey}',
                                    pm.Deterministic(f'Y{ell}{m}{chankey}',
                                                     planet_map[ell, m]))
                    setattr(self.model, f'map{chankey}', pm.Deterministic(
                        f'map{chankey}',
                        planet_map.render(projection="rect", res=100)))
                elif not hasattr(pl_params, 'fp'):
                    planet_map = starry.Map(ydeg=pl_params.ydeg, amp=0)
                else:
                    planet_map = starry.Map(ydeg=pl_params.ydeg)
                    planet_map_temp = starry.Map(ydeg=pl_params.ydeg)
                    for ell in range(1, pl_params.ydeg+1):
                        for m in range(-ell, ell+1):
                            if hasattr(pl_params, f'Y{ell}{m}'):
                                planet_map[ell, m] = getattr(pl_params,
                                                             f'Y{ell}{m}')
                                planet_map_temp[ell, m] = getattr(pl_params,
                                                                  f'Y{ell}{m}')
                    amp = pl_params.fp/tt.abs_(
                        planet_map_temp.flux(theta=0)[0])
                    planet_map.amp = amp
                self.rps.append(pl_params.rp)

                # Solve Keplerian orbital period equation for system mass
                # (otherwise starry is going to mess with P or a...)
                a = pl_params.a*pl_params.Rs*const.R_sun.value
                p = pl_params.per*(24*3600)
                Mp = ((2*np.pi*a**(3/2))/p)**2/const.G.value/const.M_sun.value

                # Initialize planet object
                planet = starry.Secondary(
                    planet_map,
                    m=Mp,
                    # Convert radius ratio to R_star units
                    r=tt.abs_(pl_params.rp)*pl_params.Rs,
                    a=pl_params.a,
                    # Another option to set inclination using impact parameter
                    # inc=tt.arccos(b/a)*180/np.pi
                    inc=pl_params.inc,
                    ecc=pl_params.ecc,
                    w=pl_params.w
                )
                planet.porb = pl_params.per
                planet.prot = pl_params.per
                planet.theta0 = 180.0
                planet.t0 = pl_params.t0

                planets.append(planet)

            # Instantiate the system
            system = starry.System(star, *planets,
                                   light_delay=self.compute_ltt)
            self.systems.append(system)

        self.update(newparams)

    def eval(self, eval=True, channel=None, piecewise=False, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        piecewise : bool; optional
            If True, return the lightcurve from each object, otherwise return
            the lightcurve of the entire system.
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

        # Currently can't separately evaluate starry models
        # (given mutual occultations)
        pid_iter = range(self.num_planets)

        if eval:
            lib = np.ma
            systems = self.fit.systems
        else:
            lib = tt
            systems = self.systems

        if piecewise:
            returnVal = []
        else:
            returnVal = lib.zeros(0)
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            temp_system = systems[chan]
            if self.mutualOccultations:
                result = temp_system.flux(time, total=False)
                fstar = result.pop(0)
                fplanets = result
            else:
                fstar = lib.ones(len(time))
                fplanets = []
                for pid in pid_iter:
                    simple_system = starry.System(
                        temp_system.primary, temp_system.secondaries[pid],
                        light_delay=temp_system.light_delay)
                    transit, eclipse = simple_system.flux(time, total=False)
                    fstar *= transit
                    fplanets.append(eclipse)

            if eval:
                fstar = fstar.eval()
                fplanets_eval = []
                for fplanet in fplanets:
                    fplanets_eval.append(fplanet.eval())
                fplanets = fplanets_eval

            if hasattr(self.parameters, 'spotrad'):
                # Re-normalize to avoid degenaricies with c0
                fstar = fstar/fstar[0]

            result = [fstar, *fplanets]

            if piecewise:
                # Return each body's lightcurve separately
                returnVal.append(result)
            else:
                # Return the system lightcurve
                lcpiece = lib.zeros(len(time))
                for piece in result:
                    lcpiece += piece
                returnVal = lib.concatenate([returnVal, lcpiece])

        return returnVal

    def compute_fp(self, pid=0, theta=0):
        """Compute the planetary flux at an arbitrary orbital position.

        Parameters
        ----------
        pid : int; optional
            Planet ID, default is 0.
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
                planet_map = system.secondaries[pid].map
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
        for chan in range(self.nchannel_fitted):
            if chan == 0:
                chankey = ''
            else:
                chankey = f'_ch{chan}'

            # Initialize PlanetParams object
            pl_params = PlanetParams(self, 0, chan, eval=True)

            if pl_params.nspots > 0:
                # Check for spots and set spot parameters if needed
                # create arrays to hold values
                spotrad = np.zeros(0)
                spotlat = np.zeros(0)
                spotlon = np.zeros(0)
                spotcon = np.zeros(0)

                for n in range(pl_params.nspots):
                    # read radii, latitudes, longitudes, and contrasts
                    if n > 0:
                        spot_id = f'{n}'
                    else:
                        spot_id = ''
                    spotrad = np.concatenate([
                        spotrad, [getattr(pl_params, f'spotrad{spot_id}'),]])
                    spotlat = np.concatenate([
                        spotlat, [getattr(pl_params, f'spotlat{spot_id}'),]])
                    spotlon = np.concatenate([
                        spotlon, [getattr(pl_params, f'spotlon{spot_id}'),]])
                    spotcon = np.concatenate([
                        spotcon, [getattr(pl_params, f'spotcon{spot_id}'),]])

                # Apply some conversions since inputs are in fleck units
                spotrad *= 90
                spotcon = 1-spotcon

                if pl_params.spotnpts is None:
                    # Have a default spotnpts for starry
                    pl_params.spotnpts = 30

                # Initialize map object and add spots
                map = starry.Map(ydeg=pl_params.spotnpts, udeg=self.udeg,
                                 inc=pl_params.spotstari)
                for n in range(pl_params.nspots):
                    map.spot(contrast=spotcon[n], radius=spotrad[n],
                             lat=spotlat[n], lon=spotlon[n])

                # Initialize star object
                star = starry.Primary(map, m=0, r=pl_params.Rs,
                                      prot=pl_params.spotrot)
            else:
                # Initialize star object without any spots
                star = starry.Primary(starry.Map(udeg=self.udeg),
                                      m=0, r=pl_params.Rs)

            if pl_params.limb_dark == 'quadratic':
                # PlanetParams takes care of doing kipping2013->quadratic
                star.map[1:] = pl_params.u
            elif pl_params.limb_dark == 'linear':
                star.map[1] = pl_params.u1
            elif pl_params.limb_dark != 'uniform':
                message = (f'ERROR: Our StarryModel is not yet able to '
                           f'handle {self.parameters.limb_dark.value} '
                           f'limb darkening.\n'
                           f'       limb_dark must be one of uniform, '
                           f'linear, quadratic, or kipping2013.')
                raise ValueError(message)

            # Setup each planet
            planets = []
            for pid in range(self.num_planets):
                # Initialize PlanetParams object for this planet
                pl_params = PlanetParams(self, pid, chan, eval=True)

                # Pixel sampling setup
                if self.pixelsampling:
                    planet_map = starry.Map(ydeg=pl_params.ydeg)
                    planet_map_temp = starry.Map(ydeg=pl_params.ydeg)

                    # Get pixel values
                    p = np.zeros(0)
                    for pix in range(self.npix):
                        pixname = 'pixel'
                        if pix > 0:
                            pixname += f'{pix}'
                        p = np.concatenate([p, [getattr(pl_params, pixname),]])

                    # Get pixel transform matrix
                    P2Y = planet_map.get_pixel_transforms(
                        oversample=self.oversample)[3]

                    # Transform pixels to spherical harmonics
                    ylms = np.dot(P2Y, p)
                    planet_map_temp.amp = ylms[0]
                    planet_map_temp[1:, :] = ylms[1:]/ylms[0]

                    amp = pl_params.fp/np.abs(planet_map_temp.flux(theta=0)[0])
                    planet_map.amp = amp*ylms[0]
                    planet_map[1:, :] = ylms[1:]/ylms[0]

                    # Store the fp, Ylm, and map for convenient access later
                    if f'fp{chankey}' not in self.freenames:
                        setattr(self.fit, f'fp{chankey}', pl_params.fp)
                    for ell in range(1, pl_params.ydeg+1):
                        for m in range(-ell, ell+1):
                            setattr(self.fit, f'Y{ell}{m}{chankey}',
                                    planet_map[ell, m])
                    setattr(self.fit, f'map{chankey}',
                            planet_map.render(projection="rect",
                                              res=100).eval())
                elif not hasattr(pl_params, 'fp'):
                    planet_map = starry.Map(ydeg=pl_params.ydeg, amp=0)
                else:
                    planet_map = starry.Map(ydeg=pl_params.ydeg)
                    planet_map_temp = starry.Map(ydeg=pl_params.ydeg)
                    for ell in range(1, pl_params.ydeg+1):
                        for m in range(-ell, ell+1):
                            if hasattr(pl_params, f'Y{ell}{m}'):
                                planet_map[ell, m] = getattr(pl_params,
                                                             f'Y{ell}{m}')
                                planet_map_temp[ell, m] = getattr(pl_params,
                                                                  f'Y{ell}{m}')
                    amp = pl_params.fp/np.abs(planet_map_temp.flux(theta=0)[0])
                    planet_map.amp = amp
                self.fit_rps.append(pl_params.rp)

                # Solve Keplerian orbital period equation for system mass
                # (otherwise starry is going to mess with P or a...)
                a = pl_params.a*pl_params.Rs*const.R_sun.value
                p = pl_params.per*(24*3600)
                Mp = ((2*np.pi*a**(3/2))/p)**2/const.G.value/const.M_sun.value

                # Initialize planet object
                planet = starry.Secondary(
                    planet_map,
                    m=Mp,
                    # Convert radius ratio to R_star units
                    r=np.abs(pl_params.rp)*pl_params.Rs,
                    a=pl_params.a,
                    # Another option to set inclination using impact parameter
                    # inc=tt.arccos(b/a)*180/np.pi
                    inc=pl_params.inc,
                    ecc=pl_params.ecc,
                    w=pl_params.w
                )
                planet.porb = pl_params.per
                planet.prot = pl_params.per
                planet.theta0 = 180.0
                planet.t0 = pl_params.t0

                planets.append(planet)

            # Instantiate the system
            system = starry.System(star, *planets,
                                   light_delay=self.compute_ltt)
            self.fit.systems.append(system)
