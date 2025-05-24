import numpy as np
from astropy import constants as const
import inspect
import jax.numpy as jnp

from jaxoplanet.orbits.keplerian import Central
from jaxoplanet.starry.ylm import Ylm, ylm_spot
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.orbit import SurfaceSystem, SurfaceBody
from jaxoplanet.starry.light_curves import surface_light_curve, light_curve

from . import JaxModel
from .AstroModel import PlanetParams
from ..limb_darkening_fit import ld_profile
from ...lib.split_channels import split


class JaxoplanetStarryModel(JaxModel):
    """Transit+Eclipse+PhaseCurve Model"""
    def __init__(self, **kwargs):
        """Initialize the model

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
        self.name = 'starry'

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

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

        # Store the ld_profile
        self.ld_from_S4 = kwargs.get('ld_from_S4')
        ld_func = ld_profile(self.parameters.limb_dark.value,
                             use_gen_ld=self.ld_from_S4)
        len_params = len(inspect.signature(ld_func).parameters)
        self.coeffs = ['u{}'.format(n) for n in range(1, len_params)]

        self.ld_from_file = kwargs.get('ld_from_file')

        if self.parameters.limb_dark.value not in ['uniform', 'linear',
                                                   'quadratic', 'kipping2013']:
            message = (f'ERROR: Our JaxoplanetStarryModel is not yet able to '
                       f'handle "{self.parameters.limb_dark.value}" '
                       f'limb darkening.\n'
                       f'       limb_dark must be one of uniform, '
                       f'linear, quadratic, or kipping2013.')
            raise ValueError(message)

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
        lcfinal : ndarray
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
        else:
            lib = jnp

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

            # Initialize PlanetParams object for this system
            pl_params = PlanetParams(self, 0, chan, eval=eval)
            # Initialize star surface
            star_surface = get_stellar_surface(pl_params, chan, eval=eval)
            # Initialize star object
            star = Central(radius=pl_params.Rs, mass=0.)
            # Initialize the system
            system = SurfaceSystem(star, star_surface)

            # Setup each planet
            for pid in range(self.num_planets):
                # Initialize PlanetParams object for this planet
                pl_params = PlanetParams(self, pid, chan, eval=eval)
                # Add the planet to the system
                system = system.add_body(get_planet(pl_params, eval=eval))

            # Compute the lightcurve
            result = light_curve(system)(time).T
            # The first element is the star, the rest are the planets
            fstar = result[0]
            fplanets = result[1:]

            # We need to normalize the planet lightcurves by the amplitude
            # of the planet surface
            if self.num_planets > 0:
                fplanets_normalized = []
                for i in range(self.num_planets):
                    amp = system.body_surfaces[i].amplitude
                    fplanets_normalized.append(fplanets[i]*amp)
                fplanets = fplanets_normalized

            # Repack the lightcurves into a single list
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


def get_stellar_surface(pl_params, chan, eval=True):
    """Get the stellar surface for the starry model.

    Parameters
    ----------
    pl_params : PlanetParams
        The planet parameters object.
    chan : int
        The channel number.
    eval : bool; optional
        If true evaluate the model, otherwise simply compile the model.
        Defaults to True.

    Returns
    -------
    star_surface : Surface
        The stellar surface.
    """
    if eval:
        lib = np
    else:
        lib = jnp

    if pl_params.nspots > 0:
        # create arrays to hold values
        spotrad = lib.array([])
        spotlat = lib.array([])
        spotlon = lib.array([])
        spotcon = lib.array([])
        spotcon0 = getattr(pl_params, 'spotcon')
        for n in range(pl_params.nspots):
            # read radii, latitudes, longitudes, and contrasts
            if n > 0:
                spot_id = f'{n}'
            else:
                spot_id = ''
            spotrad = lib.concatenate([
                spotrad, [getattr(pl_params, f'spotrad{spot_id}'),]])
            spotlat = lib.concatenate([
                spotlat, [getattr(pl_params, f'spotlat{spot_id}'),]])
            spotlon = lib.concatenate([
                spotlon, [getattr(pl_params, f'spotlon{spot_id}'),]])
            # If spotcon# isn't set, default to spotcon (from ch0)
            spotcon = lib.concatenate([
                spotcon, [getattr(pl_params, f'spotcon{spot_id}', spotcon0),]])

        # Apply some conversions since inputs are in fleck units
        spotrad *= 90
        spotcon = 1-spotcon

        if pl_params.spotnpts is None:
            # Have a default spotnpts and spotfac for starry
            pl_params.spotnpts = 300  # Default taken from starry code
            pl_params.spotfac = 300  # Default taken from starry code

        ylm = ylm_spot(ydeg=pl_params.ydeg, npts=pl_params.spotnpts,
                       spot_fac=pl_params.spotfac)
        ylm_star = ylm(spotcon, spotrad, spotlat*np.pi/180,
                       spotlon*np.pi/180)

        # Initialize the star surface
        star_surface = Surface(
            inc=pl_params.spotstari*np.pi/180,
            obl=pl_params.spotstarobl*np.pi/180,
            period=pl_params.spotrot, u=pl_params.u, y=ylm_star)
    else:
        # Initialize star surface without any spots
        star_surface = Surface(u=pl_params.u)

    return star_surface


def get_planet(pl_params, eval=True):
    """Get the planet surface for the starry model.

    Parameters
    ----------
    pl_params : PlanetParams
        The planet parameters object.
    eval : bool; optional
        If true evaluate the model, otherwise simply compile the model.
        Defaults to True.

    Returns
    -------
    planet_surface : Surface
        The planet surface.
    """
    if eval:
        lib = np
    else:
        lib = jnp

    if not hasattr(pl_params, 'fp'):
        planet_surface = None
    else:
        # Load all the Ylm coefficients into a dictionary
        planet_Ylm = dict()
        planet_Ylm_temp = dict()
        planet_Ylm[(0, 0)] = 1.
        planet_Ylm_temp[(0, 0)] = 1.
        for ell in range(1, pl_params.ydeg+1):
            for m in range(-ell, ell+1):
                planet_Ylm[(ell, m)] = getattr(
                    pl_params, f'Y{ell}{m}', 0.)
                planet_Ylm_temp[(ell, m)] = getattr(
                    pl_params, f'Y{ell}{m}', 0.)
        # Compute the necessary Y00 coefficient for the planet to have the
        # correct eclipse depth
        planet_Ylm_temp = Ylm(planet_Ylm_temp)
        planet_surface_temp = Surface(y=planet_Ylm_temp)
        amp = pl_params.fp/surface_light_curve(
            planet_surface_temp, theta=0)
        planet_Ylm[(0, 0)] = amp
        planet_Ylm = Ylm(planet_Ylm)
        planet_surface = Surface(y=planet_Ylm)

    # Solve Keplerian orbital period equation for system mass
    # (otherwise jaxoplanet is going to mess with P or a...)
    a = pl_params.a*pl_params.Rs*const.R_sun.value
    p = pl_params.per*(24*3600)
    Mp = ((2*np.pi*a**(3/2))/p)**2/const.G.value/const.M_sun.value

    # Initialize planet object
    planet = SurfaceBody(
        surface=planet_surface,
        mass=Mp,
        # Convert Rp/Rs to R_star units
        # FINDME: This code currently doesn't support negative planet radii
        radius=lib.abs(pl_params.rp*pl_params.Rs),
        # Convert a/Rs to R_star units
        semimajor=pl_params.a*pl_params.Rs,
        inclination=pl_params.inc*np.pi/180,
        time_transit=pl_params.t0,
        eccentricity=pl_params.ecc,
        omega_peri=pl_params.w*np.pi/180,
    )

    return planet
