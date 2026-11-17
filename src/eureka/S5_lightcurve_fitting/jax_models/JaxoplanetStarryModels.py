import inspect
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np
from astropy import constants as const
from jax.typing import ArrayLike
from jaxoplanet.orbits.keplerian import Central
from jaxoplanet.starry.core.rotation import dot_rotation_matrix
from jaxoplanet.starry.light_curves import light_curve, surface_light_curve
from jaxoplanet.starry.orbit import SurfaceBody, SurfaceSystem
from jaxoplanet.starry.surface import Surface
from jaxoplanet.starry.ylm import Ylm

from ...lib.split_channels import split
from ..limb_darkening_fit import ld_profile
from . import JaxModel
from .AstroModel import compute_astroparams, eclipse_midpoint_jax

jnp_Rsun = jnp.array(const.R_sun.value)
jnp_Msun = jnp.array(const.M_sun.value)
jnp_G = jnp.array(const.G.value)


@lru_cache(maxsize=32)
def _spot_basis(ydeg, npts):
    """Build jaxoplanet's fixed Legendre projection using host arrays.

    Its upstream Bp routine mixes NumPy polynomials and traced JAX arrays.
    Keeping this configuration calculation on the host avoids both tracer
    conversion and tracer caching when a spotted model is compiled.
    """
    theta = np.linspace(0., np.pi, npts)
    ell = np.arange(ydeg + 1)
    basis = np.polynomial.legendre.legvander(np.cos(theta), ydeg)
    basis *= np.sqrt(2. * ell + 1.)
    projection = np.linalg.solve(
        basis.T @ basis + 1e-9 * np.eye(ydeg + 1), basis.T)
    smoothing = .5 if ydeg < 4 else 2. / ydeg
    indices = ell * (ell + 1)
    projection *= np.exp(-.5 * indices * smoothing**2)[:, None]
    return projection, theta, indices


def _get_channel_param(
    param_dict: Dict[str, Any],
    base_name: str,
    channel: int,
) -> Any:
    '''Helper to get a possibly channel-specific parameter from param_dict.

    Tries ``f'{base_name}_ch{channel}'`` first, then falls back to
    ``base_name``. Returns ``None`` if neither is present.
    '''
    if channel != 0:
        ch_key = f'{base_name}_ch{channel}'
        if ch_key in param_dict:
            return param_dict[ch_key]
    if base_name in param_dict:
        return param_dict[base_name]
    return None


def build_stellar_surface(
    astro_star: Dict[str, Any],
    param_dict: Dict[str, Any],
    spot_bases: List[str],
    ydeg: int,
    default_spotnpts: int,
    default_spotfac: int,
    channel: int,
) -> Surface:
    '''Construct the stellar Surface (with optional spots) for starry.

    Parameters
    ----------
    astro_star : dict
        Output of :func:`compute_astroparams` for pid=0 and the given channel.
    param_dict : dict
        Full parameter dictionary for this fit.
    spot_bases : list of str
        Base keys for spots (e.g. ['spotrad', 'spotrad1', ...]) with no
        channel suffix.
    ydeg : int
        Maximum spherical harmonic degree for the star map.
    default_spotnpts : int
        Default number of quadrature points for spots (if not set in params).
    default_spotfac : int
        Default spot_fac parameter for :func:`ylm_spot`.
    channel : int
        Channel index.

    Returns
    -------
    Surface
        The stellar surface with or without spots.
    '''
    # Collect per-spot parameters from param_dict
    spot_rads: List[jnp.ndarray] = []
    spot_lats: List[jnp.ndarray] = []
    spot_lons: List[jnp.ndarray] = []
    spot_cons: List[jnp.ndarray] = []

    # Global / default contrast for this channel, if present
    default_con = astro_star.get('spotcon')
    if default_con is None:
        default_con = _get_channel_param(param_dict, 'spotcon', channel)

    for base in spot_bases:
        # base is something like 'spotrad', 'spotrad1', 'spotrad2', ...
        suffix = base[len('spotrad'):]
        rad = _get_channel_param(param_dict, f'spotrad{suffix}', channel)
        lat = _get_channel_param(param_dict, f'spotlat{suffix}', channel)
        lon = _get_channel_param(param_dict, f'spotlon{suffix}', channel)
        con = _get_channel_param(param_dict, f'spotcon{suffix}', channel)

        if con is None:
            con = default_con

        if rad is None or lat is None or lon is None or con is None:
            continue

        spot_rads.append(jnp.asarray(rad))
        spot_lats.append(jnp.asarray(lat))
        spot_lons.append(jnp.asarray(lon))
        spot_cons.append(jnp.asarray(con))

    nspots = len(spot_rads)

    if nspots == 0:
        # No spots: simple limb-darkened disk
        return Surface(u=astro_star['u'])

    # Convert lists to arrays and apply fleck -> starry conversions
    spotrad_arr = jnp.arcsin(jnp.stack(spot_rads))
    spotlat_arr = jnp.stack(spot_lats) * jnp.pi / 180.
    spotlon_arr = jnp.stack(spot_lons) * jnp.pi / 180.
    # fleck defines contrast as (1 - spot_contrast); starry uses 1 - contrast
    spotcon_arr = 1. - jnp.stack(spot_cons)

    spotnpts = astro_star.get('spotnpts') or default_spotnpts
    spotfac = astro_star.get('spotfac', default_spotfac)

    projection, theta, indices = _spot_basis(int(ydeg), int(spotnpts))
    # Each expansion includes a uniform background, so add only its deficit
    # to the shared background.
    coefficients = jnp.zeros((ydeg + 1)**2).at[0].set(1.)
    for i in range(nspots):
        profile = -jax.nn.sigmoid(
            -spotfac * (jnp.asarray(theta) - spotrad_arr[i]))
        spot = jnp.zeros((ydeg + 1)**2).at[indices].set(
            jnp.asarray(projection) @ profile * spotcon_arr[i])
        spot = dot_rotation_matrix(ydeg, 1., 0., 0., spotlat_arr[i])(spot)
        spot = dot_rotation_matrix(ydeg, 0., 1., 0., -spotlon_arr[i])(spot)
        coefficients = coefficients + spot
    ylm_star = Ylm.from_dense(coefficients, normalize=False)

    inc_deg = astro_star.get('spotstari', astro_star.get('inc', 90.))
    obl_deg = astro_star.get('spotstarobl', 0.)
    period = astro_star.get('spotrot', 1e12)

    star_surface = Surface(
        inc=inc_deg * jnp.pi / 180.,
        obl=obl_deg * jnp.pi / 180.,
        period=period,
        u=astro_star['u'],
        y=ylm_star,
    )
    return star_surface


def build_planet_body(
    astro: Dict[str, Any],
) -> SurfaceBody:
    '''Construct a SurfaceBody for a planet (map + orbit) for starry.

    Parameters
    ----------
    astro : dict
        Output of :func:`compute_astroparams` for the given planet and channel.

    Returns
    -------
    SurfaceBody
        The starry surface + Keplerian orbit for this planet.
    '''
    per = astro["per"]
    t_ecl = eclipse_midpoint_jax(astro)

    # Choose map phase so that the rotational phase at mid-eclipse is 0:
    #   phi(t) = 2*pi*t / P + phase0
    #   phi(t_ecl) = 0  -> phase0 = -2*pi*t_ecl/P
    phase0 = -2. * jnp.pi * (t_ecl / per)

    # Gather any Y_lm coefficients stored in astro as Y{ell}{m}[_ch#]
    planet_Ylm_dict: Dict[tuple[int, int], jnp.ndarray] = {
        (0, 0): jnp.array(1.)}
    planet_Ylm_temp: Dict[tuple[int, int], jnp.ndarray] = {
        (0, 0): jnp.array(1.)}

    for key, val in astro.items():
        match = re.fullmatch(r'Y([0-9]+)(-?[0-9]+)(?:_ch[0-9]+)?', key)
        if match:
            ell = int(match.group(1))
            mm = int(match.group(2))
            planet_Ylm_dict[(ell, mm)] = jnp.asarray(val)
            planet_Ylm_temp[(ell, mm)] = jnp.asarray(val)

    # Normalize Y_00 so that the phase-curve amplitude matches fp
    planet_Ylm_temp_obj = Ylm(planet_Ylm_temp)
    planet_surface_temp = Surface(y=planet_Ylm_temp_obj,
                                  period=per, phase=phase0)
    scale = astro['fp'] / surface_light_curve(planet_surface_temp,
                                              theta=0.)

    planet_Ylm_obj = Ylm(planet_Ylm_dict)
    planet_surface = Surface(y=planet_Ylm_obj, period=per, phase=phase0,
                             amplitude=scale, normalize=False)

    # Solve Keplerian equation for the system mass, as in the transit model
    a_m = astro['a'] * astro['Rs'] * jnp_Rsun
    p_s = astro['per'] * (24. * 3600.)
    Mp = ((2. * jnp.pi * a_m ** (3. / 2.)) / p_s) ** 2
    Mp /= jnp_G * jnp_Msun

    planet = SurfaceBody(
        surface=planet_surface,
        mass=Mp,
        # Convert Rp/Rs to R_star units (allow negative rp via abs)
        radius=jnp.abs(astro['rp'] * astro['Rs']),
        # Convert a/Rs to R_star units
        semimajor=astro['a'] * astro['Rs'],
        inclination=astro['inc_rad'],
        time_transit=astro['t0'],
        eccentricity=astro['ecc'],
        omega_peri=astro['w_rad'],
    )
    return planet


def evaluate_starry_model_jax(
    time_global: ArrayLike,
    nints: List[int],
    multwhite: bool,
    fitted_channels: List[int],
    num_planets: int,
    param_dict: Dict[str, Any],
    ydeg: int,
    spot_bases: List[str],
    default_spotnpts: int,
    default_spotfac: int,
    piecewise: bool = False,
) -> Any:
    '''Evaluate the starry-based stellar+planetary model in a JAX-pure way.

    This mirrors :meth:`JaxoplanetStarryModel.eval` but takes all required
    inputs explicitly, making it suitable to call inside
    :func:`evaluate_astrophysical_model_jax`.

    Parameters
    ----------
    time_global : array-like
        Full time array for the observation.
    nints : list[int]
        Number of integrations per channel.
    multwhite : bool
        Whether this is a multwhite fit.
    fitted_channels : list[int]
        Channel indices to evaluate.
    num_planets : int
        Number of planets in the system.
    param_dict : dict
        Flat parameter dictionary.
    ydeg : int
        Spherical harmonic degree for the stellar/planet maps.
    spot_bases : list of str
        Base names for spot parameters (e.g. ['spotrad', 'spotrad1', ...]).
    default_spotnpts : int
        Default quadrature points for spots if not specified.
    default_spotfac : int
        Default spot_fac factor if not specified.
    piecewise : bool; optional
        If True, return per-body light curves per channel; otherwise return
        a single concatenated system light curve.

    Returns
    -------
    list or jnp.ndarray
        If ``piecewise=True``, returns a list of length
        ``len(fitted_channels)``, each element being
        ``[f_star, f_planet0, f_planet1, ...]`` for that
        channel. If ``piecewise=False``, returns the total system flux
        across all selected channels as a 1D array.
    '''
    time_global = jnp.array(time_global)
    nchan = len(fitted_channels)

    # Total length across selected channels
    if multwhite:
        total_len = sum(nints[chan] for chan in fitted_channels)
    else:
        total_len = time_global.size * nchan

    if piecewise:
        per_channel_results: List[List[jnp.ndarray]] = []
    else:
        lcfinal = jnp.zeros(total_len)
        offset = 0

    for chan in fitted_channels:
        # Per-channel time array
        if multwhite:
            time = split([time_global], nints, chan)[0]
        else:
            time = time_global

        # Star astroparams and surface
        astro_star = compute_astroparams(param_dict, channel=chan, pid=0)
        star_surface = build_stellar_surface(
            astro_star=astro_star,
            param_dict=param_dict,
            spot_bases=spot_bases,
            ydeg=ydeg,
            default_spotnpts=default_spotnpts,
            default_spotfac=default_spotfac,
            channel=chan,
        )
        star = Central(radius=astro_star['Rs'], mass=0.)
        system = SurfaceSystem(star, star_surface)

        # Add planets
        for pid in range(num_planets):
            astro = compute_astroparams(param_dict, channel=chan, pid=pid)
            planet = build_planet_body(astro)
            system = system.add_body(planet)

        # Compute the light curves: first row is star, rest are planets
        result = light_curve(system)(time).T
        fstar = result[0]
        fplanets = result[1:]

        if piecewise:
            components: List[jnp.ndarray] = [fstar, *fplanets]
            per_channel_results.append(components)
        else:
            lcpiece = fstar
            lcpiece = lcpiece + jnp.sum(fplanets, axis=0)
            lcfinal = lcfinal.at[offset:(offset + len(lcpiece))].set(lcpiece)
            offset += len(lcpiece)

    if piecewise:
        return per_channel_results
    return lcfinal


class JaxoplanetStarryModel(JaxModel):
    '''Transit+Eclipse+PhaseCurve model using jaxoplanet.starry.'''

    def __init__(self, **kwargs: Any) -> None:
        '''Initialize the model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            ``eureka.S5_lightcurve_fitting.models.Model.__init__()``.
            Can pass in ``parameters``, ``longparamlist``, ``nchan``,
            and ``paramtitles`` here.
        '''
        super().__init__(**kwargs)
        self.name = 'starry'
        self.modeltype = 'physical'

        if getattr(self, 'pixelsampling', False):
            raise NotImplementedError(
                'Pixel sampling has not been ported to jaxoplanet. '
                'Fit spherical harmonic coefficients instead.')
        if getattr(self, 'compute_ltt', False):
            raise NotImplementedError(
                'Light-travel-time correction is not implemented for starry. '
                'Set compute_ltt=False for this model.')
        if (self.num_planets > 1 and
                getattr(self, 'mutualOccultations', False)):
            raise NotImplementedError(
                'jaxoplanet does not model mutual planetary occultations. '
                'Set mutualOccultations=False to omit these events.')

        log = kwargs.get('log')

        required = ['Rs']
        missing = [name not in self.paramtitles for name in required]
        if any(missing):
            absent = [r for r, m in zip(required, missing) if m]
            message = (
                f"Missing required params {absent} in your EPF. "
                "Make sure it is not set to 'independent' as this is no "
                "longer a supported option; you can set these parameters to "
                "fixed if you want to maintain the old 'independent' "
                "behavior."
            )
            raise AssertionError(message)

        # Store the ld_profile
        self.ld_from_S4 = kwargs.get('ld_from_S4')
        ld_name = self.parameters.limb_dark.value
        ld_func = ld_profile(ld_name, use_gen_ld=self.ld_from_S4)
        len_params = len(inspect.signature(ld_func).parameters)
        self.coeffs = [f'u{n}' for n in range(1, len_params)]

        self.ld_from_file = kwargs.get('ld_from_file')

        if ld_name not in ['uniform', 'linear', 'quadratic', 'kipping2013']:
            message = (
                "ERROR: Our JaxoplanetStarryModel is not yet able to "
                f"handle '{ld_name}' limb darkening.\n"
                "       limb_dark must be one of uniform, linear, "
                "quadratic, or kipping2013."
            )
            raise ValueError(message)

        # Replace u parameters with generated limb-darkening values
        if self.ld_from_S4 or self.ld_from_file:
            log.writelog('Using the following limb-darkening values:')
            self.ld_array = kwargs.get('ld_coeffs')
            for c in range(self.nchannel_fitted):
                chan = self.fitted_channels[c]
                if self.ld_from_S4:
                    ld_array = self.ld_array[len_params - 2]
                else:
                    ld_array = self.ld_array
                for u in self.coeffs:
                    index = np.where(np.array(self.paramtitles) == u)[0]
                    if len(index) != 0:
                        item = self.longparamlist[c][index[0]]
                        param = int(item.split('_')[0][-1])
                        ld_val = ld_array[chan][param - 1]
                        log.writelog(f'{item}: {ld_val}')
                        # Use the file value as the starting guess
                        self.parameters.dict[item][0] = ld_val
                        # In a normal prior, center at the file value
                        if (
                            self.parameters.dict[item][-1] == 'N'
                            and self.recenter_ld_prior
                        ):
                            self.parameters.dict[item][-3] = ld_val
                        # Update the non-dictionary form as well
                        setattr(self.parameters, item,
                                self.parameters.dict[item])

        # Optional spot contrast file handling (as before)
        self.spotcon_file = getattr(self, 'spotcon_file', None)
        self.recenter_spotcon_prior = getattr(
            self, 'recenter_spotcon_prior', True
        )

        if self.spotcon_file:
            try:
                spot_coeffs = np.genfromtxt(self.spotcon_file)
            except FileNotFoundError as exc:
                raise Exception(
                    f'The spot contrast file {self.spotcon_file} '
                    'could not be found.'
                ) from exc

            nspots = len(
                [
                    s
                    for s in self.parameters.dict.keys()
                    if 'spotrad' in s and '_' not in s
                ]
            )

            if len(spot_coeffs.shape) == 1:
                spot_coeffs = np.repeat(
                    spot_coeffs[np.newaxis, :], nspots, axis=0
                )

            log.writelog('Using the following spot contrast values:')
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
                        log.writelog(f'{item}: {contrast_val}')
                        self.parameters.dict[item][0] = contrast_val
                        if (
                            self.parameters.dict[item][-1] == 'N'
                            and self.recenter_spotcon_prior
                        ):
                            self.parameters.dict[item][-3] = contrast_val
                        setattr(self.parameters, item,
                                self.parameters.dict[item])

        # Precompute spot base names (no channel suffix), e.g. spotrad,
        # spotrad1, etc.
        self.spot_bases = sorted({
            name.split('_')[0] for name in self.parameters.dict
            if name.split('_')[0].startswith('spotrad')
        })

        # Degree of spherical harmonics for star/planet maps
        self.ydeg = getattr(self, 'ydeg', None)
        if self.ydeg is None:
            meta = kwargs.get('meta')
            if meta is not None and hasattr(meta, 'ydeg'):
                self.ydeg = meta.ydeg
        if self.ydeg is None:
            if log is not None:
                log.writelog(
                    'WARNING: ydeg not found for JaxoplanetStarryModel; '
                    'defaulting to ydeg = 0.'
                )
            self.ydeg = 0

        # Defaults for spotnpts and spotfac if not passed in parameters
        self.default_spotnpts = getattr(self, 'spotnpts', 300)
        self.default_spotfac = getattr(self, 'spotfac', 300)

    def eval(
        self,
        param_dict: Optional[Dict[str, Any]] = None,
        channel: Optional[int] = None,
        piecewise: bool = False,
        **kwargs: Any,
    ) -> Any:
        '''Evaluate the starry model (star + planets).

        Parameters
        ----------
        param_dict : dict; optional
            If None, uses values from ``self.parameters`` (fitted mode).
        channel : int; optional
            If provided, only evaluate for the specified channel.
        piecewise : bool; optional
            If True, return each body's light curve separately (star first,
            then each planet) for each channel. If False, return the total
            system flux concatenated across channels.
        **kwargs : dict
            Reserved for future use.

        Returns
        -------
        list or jnp.ndarray
            See :func:`evaluate_starry_model_jax` for details.
        '''
        param_dict = self._prepare_param_dict(param_dict)

        if channel is None:
            fitted_channels = self.fitted_channels
        else:
            fitted_channels = [channel]

        return evaluate_starry_model_jax(
            time_global=kwargs.get('time', self.time),
            nints=kwargs.get('nints', self.nints),
            multwhite=self.multwhite,
            fitted_channels=fitted_channels,
            num_planets=self.num_planets,
            param_dict=param_dict,
            ydeg=self.ydeg,
            spot_bases=self.spot_bases,
            default_spotnpts=self.default_spotnpts,
            default_spotfac=self.default_spotfac,
            piecewise=piecewise,
        )
