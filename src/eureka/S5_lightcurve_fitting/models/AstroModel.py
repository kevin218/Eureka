import numpy as np
import astropy.constants as const
from copy import copy
import inspect

from .Model import Model
from .KeplerOrbit import KeplerOrbit
from ..limb_darkening_fit import ld_profile
from ...lib.split_channels import split
from ...lib.util import resolve_param_key


class PlanetParams():
    """
    Define planet parameters.
    """
    def __init__(self, model, pid=0, channel=0, wl=0, eval=True):
        """
        Set attributes to PlanetParams object.

        Parameters
        ----------
        model : object
            The model.eval object that contains a dictionary of parameter names
            and their current values.
        pid : int; optional
            Planet ID, default is 0.
        channel : int, optional
            The channel number for multi-wavelength fits or mutli-white fits.
            Defaults to 0.
        wl : int, optional
            The wavelength channel for multi-wavelength fits. Can be used for
            shared parameters in multi-white fits which should be different
            between channels. Defaults to 0.
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
        """

        if eval:
            parameterObject = model.parameters
            lib = np
        else:
            # No other option is currently supported until jax is added
            raise NotImplementedError(
                'JAX support is not yet implemented. '
                'Please use eval=True to evaluate the model in numpy mode.')

        # Planet/Channel/Wavelength identifiers (ids remain for external uses)
        self.pid = pid
        self.pid_id = '' if pid == 0 else f'_pl{self.pid}'

        self.channel = channel
        self.channel_id = '' if channel == 0 else f'_ch{self.channel}'

        self.wl = wl
        self.wl_id = '' if wl == 0 else f'_wl{self.wl}'

        # Transit/eclipse parameters
        self.t0 = None
        self.rprs = None
        self.rp = None
        self.rprs2 = None
        self.rp2 = None
        self.phi = 90.
        self.inc = None
        self.b = None
        self.ars = None
        self.a = None
        self.per = None
        self.ecc = None
        self.w = None
        self.ecosw = None
        self.esinw = None
        self.fpfs = None
        self.fp = None
        self.t_secondary = None
        # Harmonica parameters
        self.a1 = 0.
        self.b1 = 0.
        self.a2 = 0.
        self.b2 = 0.
        self.a3 = 0.
        self.b3 = 0.
        # POET phase curve parameters
        self.cos1_amp = 0.
        self.cos1_off = 0.
        self.cos2_amp = 0.
        self.cos2_off = 0.
        # Sinusoidal phase curve parameters
        self.AmpCos1 = 0.
        self.AmpSin1 = 0.
        self.AmpCos2 = 0.
        self.AmpSin2 = 0.
        # Quasi-Lambertian parameters
        self.quasi_gamma = 0.
        self.quasi_offset = 0.
        # Limb-darkening parameters
        self.u1 = 0.
        self.u2 = 0.
        self.u3 = 0.
        self.u4 = 0.
        # Fleck/starry star-spot parameters
        # Figure out how many star spots
        self.nspots = len([s for s in model.parameters.dict.keys()
                           if 'spotrad' in s and '_' not in s])
        self.spotrad = 0.
        self.spotlat = 0.
        self.spotlon = 0.
        self.spotcon = 0.
        for n in range(1, self.nspots):
            # read radii, latitudes, longitudes, and contrasts
            spot_id = f'{n}'
            setattr(self, f'spotrad{spot_id}', 0.)
            setattr(self, f'spotlat{spot_id}', 0.)
            setattr(self, f'spotlon{spot_id}', 0.)
            setattr(self, f'spotcon{spot_id}', 0.)
        self.spotstari = 90.
        self.spotstarobl = 0.
        self.spotrot = None
        self.spotnpts = None

        # Figure out how many planet map pixels
        self.npix = len([s for s in model.parameters.dict.keys()
                         if 'pixel' in s and '_' not in s])
        for pix in range(self.npix):
            pixname = 'pixel' if pix == 0 else f'pixel{pix}'
            setattr(self, pixname, 0.)

        # Figure out how many planet Ylm spherical harmonics
        ylm_params = [i for i, par in enumerate(model.parameters.dict.keys())
                      if par.startswith('Y') and par[1:].isnumeric()]
        if len(ylm_params) > 0:
            l_vals = [int(list(model.parameters.dict.keys())[ind][1])
                      for ind in ylm_params]
            self.ydeg = max(l_vals)
            for ell in range(1, self.ydeg+1):
                for m in range(-ell, ell+1):
                    setattr(self, f'Y{ell}{m}', 0.)
        else:
            self.ydeg = 0

        # ----- Local helper for wl>ch>base resolution ------------------------
        def _get_param(base, pid_override=None):
            """Return value for base at (pid, channel, wl) or None."""
            pid_use = self.pid if pid_override is None else pid_override
            key = resolve_param_key(
                base, model.parameters, pid=pid_use,
                channel=self.channel, wl=self.wl
            )
            if key in model.parameters.dict:
                val = getattr(parameterObject, key)
                return val.value if eval else val
            return None
        # ---------------------------------------------------------------------

        # Load in values using resolver; keep defaults if not found
        for item in list(self.__dict__.keys()):
            item_base = item
            try:
                val = _get_param(item_base, pid_override=self.pid)
                if val is None:
                    if (item in [f'u{i}' for i in range(1, 5)] or
                            item.startswith('spot')):
                        val = _get_param(item_base, pid_override=0)
                if val is not None:
                    setattr(self, item, val)
            except (KeyError, AttributeError):
                # Item not in model.parameters, so leave it as default
                pass

        # Allow for rp or rprs
        if self.rprs is None:
            val = _get_param('rp', pid_override=self.pid)
            if val is not None:
                self.rprs = val
        if self.rp is None:
            val = _get_param('rprs', pid_override=self.pid)
            if val is not None:
                self.rp = val

        # Allow for rp2 or rprs2
        if self.rprs2 is None:
            val = _get_param('rp2', pid_override=self.pid)
            if val is not None:
                self.rprs2 = val
        if self.rp2 is None:
            val = _get_param('rprs2', pid_override=self.pid)
            if val is not None:
                self.rp2 = val

        # Allow for a or ars
        if self.ars is None:
            val = _get_param('a', pid_override=self.pid)
            if val is not None:
                self.ars = val
        if self.a is None:
            val = _get_param('ars', pid_override=self.pid)
            if val is not None:
                self.a = val

        # b from inc (only if inc and a exist; validate inc range)
        if self.b is None:
            inc_value = _get_param('inc', pid_override=self.pid)
            a_value = self.a
            if (inc_value is not None) and (a_value is not None):
                if not (0.0 <= inc_value <= 180.0):
                    raise AssertionError(
                        f"inc={inc_value}° is outside [0, 180]."
                    )
                self.b = a_value * lib.cos(inc_value * np.pi / 180.0)
        # inc from b (domain-safe; clamp only tiny roundoff)
        if self.inc is None:
            b_value = _get_param('b', pid_override=self.pid)
            a_value = self.a
            if (b_value is not None and a_value is not None
                    and a_value != 0):
                tol = 1e-12
                ratio = b_value / a_value
                if not lib.isfinite(ratio):
                    raise AssertionError('Invalid b/a: non-finite value.')
                if ratio < -1.0 - tol or ratio > 1.0 + tol:
                    raise AssertionError(
                        'Inconsistent geometry: b/a outside [-1, 1].'
                    )
                ratio = min(1.0, max(-1.0, ratio))
                self.inc = lib.arccos(ratio) * 180.0 / np.pi

        # Allow for (ecc, w) or (ecosw, esinw)
        if (self.ecosw is None) and self.ecc == 0:
            self.ecosw = 0.
            self.esinw = 0.
        elif (self.ecc is None) and self.ecosw == 0 and self.esinw == 0:
            self.ecc = 0.
            self.w = 180.
        if self.ecosw is None:
            ecc_val = _get_param('ecc', pid_override=self.pid)
            w_val = _get_param('w', pid_override=self.pid)
            if ecc_val is not None and w_val is not None:
                self.ecosw = ecc_val * lib.cos(w_val * np.pi / 180.0)
                self.esinw = ecc_val * lib.sin(w_val * np.pi / 180.0)
        if self.ecc is None:
            ecosw_val = _get_param('ecosw', pid_override=self.pid)
            esinw_val = _get_param('esinw', pid_override=self.pid)
            if ecosw_val is not None and esinw_val is not None:
                self.ecc = lib.sqrt(ecosw_val**2 + esinw_val**2)
                self.w = lib.arctan2(esinw_val, ecosw_val) * 180.0 / np.pi

        # Allow for fp or fpfs
        if self.fpfs is None:
            val = _get_param('fp', pid_override=self.pid)
            if val is not None:
                self.fpfs = val
        if self.fp is None:
            val = _get_param('fpfs', pid_override=self.pid)
            if val is not None:
                self.fp = val
        if self.fpfs is None:
            self.fpfs = 0.
        if self.fp is None:
            self.fp = 0.

        # Set stellar radius
        if 'Rs' in model.parameters.dict:
            ptype_Rs = model.parameters.dict['Rs'][1]
            if ptype_Rs == 'free':
                rs_key = resolve_param_key('Rs', model.parameters,
                                           channel=self.channel, wl=self.wl)
            else:
                rs_key = 'Rs'

            try:
                value = getattr(parameterObject, rs_key)
            except AttributeError:
                msg = ('Missing required parameter Rs in your EPF. '
                       'Make sure it is not set to "independent" as this is '
                       'no longer supported; set it to "fixed" to keep the '
                       'old "independent" behavior.')
                raise AssertionError(msg)
            if eval:
                value = value.value
            self.Rs = value

        # Nicely packaging limb-darkening coefficients
        if not hasattr(model.parameters, 'limb_dark'):
            self.limb_dark = 'uniform'
        else:
            self.limb_dark = model.parameters.limb_dark.value
        ld_func = ld_profile(self.limb_dark)
        len_params = len(inspect.signature(ld_func).parameters)
        coeffs = ['u{}'.format(n) for n in range(1, len_params)]
        self.u = [getattr(self, coeff) for coeff in coeffs]
        if self.limb_dark == '4-parameter':
            self.limb_dark = 'nonlinear'
        elif self.limb_dark == 'kipping2013':
            self.limb_dark = 'quadratic'
            if eval:
                self.u_original = copy(self.u)
                self.u1 = 2*lib.sqrt(self.u[0])*self.u[1]
                self.u2 = lib.sqrt(self.u[0])*(1-2*self.u[1])
                self.u = lib.array([self.u1, self.u2])
            else:
                self.u1 = 2*lib.sqrt(self.u1)*self.u2
                self.u2 = lib.sqrt(self.u1)*(1-2*self.u2)
                self.u = lib.array([self.u1, self.u2])

        # Nicely packaging Harmonica coefficients
        self.ab = lib.array([self.rp,
                             self.a1, self.b1,
                             self.a2, self.b2,
                             self.a3, self.b3])

        # Make sure (e, w, ecosw, and esinw) are all defined (assuming e=0)
        if self.ecc is None:
            self.ecc = 0.
            self.w = 180.
            self.ecosw = 0.
            self.esinw = 0.

        if self.spotrot is None:
            # spotrot will default to 10k years (important if t0 is not ~0)
            self.spotrot = 3650000.
            self.fleck_fast = True
        else:
            self.fleck_fast = False

        self.inc_rad = self.inc * np.pi / 180
        self.w_rad = self.w * np.pi / 180


class AstroModel(Model):
    """A model that combines all astrophysical components."""
    def __init__(self, components, **kwargs):
        """Initialize the phase curve model.

        Parameters
        ----------
        components : list
            A list of eureka.S5_lightcurve_fitting.models.Model which together
            comprise the astrophysical model.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from Model class
        super().__init__(components=components, **kwargs)
        self.name = 'astrophysical model'

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

    @property
    def components(self):
        """A getter for the flux."""
        return self._components

    @components.setter
    def components(self, components):
        """A setter for the flux

        Parameters
        ----------
        flux_array : sequence
            The flux array
        """
        self._components = components
        self.transit_model = None
        self.eclipse_model = None
        self.phasevariation_models = []
        self.stellar_models = []
        for component in self.components:
            if 'transit' in component.name.lower():
                self.transit_model = component
            elif 'eclipse' in component.name.lower():
                self.eclipse_model = component
            elif 'phase curve' in component.name.lower():
                self.phasevariation_models.append(component)
            else:
                self.stellar_models.append(component)

    @property
    def fit(self):
        """A getter for the fit object."""
        return self._fit

    @fit.setter
    def fit(self, fit):
        """A setter for the fit object.

        Parameters
        ----------
        fit : object
            The fit object
        """
        self._fit = fit
        for component in self.components:
            component.fit = fit

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

        pid_input = copy(pid)
        if pid_input is None:
            pid_iter = range(self.num_planets)
        else:
            pid_iter = [pid_input,]

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Set all parameters
        lcfinal = np.ma.zeros(0)
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            starFlux = np.ma.ones(len(time))
            for component in self.stellar_models:
                starFlux *= component.eval(channel=chan, eval=eval, **kwargs)
            if self.transit_model is not None:
                starFlux *= self.transit_model.eval(channel=chan,
                                                    pid=pid_input,
                                                    **kwargs)

            planetFluxes = np.ma.zeros(len(time))
            for pid in pid_iter:
                # Initial default value
                planetFlux = 0

                if self.eclipse_model is not None:
                    # User is fitting an eclipse model
                    planetFlux = self.eclipse_model.eval(channel=chan, pid=pid,
                                                         **kwargs)
                elif len(self.phasevariation_models) > 0:
                    # User is dealing with phase variations of a
                    # non-eclipsing object
                    planetFlux = np.ma.ones(len(time))

                for model in self.phasevariation_models:
                    planetFlux *= model.eval(channel=chan, pid=pid, **kwargs)

                planetFluxes += planetFlux

            lcfinal = np.ma.append(lcfinal, starFlux+planetFluxes)
        return lcfinal


def get_ecl_midpt(params, lib=np):
    """
    Return the time of secondary eclipse center.

    Parameters
    ----------
    params : object
        Contains the physical parameters for the transit model.
    lib : library; optional
        Either np (numpy) or jnp (jax.numpy), depending on whether the
        code is being run in numpy or jax mode. Defaults to np.

    Returns
    -------
    t_secondary : float
        The time of secondary eclipse.
    """
    # Start with primary transit
    TA = np.pi/2-params.w*np.pi/180
    E = 2*lib.arctan(lib.sqrt((1-params.ecc)/(1+params.ecc))
                     * lib.tan(TA/2))
    M = E-params.ecc*lib.sin(E)
    phase_tr = M/2/np.pi

    # Now do secondary eclipse
    TA = 3*np.pi/2-params.w*np.pi/180
    E = 2*lib.arctan(lib.sqrt((1-params.ecc)/(1+params.ecc))
                     * lib.tan(TA/2))
    M = E-params.ecc*lib.sin(E)
    phase_ecl = M/2/np.pi

    return params.t0+params.per*(phase_ecl-phase_tr)


def true_anomaly(model, t, lib=np, xtol=1e-10):
    """Convert time to true anomaly, numerically.

    Parameters
    ----------
    params : object
        Contains the physical parameters for the transit model.
    t : ndarray
        The time in days.
    lib : library; optional
        Either np (numpy) or jnp (jax.numpy), depending on whether the
        code is being run in numpy or jax mode. Defaults to np.
    xtol : float; optional
        tolarance on error in eccentric anomaly (calculated along the way).
        Defaults to 1e-10.

    Returns
    -------
    ndarray
        The true anomaly in radians.
    """
    return 2.*lib.arctan(lib.sqrt((1.+model.ecc)/(1.-model.ecc)) *
                         lib.tan(eccentric_anomaly(model, t, lib,
                                                   xtol=xtol)/2.))


def eccentric_anomaly(model, t, lib=np, xtol=1e-10):
    """Convert time to eccentric anomaly, numerically.

    Parameters
    ----------
    model : Model
        The model (which contains the orbital parameters).
    t : ndarray
        The time in days.
    lib : library; optional
        Either np (numpy) or jnp (jax.numpy), depending on whether the
        code is being run in numpy or jax mode. Defaults to np.
    xtol : float; optional
        tolarance on error in eccentric anomaly. Defaults to 1e-10.

    Returns
    -------
    ndarray
        The eccentric anomaly in radians.
    """
    ta_peri = np.pi/2.-model.w*np.pi/180.
    ea_peri = 2.*lib.arctan(lib.sqrt((1.-model.ecc)/(1.+model.ecc)) *
                            lib.tan(ta_peri/2.))
    ma_peri = ea_peri - model.ecc*np.sin(ea_peri)
    t_peri = (model.t0 - (ma_peri/(2.*np.pi)*model.per))

    if lib not in [np, np.ma]:
        t_peri = t_peri + model.per*(lib.lt(t_peri, 0))
    elif t_peri < 0:
        t_peri = t_peri + model.per

    M = ((t-t_peri) * 2.*np.pi/model.per) % (2.*np.pi)

    E = FSSI_Eccentric_Inverse(model, M, lib, xtol)

    return E


def FSSI_Eccentric_Inverse(model, M, lib=np, xtol=1e-10):
    """Convert mean anomaly to eccentric anomaly using FSSI algorithm.

    Algorithm based on that from Tommasini+2018.

    Parameters
    ----------
    model : Model
        The model (which contains the orbital parameters).
    M : ndarray
        The mean anomaly in radians.
    lib : library; optional
        Either np (numpy) or jnp (jax.numpy), depending on whether the
        code is being run in numpy or jax mode. Defaults to numpy.
    xtol : float; optional
        tolarance on error in eccentric anomaly. Defaults to 1e-10.

    Returns
    -------
    ndarray
        The eccentric anomaly in radians.
    """
    xtol = np.max([1e-15, xtol])
    nGrid = (xtol/100.)**(-1./4.)

    xGrid = lib.arange(0, 2.*np.pi, 2*np.pi/int(nGrid))

    def f(ea):
        return ea - model.ecc*lib.sin(ea)

    def fP(ea):
        return 1. - model.ecc*lib.cos(ea)

    return FSSI(M, xGrid, f, fP, lib)


def FSSI(Y, x, f, fP, lib=np):
    """Fast Switch and Spline Inversion method from Tommasini+2018.

    Parameters
    ----------
    Y : ndarray
        The f(x) values to invert.
    x : ndarray
        x values spanning the domain (more values for higher precision).
    f : callable
        The function f.
    fP : callable
        The first derivative of the function f with respect to x.
    lib : library; optional
        Either np (numpy) or jnp (jax.numpy), depending on whether the
        code is being run in numpy or jax mode. Defaults to np.

    Returns
    -------
    ndarray
        The numerical approximation of f^-(y).
    """
    y = f(x)
    d = 1./fP(x)

    x0 = x[:-1]
    x1 = x[1:]
    y0 = y[:-1]
    y1 = y[1:]
    d0 = d[:-1]
    d1 = d[1:]

    c0 = x0
    c1 = d0

    dx = x0 - x1
    dy = y0 - y1
    dy2 = dy*dy

    c2 = ((2.*d0 + d1)*dy - 3.*dx)/dy2
    c3 = ((d0 + d1)*dy - 2.*dx)/(dy2*dy)

    if lib in [np, np.ma]:
        j = np.searchsorted(y1, Y)
        # Protect against indexing beyond the size of the array
        j[j >= y1.size] = y1.size-1
    else:
        j = lib.extra_ops.searchsorted(y1, Y)
        # Protect against indexing beyond the size of the array
        # Theano tensors don't allow item assignment
        mask = lib.ge(j, y1.size)
        j = j*(1-mask) + (y1.size-1)*mask

    P1 = Y - y0[j]
    P2 = P1*P1

    return c0[j] + c1[j]*P1 + c2[j]*P2 + c3[j]*P2*P1


def correct_light_travel_time(time, pl_params):
    '''Correct for the finite light travel speed.

    This function uses the KeplerOrbit.py file from the Bell_EBM package
    as that code includes a newer, faster method of solving Kepler's equation
    based on Tommasini+2018.

    Parameters
    ----------
    time : ndarray
        The times at which observations were collected
    pl_params : batman.TransitParams or poet.TransitParams
        The TransitParams object that contains information on the orbit.

    Returns
    -------
    time : ndarray
        Updated times that can be put into batman transit and eclipse functions
        that will give the expected results assuming a finite light travel
        speed.
    '''
    # Need to convert from a/Rs to a in meters
    a = pl_params.a * (pl_params.Rs*const.R_sun.value)

    if pl_params.ecc > 0:
        # Need to solve Kepler's equation, so use the KeplerOrbit class
        # for rapid computation. In the SPIDERMAN notation z is the radial
        # coordinate, while for Bell_EBM the radial coordinate is x
        orbit = KeplerOrbit(a=a, Porb=pl_params.per, inc=pl_params.inc,
                            t0=pl_params.t0, e=pl_params.ecc, argp=pl_params.w)
        old_x, _, _ = orbit.xyz(time)
        transit_x, _, _ = orbit.xyz(pl_params.t0)
    else:
        # No need to solve Kepler's equation for circular orbits, so save
        # some computation time
        transit_x = a*np.sin(pl_params.inc*np.pi/180)
        old_x = transit_x*np.cos(2*np.pi*(time-pl_params.t0)/pl_params.per)

    # Get the radial distance variations of the planet
    delta_x = transit_x - old_x

    # Compute for light travel time (and convert to days)
    delta_t = (delta_x/const.c.value)/(3600.*24.)

    # Subtract light travel time as a first-order correction
    # Batman will then calculate the model at a slightly earlier time
    return time-delta_t.flatten()