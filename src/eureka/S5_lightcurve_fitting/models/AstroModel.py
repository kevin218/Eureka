import numpy as np
import astropy.constants as const
from copy import deepcopy

from .Model import Model
from . import KeplerOrbit
from ...lib.split_channels import split


class AstroModel(Model):
    """A model which combines all astrophysical components."""
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

        pid_input = deepcopy(pid)
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
                if self.eclipse_model is not None:
                    planetFlux = self.eclipse_model.eval(channel=chan, pid=pid,
                                                         **kwargs)
                else:
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
        Either np (numpy) or tt (theano.tensor), depending on whether the
        code is being run in numpy or theano mode. Defaults to np.

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
        Either np (numpy) or tt (theano.tensor), depending on whether the
        code is being run in numpy or theano mode. Defaults to np.
    xtol : float; optional
        tolarance on error in eccentric anomaly (calculated along the way).
        Defaults to 1e-10.

    Returns
    -------
    ndarray
        The true anomaly in radians.

    Notes
    -----
    History:

    - March 2023 Taylor Bell
        Based on Bell_EBM code, but modified to enable theano code.
    """
    return 2.*lib.arctan(lib.sqrt((1.+model.ecc)/(1.-model.ecc)) *
                         lib.tan(eccentric_anomaly(model, t, lib,
                                                   xtol=xtol)/2.))


def eccentric_anomaly(model, t, lib=np, xtol=1e-10):
    """Convert time to eccentric anomaly, numerically.

    Parameters
    ----------
    model : pymc3.Model
        The PyMC3 model (which contains the orbital parameters).
    t : ndarray
        The time in days.
    lib : library; optional
        Either np (numpy) or tt (theano.tensor), depending on whether the
        code is being run in numpy or theano mode. Defaults to np.
    xtol : float; optional
        tolarance on error in eccentric anomaly. Defaults to 1e-10.

    Returns
    -------
    ndarray
        The eccentric anomaly in radians.

    Notes
    -----
    History:

    - March 2023 Taylor Bell
        Based on Bell_EBM code, but modified to enable theano code.
    """
    ta_peri = np.pi/2.-model.w*np.pi/180.
    ea_peri = 2.*lib.arctan(lib.sqrt((1.-model.ecc)/(1.+model.ecc)) *
                            lib.tan(ta_peri/2.))
    ma_peri = ea_peri - model.ecc*np.sin(ea_peri)
    t_peri = (model.t0 - (ma_peri/(2.*np.pi)*model.per))

    if lib != np:
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
    model : pymc3.Model
        The PyMC3 model (which contains the orbital parameters).
    M : ndarray
        The mean anomaly in radians.
    lib : library; optional
        Either np (numpy) or tt (theano.tensor), depending on whether the
        code is being run in numpy or theano mode. Defaults to numpy.
    xtol : float; optional
        tolarance on error in eccentric anomaly. Defaults to 1e-10.

    Returns
    -------
    ndarray
        The eccentric anomaly in radians.

    Notes
    -----
    History:

    - March 2023 Taylor Bell
        Based on Bell_EBM code, but modified to enable theano code.
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
        Either np (numpy) or tt (theano.tensor), depending on whether the
        code is being run in numpy or theano mode. Defaults to np.

    Returns
    -------
    ndarray
        The numerical approximation of f^-(y).

    Notes
    -----
    History:

    - March 2023 Taylor Bell
        Based on Bell_EBM code, but modified to enable theano code.
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

    if lib == np:
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

    Notes
    -----
    History:

    - 2022-03-31 Taylor J Bell
        Initial version based on the Bell_EMB KeplerOrbit.py file by
        Taylor J Bell and the light travel time calculations of SPIDERMAN's
        web.c file by Tom Louden
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
        transit_x = a*np.sin(pl_params.inc)
        old_x = transit_x*np.cos(2*np.pi*(time-pl_params.t0)/pl_params.per)

    # Get the radial distance variations of the planet
    delta_x = transit_x - old_x

    # Compute for light travel time (and convert to days)
    delta_t = (delta_x/const.c.value)/(3600.*24.)

    # Subtract light travel time as a first-order correction
    # Batman will then calculate the model at a slightly earlier time
    return time-delta_t.flatten()