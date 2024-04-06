import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from . import PyMC3Model
from .StarryModel import PlanetParams
from ...lib.split_channels import split


class SinusoidPhaseCurveModel(PyMC3Model):
    def __init__(self, starry_model=None, **kwargs):
        """Initialize the model.

        Parameters
        ----------
        starry_model : eureka.S5_lightcurve_fitting.differentiable_models.StarryModel
            The starry model to combined with this phase curve model.
            Defaults to None.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.differentiable_models.PyMC3Model.__init__().
        """  # NOQA: E501
        # Inherit from PyMC3Model class
        super().__init__(**kwargs,
                         components=[starry_model,],
                         modeltype='physical')

        # Temporary code to switch between different starry models
        self.starry_model_name = starry_model.name

    def eval(self, eval=True, channel=None, pid=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        eval : bool; optional
            If true evaluate the model, otherwise simply compile the model.
            Defaults to True.
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        pid : int; optional
            Planet ID, default is None which combines the models from
            all planets.
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

        if pid is None:
            pid_iter = range(self.num_planets)
        else:
            pid_iter = [pid,]

        if eval:
            lib = np.ma
            model = self.fit
            # systems = self.components[0].fit.systems
        else:
            lib = tt
            model = self.model
            # systems = self.components[0].systems

        lcfinal = lib.zeros(0)
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            for pid in pid_iter:
                # Initialize model
                pl_params = PlanetParams(model, pid, chan, eval=eval)

                if (eval and pl_params.AmpCos1 == 0 and pl_params.AmpSin1 == 0
                        and pl_params.AmpCos2 == 0 and pl_params.AmpSin2 == 0):
                    # Don't waste time running the following code
                    phaseVars = np.ma.ones_like(time)
                    continue

                if pl_params.t_secondary is None:
                    # If not explicitly fitting for the time of eclipse, get
                    # the time of eclipse from the time of transit, period,
                    # eccentricity, and argument of periastron
                    pl_params.t_secondary = get_ecl_midpt(pl_params)

                # Compute orbital phase
                if pl_params.ecc == 0:
                    # the planet is on a circular orbit
                    t = self.time - pl_params.t_secondary
                    phi = 2*np.pi/pl_params.per*t
                else:
                    # the planet is on an eccentric orbit
                    anom = true_anomaly(pl_params, lib, self.time)
                    phi = anom + pl_params.w*np.pi/180 + np.pi/2

                # calculate the phase variations
                if eval and pl_params.AmpCos2 == 0 and pl_params.AmpSin2 == 0:
                    # Skip multiplying by a bunch of zeros to speed up fitting
                    phaseVars = (1 + pl_params.AmpCos1*(lib.cos(phi)-1) +
                                 pl_params.AmpSin1*lib.sin(phi))
                else:
                    phaseVars = (1 + pl_params.AmpCos1*(lib.cos(phi)-1) +
                                 pl_params.AmpSin1*lib.sin(phi) +
                                 pl_params.AmpCos2*(lib.cos(2*phi)-1) +
                                 pl_params.AmpSin2*lib.sin(2*phi))

            if eval:
                # Evaluate if needed
                phaseVars = phaseVars.eval()
            lcfinal = lib.concatenate([lcfinal, phaseVars])

        return lcfinal

def get_ecl_midpt(params, lib):
    """
    Return the time of secondary eclipse center.

    Parameters
    ----------
    params : object
        Contains the physical parameters for the transit model.
    lib : library
        Either np (numpy) or tt (theano.tensor), depending on whether the
        code is being run in numpy or theano mode.

    Returns
    -------
    t_secondary : float
        The time of secondary eclipse.
    """
    # FINDME: The this function can likely heavily reuse the functions below

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

def true_anomaly(model, lib, t, xtol=1e-10):
    """Convert time to true anomaly, numerically.

    Parameters
    ----------
    model : pymc3.Model
        The PyMC3 model (which contains the orbital parameters).
    lib : library
        Either np (numpy) or tt (theano.tensor), depending on whether the
        code is being run in numpy or theano mode.
    t : ndarray
        The time in days.
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
                         lib.tan(eccentric_anomaly(model, lib, t,
                                                   xtol=xtol)/2.))


def eccentric_anomaly(model, lib, t, xtol=1e-10):
    """Convert time to eccentric anomaly, numerically.

    Parameters
    ----------
    model : pymc3.Model
        The PyMC3 model (which contains the orbital parameters).
    lib : library
        Either np (numpy) or tt (theano.tensor), depending on whether the
        code is being run in numpy or theano mode.
    t : ndarray
        The time in days.
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

    if ((lib == tt and tt.lt(t_peri, 0)) or 
            (t_peri < 0)):
        t_peri = t_peri + model.per

    M = ((t-t_peri) * 2.*np.pi/model.per) % (2.*np.pi)

    E = FSSI_Eccentric_Inverse(model, lib, M, xtol)

    return E


def FSSI_Eccentric_Inverse(model, lib, M, xtol=1e-10):
    """Convert mean anomaly to eccentric anomaly using FSSI algorithm.
    
    Algorithm based on that from Tommasini+2018.

    Parameters
    ----------
    model : pymc3.Model
        The PyMC3 model (which contains the orbital parameters).
    lib : library
        Either np (numpy) or tt (theano.tensor), depending on whether the
        code is being run in numpy or theano mode.
    M : ndarray
        The mean anomaly in radians.
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

    return FSSI(lib, M, x=xGrid, f=f, fP=fP)


def FSSI(lib, Y, x, f, fP):
    """Fast Switch and Spline Inversion method from Tommasini+2018.

    Parameters
    ----------
    lib : library
        Either np (numpy) or tt (theano.tensor), depending on whether the
        code is being run in numpy or theano mode.
    Y : ndarray
        The f(x) values to invert.
    x : ndarray
        x values spanning the domain (more values for higher precision).
    f : callable
        The function f.
    fP : callable
        The first derivative of the function f with respect to x.

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
        j = tt.extra_ops.searchsorted(y1, Y)
        # Protect against indexing beyond the size of the array
        # Theano tensors don't allow item assignment
        mask = tt.ge(j, y1.size)
        j = j*(1-mask) + (y1.size-1)*mask

    P1 = Y - y0[j]
    P2 = P1*P1

    return c0[j] + c1[j]*P1 + c2[j]*P2 + c3[j]*P2*P1
