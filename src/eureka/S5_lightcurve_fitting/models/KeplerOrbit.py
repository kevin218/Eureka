import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as const
import scipy.optimize

from ...lib import plots


class KeplerOrbit(object):
    """A Keplerian orbit.

    Code taken from:
    https://github.com/taylorbell57/Bell_EBM/blob/master/Bell_EBM/KeplerOrbit.py

    Attributes
    ----------
    a : float
        The semi-major axis in m.
    inc : float
        The orbial inclination (in degrees above face-on)
    t0 : float
        The linear ephemeris in days.
    e : float
        The orbital eccentricity.
    Prot : float
        Body 2's rotational period in days.
    Omega : float
        The longitude of ascending node (in degrees CCW from
        line-of-sight).
    argp : float
        The argument of periastron (in degrees CCW from Omega).
    obliq : float; optional
        The obliquity (axial tilt) of body 2 (in degrees toward body 1).
    argobliq : float; optional
        The reference orbital angle used for obliq (in degrees from
        inferior conjunction).
    t_peri : float
        Time of body 2's closest approach to body 1.
    t_ecl : float
        Time of body 2's eclipse by body 1.
    mean_motion : float
        The mean motion in radians.
    """

    @property
    def m1(self):
        """Body 1's mass in kg.

        If no period was provided when the orbit was initialized,
        changing this will update the period.

        Returns
        -------
        float
            Body 1's mass in kg.
        """
        return self._m1

    @property
    def m2(self):
        """Body 2's mass in kg.

        If no period was provided when the orbit was initialized,
        changing this will update the period.

        Returns
        -------
        float
            Body 2's mass in kg.
        """
        return self._m2

    @property
    def phase_eclipse(self):
        """Phase of eclipse.

        Read-only.

        Returns
        -------
        float
            The orbital phase of eclipse.
        """
        return self.get_phase(self.t_ecl)

    @property
    def phase_periastron(self):
        """Phase of periastron.

        Read-only.

        Returns
        -------
        float
            The orbital phase of periastron.
        """
        return self.get_phase(self.t_peri)

    @property
    def phase_transit(self):
        """Phase of transit.

        Read-only.

        Returns
        -------
        float
            The orbital phase of transit.
        """
        return 0

    @property
    def Porb(self):
        """Orbital period.

        Changing this will update Prot if none was provided when the orbit
        was initialized.

        Returns
        -------
        float
            Body 2's orbital period in days.
        """
        return self._Porb

    @property
    def t_trans(self):
        """Time of transit.

        Read-only.

        Returns
        -------
        float
            Time of body 1's eclipse by body 2.
        """
        return self.t0

    def __init__(self, a=const.au.value,            # orbital parameters
                 Porb=None, inc=90., t0=0.,         # orbital parameters
                 e=0., Omega=270., argp=90.,        # orbital parameters
                 obliq=0., argobliq=0., Prot=None,  # spin parameters
                 m1=const.M_sun.value, m2=0.):      # mass parameters
        """Initialization function.

        Parameters
        ----------
        a : float; optional
            The semi-major axis in m. Detaults to 1 au.
        Porb : float; optional
            The orbital period in days. Defaults to None which
            computes the period using the m1 and m2 masses.
        inc : float; optional
            The orbial inclination (in degrees above face-on).
            Defaults to 90.
        t0 : float; optional
            The linear ephemeris in days. Defaults to 0.
        e : float; optional
            The orbital eccentricity. Defaults to 0.
        Omega : float; optional
            The longitude of ascending node (in degrees CCW from
            line-of-sight). Defaults to 270.
        argp : float; optional
            The argument of periastron (in degrees CCW from Omega).
            Defaults to 90.
        obliq : float; optional
            The obliquity (axial tilt) of body 2 (in degrees toward body 1).
            Defaults to 0.
        argobliq : float; optional
            The reference orbital angle used for obliq (in degrees from
            inferior conjunction). Defaults to 0.
        Prot : float; optional.
            The rotational period of body 2. Defaults to None which
            sets Prot equal to Porb (and will be updated when Porb is
            updated).
        m1 : float; optional
            The mass of body 1 in kg. Defaults to Msun.
        m2 : float; optional
            The mass of body 2 in kg. Defaults to 0.
        """
        self.e = e
        self.a = a
        self.inc = inc
        self.Omega = Omega
        self.argp = argp
        self.t0 = t0
        self._m1 = m1
        self._m2 = m2

        # Obliquity Attributes
        self.obliq = obliq  # degrees toward star
        self.argobliq = argobliq  # degrees from t0
        if -90. <= self.obliq <= 90.:
            self.ProtSign = 1.
        else:
            self.ProtSign = -1.

        # Input Period Attributes
        self.Porb_input = Porb
        self.Prot_input = Prot

        # Set Porb and dependent parameters
        if self.Porb_input is None:
            self.Porb = self.solve_period()
        else:
            self.Porb = Porb

        return

    @m1.setter
    def m1(self, m1):
        """m1 setter.

        Updates Porb if not explicity set during init."""
        self._m1 = m1
        if self.Porb_input is None:
            self.Porb = self.solve_period()
        return

    @m2.setter
    def m2(self, m2):
        """m2 setter.

        Updates Porb if not explicity set during init."""
        self._m2 = m2
        if self.Porb_input is None and self.m1 is not None:
            self.Porb = self.solve_period()
        return

    @Porb.setter
    def Porb(self, Porb):
        """Porb setter.

        Updates other computed parameters as needed.
        """
        self._Porb = Porb

        # Update self.Prot if needed
        if self.Prot_input is None:
            self.Prot = self.Porb*self.ProtSign

        self.t_peri = (self.t0 -
                       (self.ta_to_ma(np.pi/2.-self.argp*np.pi/180.) /
                        (2.*np.pi)*self.Porb))
        if self.t_peri < 0.:
            self.t_peri = self.Porb + self.t_peri

        self.t_ecl = (self.t0 +
                      (self.ta_to_ma(3.*np.pi/2.-self.argp*np.pi/180.)
                       - self.ta_to_ma(1.*np.pi/2.-self.argp*np.pi/180.)
                       )/(2.*np.pi)*self.Porb)
        if self.t_ecl < 0.:
            self.t_ecl = self.Porb + self.t_ecl

        self.mean_motion = 2.*np.pi/self.Porb

    def solve_period(self):
        """Find the Keplerian orbital period.

        Returns
        -------
        float
            The Keplerian orbital period.
        """
        return ((2.*np.pi*self.a**(3./2.)) /
                (np.sqrt(const.G.value*(self.m1+self.m2))) /
                (24.*3600.))

    def ta_to_ea(self, ta):
        """Convert true anomaly to eccentric anomaly.

        Parameters
        ----------
        ta : ndarray
            The true anomaly in radians.

        Returns
        -------
        ndarray
            The eccentric anomaly in radians.
        """
        return 2.*np.arctan(np.sqrt((1.-self.e)/(1.+self.e))*np.tan(ta/2.))

    def ea_to_ma(self, ea):
        """Convert eccentric anomaly to mean anomaly.

        Parameters
        ----------
        ea : ndarray
            The eccentric anomaly in radians.

        Returns
        -------
        ndarray
            The mean anomaly in radians.
        """
        return ea - self.e*np.sin(ea)

    def ta_to_ma(self, ta):
        """Convert true anomaly to mean anomaly.

        Parameters
        ----------
        ta : ndarray
            The true anomaly in radians.

        Returns
        -------
        ndarray
            The mean anomaly in radians.
        """
        return self.ea_to_ma(self.ta_to_ea(ta))

    def mean_anomaly(self, t):
        """Convert time to mean anomaly.

        Parameters
        ----------
        t : ndarray
            The time in days.

        Returns
        -------
        ndarray
            The mean anomaly in radians.
        """
        return ((t-self.t_peri) * self.mean_motion) % (2.*np.pi)

    def eccentric_anomaly(self, t, useFSSI=None, xtol=1e-10):
        """Convert time to eccentric anomaly, numerically.

        Parameters
        ----------
        t : ndarray
            The time in days.
        useFSSI : bool; optional
            Whether or not to use FSSI to invert Kepler's equation.
            Defaults to None which uses FSSI if t.size > 8.
        xtol : float; optional
            tolarance on error in eccentric anomaly. Defaults to 1e-10.

        Returns
        -------
        ndarray
            The eccentric anomaly in radians.
        """
        if not isinstance(t, np.ndarray):
            t = np.array([t])
        tShape = t.shape
        t = t.flatten()

        # Allow auto-switching for fast ODE runs and fast lightcurves
        if useFSSI is None and t.size < 8.:
            useFSSI = False
        elif useFSSI is None:
            useFSSI = True

        M = self.mean_anomaly(t)

        if useFSSI:
            E = self.FSSI_Eccentric_Inverse(M, xtol)
        else:
            E = self.Newton_Eccentric_Inverse(M, xtol)

        # Make some commonly used values exact
        E[np.abs(E) < xtol] = 0.
        E[np.abs(E-2*np.pi) < xtol] = 2.*np.pi
        E[np.abs(E-np.pi) < xtol] = np.pi

        return E.reshape(tShape)

    def Newton_Eccentric_Inverse(self, M, xtol=1e-10):
        """Convert mean anomaly to eccentric anomaly using Newton.

        Parameters
        ----------
        M : ndarray
            The mean anomaly in radians.
        xtol : float; optional
            tolarance on error in eccentric anomaly. Defaults to 1e-10.

        Returns
        -------
        ndarray
            The eccentric anomaly in radians.
        """
        def f(E):
            return E - self.e*np.sin(E) - M
        if self.e < 0.8:
            E0 = M
        else:
            E0 = np.pi*np.ones_like(M)
        E = scipy.optimize.fsolve(f, E0, xtol=xtol)

        return E

    def FSSI_Eccentric_Inverse(self, M, xtol=1e-10):
        """Convert mean anomaly to eccentric anomaly using FSSI algorithm.

        Algorithm based on that from Tommasini+2018.

        Parameters
        ----------
        M : ndarray
            The mean anomaly in radians.
        xtol : float; optional
            tolarance on error in eccentric anomaly. Defaults to 1e-10.

        Returns
        -------
        ndarray
            The eccentric anomaly in radians.
        """
        xtol = np.max([1e-15, xtol])
        nGrid = (xtol/100.)**(-1./4.)

        xGrid = np.linspace(0, 2.*np.pi, int(nGrid))

        def f(ea):
            return ea - self.e*np.sin(ea)

        def fP(ea):
            return 1. - self.e*np.cos(ea)

        return self.FSSI(M, x=xGrid, f=f, fP=fP)

    def FSSI(self, Y, x, f, fP):
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

        j = np.searchsorted(y1, Y)
        P1 = Y - y0[j]

        P2 = P1*P1
        return c0[j] + c1[j]*P1 + c2[j]*P2 + c3[j]*P2*P1

    def true_anomaly(self, t, xtol=1e-10):
        """Convert time to true anomaly, numerically.

        Parameters
        ----------
        t : ndarray
            The time in days.
        xtol : float; optional
            tolarance on error in eccentric anomaly (calculated along the way).
            Defaults to 1e-10.
        Returns
        -------
        ndarray
            The true anomaly in radians.
        """
        return 2.*np.arctan(np.sqrt((1.+self.e)/(1.-self.e)) *
                            np.tan(self.eccentric_anomaly(t, xtol=xtol)/2.))

    def distance(self, t=None, TA=None, xtol=1e-10):
        """Find the separation between the two bodies.

        Parameters
        ----------
        t : ndarray
            The time in days.
        TA : ndarray
            The true anomaly in radians (if t and TA are given, only TA
            will be used).
        xtol : float; optional
            tolarance on error in eccentric anomaly (calculated along the way).
            Defaults to 1e-10.

        Returns
        -------
        ndarray
            The separation between the two bodies.
        """
        if TA is None:
            if t is None:
                t = np.array([[0]])
            TA = self.true_anomaly(t, xtol=xtol)

        return self.a*(1.-self.e**2.)/(1.+self.e*np.cos(TA))

    def xyz(self, t, xtol=1e-10):
        """Find the coordinates of body 2 with respect to body 1.

        Parameters
        ----------
        t : ndarray
            The time in days.
        xtol : float; optional
            tolarance on error in eccentric anomaly (calculated along the way).
            Defaults to 1e-10.

        Returns
        -------
        list
            A list of 3 ndarrays containing the x,y,z coordinate of body 2
            with respect to body 1.
            - The x coordinate is along the line-of-sight.
            - The y coordinate is perpendicular to the line-of-sight and
            in the orbital plane.
            - The z coordinate is perpendicular to the line-of-sight and above
            the orbital plane
        """
        E = self.eccentric_anomaly(t, xtol=xtol)

        # The following code is roughly based on:
        # https://space.stackexchange.com/questions/8911/determining-orbital-position-at-a-future-point-in-time
        P = self.a*(np.cos(E)-self.e)
        Q = self.a*np.sin(E)*np.sqrt(1.-self.e**2.)

        # Rotate by argument of periapsis
        x = (np.cos(self.argp*np.pi/180.-np.pi/2.)*P -
             np.sin(self.argp*np.pi/180.-np.pi/2.)*Q)
        y = (np.sin(self.argp*np.pi/180.-np.pi/2.)*P +
             np.cos(self.argp*np.pi/180.-np.pi/2.)*Q)

        # Rotate by inclination
        z = -np.sin(np.pi/2.-self.inc*np.pi/180.)*x
        x = np.cos(np.pi/2.-self.inc*np.pi/180.)*x

        # Rotate by longitude of ascending node
        xtemp = x
        x = -(np.sin(self.Omega*np.pi/180.)*xtemp +
              np.cos(self.Omega*np.pi/180.)*y)
        y = (np.cos(self.Omega*np.pi/180.)*xtemp -
             np.sin(self.Omega*np.pi/180.)*y)

        return x, y, z

    def get_phase(self, t, TA=None):
        """Get the orbital phase.

        Parameters
        ----------
        t : ndarray
            The time in days.
        TA : ndarray; optional
            The true anomaly. Defaults to None which calculates the TA
            using self.true_anomaly(t).

        Returns
        -------
        ndarray
            The orbital phase.
        """
        if TA is None:
            TA = self.true_anomaly(t)

        phase = (TA-self.true_anomaly(self.t0))/(2.*np.pi)
        phase = phase + 1.*(phase < 0.).astype(int)
        return phase

    def get_ssp(self, t, TA=None):
        """Calculate the sub-stellar longitude and latitude.

        Parameters
        ----------
        t : ndarray
            The time in days.
        TA : ndarray; optional
            The true anomaly. Defaults to None which calculates the TA
            using self.true_anomaly(t).

        Returns
        -------
        list
            A list of 2 ndarrays containing the sub-stellar longitude
            and latitude. Each ndarray is in the same shape as t.
        """
        if self.e == 0. and self.Prot == self.Porb:
            if not isinstance(t, np.ndarray):
                sspLon = np.zeros_like([t])
            else:
                sspLon = np.zeros_like(t)
        else:
            if TA is None:
                TA = self.true_anomaly(t)

            sspLon = (TA*180./np.pi - (t-self.t0)/self.Prot*360. +
                      self.Omega+self.argp)
            sspLon = (sspLon % 180. +
                      (-180.)*(np.rint(np.floor(sspLon % 360./180.) > 0)))

        if self.obliq == 0.:
            if not isinstance(t, np.ndarray):
                sspLat = np.zeros_like([t])
            else:
                sspLat = np.zeros_like(t)
        else:
            sspLat = (self.obliq*np.cos(self.get_phase(t, TA)*2. *
                      np.pi-self.argobliq*np.pi/180.))

        return sspLon, sspLat

    def get_sop(self, t):
        """Calculate the sub-observer longitude and latitude.

        Parameters
        ----------
        t : ndarray
            The time in days.

        Returns
        -------
        list
            A list of 2 ndarrays containing the sub-observer longitude
            and latitude. Each ndarray is in the same shape as t.
        """
        sopLon = 180.-((t-self.t0)/self.Prot)*360.
        sopLon = (sopLon % 180. +
                  (-180.)*(np.rint(np.floor(sopLon % 360./180.) > 0.)))
        sopLat = 90.-self.inc-self.obliq
        return sopLon, sopLat

    @plots.apply_style
    def plot_orbit(self):
        """A convenience routine to visualize the orbit

        Returns
        -------
        figure
            The figure containing the plot.
        """
        t = np.linspace(0., self.Porb, 100, endpoint=False)

        x, y, z = np.array(self.xyz(t))/const.au.value

        lim = 1.2*np.max([np.max(np.abs(x)), np.max(np.abs(y)),
                          np.max(np.abs(z))])

        xTrans, yTrans, zTrans = np.array(self.xyz(self.t0))/const.au.value
        xEcl, yEcl, zEcl = np.array(self.xyz(self.t_ecl))/const.au.value
        xPeri, yPeri, zPeri = np.array(self.xyz(self.t_peri))/const.au.value

        fig, axes = plt.subplots(3, 1, sharex=False, figsize=(4, 12))

        axes[0].plot(y, x, '.', c='k', ms=2)
        axes[0].plot(0, 0, '*', c='r', ms=15)
        axes[0].plot(yTrans, xTrans, 'o', c='b', ms=10, label=r'$\rm Transit$')
        axes[0].plot(yEcl, xEcl, 'o', c='k', ms=7, label=r'$\rm Eclipse$')
        if self.e != 0:
            axes[0].plot(yPeri, xPeri, 'o', c='r', ms=5,
                         label=r'$\rm Periastron$')
        axes[0].set_xlabel('$y$')
        axes[0].set_ylabel('$x$')
        axes[0].set_xlim(-lim, lim)
        axes[0].set_ylim(-lim, lim)
        axes[0].invert_yaxis()
        axes[0].legend(loc=6, bbox_to_anchor=(1, 0.5))

        axes[1].plot(y, z, '.', c='k', ms=2)
        axes[1].plot(0, 0, '*', c='r', ms=15)
        axes[1].plot(yTrans, zTrans, 'o', c='b', ms=10)
        axes[1].plot(yEcl, zEcl, 'o', c='k', ms=7)
        if self.e != 0:
            axes[1].plot(yPeri, zPeri, 'o', c='r', ms=5)
        axes[1].set_xlabel('$y$')
        axes[1].set_ylabel('$z$')
        axes[1].set_xlim(-lim, lim)
        axes[1].set_ylim(-lim, lim)

        axes[2].plot(x, z, '.', c='k', ms=2)
        axes[2].plot(0, 0, '*', c='r', ms=15)
        axes[2].plot(xTrans, zTrans, 'o', c='b', ms=10)
        axes[2].plot(xEcl, zEcl, 'o', c='k', ms=7)
        if self.e != 0:
            axes[2].plot(xPeri, zPeri, 'o', c='r', ms=5)
        axes[2].set_xlabel('$x$')
        axes[2].set_ylabel('$z$')
        axes[2].set_xlim(-lim, lim)
        axes[2].set_ylim(-lim, lim)

        fig.get_layout_engine().set(hspace=0.35)

        return fig
