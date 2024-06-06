import numpy as np
import batman as bm

from .Model import Model
from .BatmanModels import BatmanTransitModel, BatmanEclipseModel, \
    PlanetParams, get_ecl_midpt
from ...lib.split_channels import split


class PoetTransitModel(BatmanTransitModel):
    """Transit Model"""
    def __init__(self, **kwargs):
        """Initialize the transit model

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
        # Define transit model to be used
        self.transit_model = TransitModel


class PoetEclipseModel(BatmanEclipseModel):
    """Eclipse Model"""
    def __init__(self, **kwargs):
        """Initialize the eclipse model

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
        # Define transit model to be used
        self.transit_model = TransitModel


class PoetPCModel(Model):
    """Phase Curve Model"""
    def __init__(self, transit_model=None, eclipse_model=None, **kwargs):
        """Initialize the phase curve model

        Parameters
        ----------
        transit_model : eureka.S5_lightcurve_fitting.models.Model; optional
            The transit model to use for this phase curve model.
            Defaults to None.
        eclipse_model : eureka.S5_lightcurve_fitting.models.Model; optional
            The eclipse model to use for this phase curve model.
            Defaults to None.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        self.components = []
        self.transit_model = transit_model
        self.eclipse_model = eclipse_model
        if transit_model is not None:
            self.components.append(self.transit_model)
        if eclipse_model is not None:
            self.components.append(self.eclipse_model)

        # Inherit from Model class
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        # Check if should enforce positivity
        if not hasattr(self, 'force_positivity'):
            self.force_positivity = False

    @property
    def time(self):
        """A getter for the time."""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time."""
        time_array = np.ma.masked_array(time_array)
        self._time = time_array
        if self.transit_model is not None:
            self.transit_model.time = time_array
        if self.eclipse_model is not None:
            self.eclipse_model.time = time_array

    @property
    def nints(self):
        """A getter for the nints."""
        return self._nints

    @nints.setter
    def nints(self, nints_array):
        """A setter for the nints."""
        self._nints = nints_array
        if self.transit_model is not None:
            self.transit_model.nints = nints_array
        if self.eclipse_model is not None:
            self.eclipse_model.nints = nints_array

    def update(self, newparams, **kwargs):
        """Update the model with new parameter values.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.update().
        """
        super().update(newparams, **kwargs)
        if self.transit_model is not None:
            self.transit_model.update(newparams, **kwargs)
        if self.eclipse_model is not None:
            self.eclipse_model.update(newparams, **kwargs)

    def eval(self, channel=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
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

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Set all parameters
        lcfinal = np.ma.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            light_curve = np.ma.zeros(time.shape)
            for pid in range(self.num_planets):
                # Initialize planet
                poet_params = PlanetParams(self, pid, chan)

                poet_params.limb_dark = 'uniform'
                poet_params.u = []

                if poet_params.t_secondary is None:
                    # If not explicitly fitting for the time of eclipse, get
                    # the time of eclipse from the time of transit, period,
                    # eccentricity, and argument of periastron
                    poet_params.t_secondary = get_ecl_midpt(poet_params)

                # calculate the phase variations
                p = poet_params.per
                t1 = poet_params.cos1_off*p/360. - poet_params.t_secondary
                t2 = poet_params.cos2_off*p/360. - poet_params.t_secondary
                phaseVars = (poet_params.cos1_amp/2 *
                             np.cos(2*np.pi*(time+t1)/p) +
                             poet_params.cos2_amp/2 *
                             np.cos(4*np.pi*(time+t2)/p))
                # Apply normalizing offset
                ieclipse = np.argmin(np.abs(time-poet_params.t_secondary))
                phaseVars += 1 - phaseVars[ieclipse]

                # If requested, force positive phase variations
                if self.force_positivity and np.ma.any(phaseVars < 0):
                    # Returning nans or infs breaks the fits, so this was
                    # the best I could think of
                    phaseVars = 1e12*np.ma.ones(time.shape)

                # Compute eclipse model
                if self.eclipse_model is None:
                    eclipse = 1
                else:
                    eclipse = self.eclipse_model.eval(channel=chan,
                                                      pid=pid) - 1

                light_curve += eclipse*phaseVars

            lcfinal = np.ma.append(lcfinal, light_curve)

        if self.transit_model is None:
            transit = 1
        else:
            transit = self.transit_model.eval(channel=channel)

        return transit + lcfinal


class TransitModel():
    """
    Class for generating model transit light curves.
    """
    def __init__(self, params, t, transittype="primary"):
        """
        Initializes model parameters and computes planet-star-distance

        Parameters
        ----------
        params : object
            Contains the physical parameters for the transit model.
        t : array
            Array of times.
        transittype : str; optional
            Options are primary or secondary.  Default is primary.
        """
        self.t = t
        self.t0 = params.t0
        self.rprs = params.rprs
        self.inc = params.inc
        self.ars = params.ars
        self.per = params.per
        self.u = params.u
        self.limb_dark = params.limb_dark
        self.transittype = transittype
        self.t_secondary = params.t_secondary
        self.nthreads = 4

        # Handles the case of inverse transits (rp < 0)
        self.inverse = False
        if params.rprs < 0.:
            self.inverse = True

        if self.transittype == 'primary':
            tref = self.t0
        else:
            tref = params.t_secondary

        # Compute distance, z, of planet and star midpoints
        self.z = self.ars \
            * np.sqrt(np.sin(2*np.pi*(t-tref)/self.per)**2
                      + (np.cos(self.inc*np.pi/180)
                         * np.cos(2*np.pi*(t-tref)/self.per))**2)

        # Ignore close approach on other side of the orbit
        self.z[np.where(np.bitwise_and(
            (t-tref) % self.per > self.per/4,
            (t-tref) % self.per < self.per*3/4))] = self.ars

    def light_curve(self, params):
        """
        Calculate a model light curve.

        Parameters
        ----------
        params : object
            Contains the physical parameters for the transit model.

        Returns
        -------
        lc : ndarray
            Light curve.
        """
        # Update transit params
        self.t0 = params.t0
        self.rprs = params.rprs
        self.inc = params.inc
        self.ars = params.ars
        self.per = params.per
        self.u = params.u
        self.limb_dark = params.limb_dark

        # Handle the case of inverse transits (rp < 0)
        self.inverse = False
        if params.rprs < 0.:
            self.inverse = True

        if self.transittype == 'primary':
            # Primary transit
            if self.limb_dark == "quadratic":
                lc = trquad(self.z, params.rprs, params.u[0], params.u[1])
            elif self.limb_dark == "linear":
                lc = trquad(self.z, params.rprs, params.u[0], 0)
            elif self.limb_dark == "nonlinear":
                lc = trnlldsp(self.z, params.rprs, params.u)
            elif self.limb_dark == "uniform":
                lc = uniform(self.z, params.rprs)
            else:
                raise Exception('Invalid limb darkening option.  '
                                + 'POET supports linear, quadratic, '
                                + '4-parameter, and uniform.')
            if self.inverse:
                lc = 2. - lc
        elif self.transittype == 'secondary':
            # Secondary eclipse
            lc = bm._eclipse._eclipse(self.z, np.abs(params.rprs),
                                      params.fpfs, self.nthreads)
        return lc


def uniform(z, rprs):
    """
    This function computes the primary transit shape
    using equations provided by Mandel & Agol (2002).

    Parameters
    ----------
    z : ndarray
        Distance between planet and star midpoints
    rprs : float
        Planet-to-star radius ratio

    Returns
    -------
    y : ndarray
        The flux for each point in time.

    Notes
    -----
    History:

    - 2010-11-27 Kevin Stevenson
        Original version
    - 2024-01-28 Kevin Stevenson
        Updated for Eureka!
    """
    # INGRESS/EGRESS INDICES
    iingress = np.where(np.bitwise_and((1-rprs) < z, z <= (1+rprs)))[0]
    # COMPUTE k0 & k1
    k0 = np.arccos((rprs**2 + z[iingress]**2 - 1) / 2 / rprs / z[iingress])
    k1 = np.arccos((1 - rprs**2 + z[iingress]**2) / 2 / z[iingress])

    # CALCULATE TRANSIT SHAPE
    # Baseline
    y = np.ones_like(z)
    # Full transit
    y[np.where(z <= (1-rprs))] = 1.-rprs**2
    # Ingress/egress
    y[iingress] = 1. - 1./np.pi*(k0*rprs**2 + k1 - np.sqrt((4*z[iingress]**2
                                 - (1 + z[iingress]**2 - rprs**2)**2)/4))

    return y


def trnlldsp(z, rprs, u):
    """
    This function computes the primary transit shape using non-linear
    limb-darkening equations for a "small planet" (rprs <= 0.1),
    as provided by Mandel & Agol (2002).

    Parameters
    ----------
    z : ndarray
        Distance between planet and star midpoints
    rprs : float
        Planet-to-star radius ratio
    u : ndarray
        Limb darkening parameters (u1, u2, u3, u4)

    Returns
    -------
    y : ndarray
        The flux for each point in time.
    Notes
    -----
    History:

    - 2010-12-15 Kevin Stevenson
        Converted to Python
    - 2024-01-28 Kevin Stevenson
        Updated for Eureka!
    """

    # DEFINE PARAMETERS
    (u1, u2, u3, u4) = u
    Sigma4 = 1. - u1 / 5. - u2 / 3. - 3. * u3 / 7. - u4 / 2.

    # CALCULATE TRANSIT SHAPE WITH LIMB-DARKENING
    y = np.ones_like(z, dtype=float)
    if rprs == 0:
        return y

    # INGRESS/EGRESS
    iingress = np.where(np.bitwise_and(1 - rprs < z, z <= 1 + rprs))[0]
    x = 1. - (z[iingress] - rprs) ** 2
    I1star = 1. - u1 * (1. - 4. / 5. * np.sqrt(np.sqrt(x))) \
                - u2 * (1. - 2. / 3. * np.sqrt(x)) \
                - u3 * (1. - 4. / 7. * np.sqrt(np.sqrt(x * x * x))) \
                - u4 * (1. - 4. / 8. * x)
    y[iingress] = 1. - I1star \
        * (rprs**2 * np.arccos((z[iingress] - 1.) / rprs) - (z[iingress] - 1.)
           * np.sqrt(rprs ** 2 - (z[iingress] - 1.) ** 2)) / np.pi / Sigma4

    # Full transit (except @ z=0)
    itrans = np.where(np.bitwise_and(z <= 1 - rprs, z != 0.))
    sig1 = np.sqrt(np.sqrt(1. - (z[itrans] - rprs) ** 2))
    sig2 = np.sqrt(np.sqrt(1. - (z[itrans] + rprs) ** 2))
    I2star = 1. \
        - u1 * (1. + (sig2 ** 5 - sig1 ** 5) / 5. / rprs / z[itrans]) \
        - u2 * (1. + (sig2 ** 6 - sig1 ** 6) / 6. / rprs / z[itrans]) \
        - u3 * (1. + (sig2 ** 7 - sig1 ** 7) / 7. / rprs / z[itrans]) \
        - u4 * (rprs ** 2 + z[itrans] ** 2)
    y[itrans] = 1. - rprs ** 2 * I2star / Sigma4

    # z=0 (midpoint)
    y[np.where(z == 0.)] = 1. - rprs ** 2 / Sigma4

    return y


def trquad(z, rprs, u1, u2):
    '''
    Transit model with quadratic (or linear) limb darkening.

    Parameters
    ----------
    z : ndarray
        Distance between planet and star midpoints
    rprs : float
        Planet-to-star radius ratio
    u1 : ndarray
        Linear imb darkening parameter
    u2 : ndarray
        Quadratic limb darkening parameter

    Returns
    -------
    y : ndarray
        The flux for each point in time.

    Notes
    -----
    History:

    - 2012-08-13 Kevin Stevenson
        Modified from Jason Eastman's version
    2024-01-28 Kevin Stevenson
        Updated for Eureka!
    '''

    nz = np.size(z)
    lambdad = np.zeros(nz)
    etad = np.zeros(nz)
    lambdae = np.zeros(nz)
    omega = 1. - u1 / 3. - u2 / 6.

    # # tolerance for double precision equalities
    # # special case integrations

    tol = 1e-14

    rprs = np.abs(rprs)

    z = np.where(np.abs(rprs - z) < tol, rprs, z)
    z = np.where(np.abs(rprs - 1 - z) < tol, rprs - 1., z)
    z = np.where(np.abs(1 - rprs - z) < tol, 1. - rprs, z)
    z = np.where(z < tol, 0., z)

    x1 = (rprs - z) ** 2.
    x2 = (rprs + z) ** 2.
    x3 = rprs ** 2. - z ** 2.

    # # trivial case of no planet

    if rprs <= 0.:
        muo1 = np.zeros(nz) + 1.
        return muo1

    # # Case 1 - the star is unocculted:
    # # only consider points with z < 1+rprs

    notusedyet = np.where(z < 1. + rprs)[0]
    if np.size(notusedyet) == 0:
        muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1 + 2. * u2)
                     * (lambdad + 2. / 3. * (rprs > z)) + u2 * etad) \
            / omega
        return muo1

    # Case 11 - the source is completely occulted:

    if rprs >= 1.:
        occulted = np.where(z[notusedyet] <= rprs - 1.)
        if np.size(occulted) != 0:
            ndxuse = notusedyet[occulted]
            etad[ndxuse] = 0.5  # corrected typo in paper
            lambdae[ndxuse] = 1.

            # lambdad = 0 already

            notused2 = np.where(z[notusedyet] > rprs - 1)
            if np.size(notused2) == 0:
                muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1 + 2.
                             * u2) * (lambdad + 2. / 3. * (rprs > z))
                             + u2 * etad) / omega
                return muo1
            notusedyet = notusedyet[notused2]

    # Case 2, 7, 8 - ingress/egress (uniform disk only)

    inegressuni = np.where((z[notusedyet] >= np.abs(1. - rprs))
                           & (z[notusedyet] < 1. + rprs))
    if np.size(inegressuni) != 0:
        ndxuse = notusedyet[inegressuni]
        tmp = (1. - rprs ** 2. + z[ndxuse] ** 2.) / 2. / z[ndxuse]
        tmp = np.where(tmp > 1., 1., tmp)
        tmp = np.where(tmp < -1., -1., tmp)
        kap1 = np.arccos(tmp)
        tmp = (rprs ** 2. + z[ndxuse] ** 2 - 1.) / 2. / rprs / z[ndxuse]
        tmp = np.where(tmp > 1., 1., tmp)
        tmp = np.where(tmp < -1., -1., tmp)
        kap0 = np.arccos(tmp)
        tmp = 4. * z[ndxuse] ** 2 - (1. + z[ndxuse] ** 2 - rprs ** 2) \
            ** 2
        tmp = np.where(tmp < 0, 0, tmp)
        lambdae[ndxuse] = (rprs ** 2 * kap0 + kap1 - 0.5 * np.sqrt(tmp)) \
            / np.pi

        # eta_1

        etad[ndxuse] = 1. / 2. / np.pi \
            * (kap1 + rprs ** 2 * (rprs ** 2 + 2. * z[ndxuse] ** 2)
               * kap0 - (1. + 5. * rprs ** 2 + z[ndxuse] ** 2)
               / 4. * np.sqrt((1. - x1[ndxuse]) * (x2[ndxuse] - 1.)))

    # Case 5, 6, 7 - the edge of planet lies at origin of star

    ocltor = np.where(z[notusedyet] == rprs)
    if np.size(ocltor) != 0:
        ndxuse = notusedyet[ocltor]
        if rprs < 0.5:

            # Case 5

            q = 2. * rprs  # corrected typo in paper (2k -> 2rprs)
            (Ek, Kk) = ellke(q)

            # lambda_4

            lambdad[ndxuse] = 1. / 3. + 2. / 9. / np.pi \
                * (4. * (2. * rprs ** 2 - 1.) * Ek
                   + (1. - 4. * rprs ** 2) * Kk)

            # eta_2

            etad[ndxuse] = rprs ** 2 / 2. \
                * (rprs ** 2 + 2. * z[ndxuse] ** 2)
            lambdae[ndxuse] = rprs ** 2  # uniform disk
        elif rprs > 0.5:

            # Case 7

            q = 0.5 / rprs  # corrected typo in paper (1/2k -> 1/2rprs)
            (Ek, Kk) = ellke(q)

            # lambda_3

            lambdad[ndxuse] = 1. / 3. + 16. * rprs / 9. / np.pi \
                * (2. * rprs ** 2 - 1.) * Ek \
                - (32. * rprs ** 4 - 20. * rprs ** 2 + 3.) \
                / 9. / np.pi / rprs * Kk

        else:
            # etad = eta_1 already
            # Case 6

            lambdad[ndxuse] = 1. / 3. - 4. / np.pi / 9.
            etad[ndxuse] = 3. / 32.
        notused3 = np.where(z[notusedyet] != rprs)
        if np.size(notused3) == 0:
            muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1 + 2. * u2)
                         * (lambdad + 2. / 3. * (rprs > z)) + u2
                         * etad) / omega
            return muo1
        notusedyet = notusedyet[notused3]

    # Case 2, Case 8 - ingress/egress (with limb darkening)

    inegress = np.where((z[notusedyet] > 0.5 + np.abs(rprs - 0.5))
                        & (z[notusedyet] < 1. + rprs) | (rprs > 0.5)
                        & (z[notusedyet] > np.abs(1. - rprs))
                        & (z[notusedyet] < rprs))  # , complement=notused4)
    if np.size(inegress) != 0:
        ndxuse = notusedyet[inegress]
        q = np.sqrt((1. - x1[ndxuse]) / (x2[ndxuse] - x1[ndxuse]))
        (Ek, Kk) = ellke(q)
        n = 1. / x1[ndxuse] - 1.

        # lambda_1:

        lambdad[ndxuse] = 2. / 9. / np.pi / np.sqrt(x2[ndxuse] - x1[ndxuse]) \
            * (((1. - x2[ndxuse]) * (2. * x2[ndxuse] + x1[ndxuse] - 3.)
               - 3. * x3[ndxuse] * (x2[ndxuse] - 2.)) * Kk
               + (x2[ndxuse] - x1[ndxuse]) * (z[ndxuse] ** 2 + 7.
               * rprs ** 2 - 4.) * Ek - 3. * x3[ndxuse] / x1[ndxuse]
               * ellpic_bulirsch(n, q))
        notused4 = np.where(
            ((z[notusedyet] <= 0.5 + np.abs(rprs - 0.5)) |
             (z[notusedyet] >= 1.0 + rprs))
            & ((rprs <= 0.5) | (z[notusedyet] <= np.abs(1.0 - rprs)) |
               (z[notusedyet] >= rprs)))

        if np.size(notused4) == 0:
            muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1 + 2. * u2)
                         * (lambdad + 2. / 3. * (rprs > z)) + u2
                         * etad) / omega
            return muo1
        notusedyet = notusedyet[notused4]

    # Case 3, 4, 9, 10 - planet completely inside star

    if rprs < 1.:
        inside = np.where(z[notusedyet] <= 1. - rprs)
        if np.size(inside) != 0:
            ndxuse = notusedyet[inside]

            # # eta_2

            etad[ndxuse] = rprs ** 2 / 2. * (rprs ** 2 + 2. * z[ndxuse] ** 2)

            # # uniform disk

            lambdae[ndxuse] = rprs ** 2

            # # Case 4 - edge of planet hits edge of star

            edge = np.where(z[ndxuse] == 1. - rprs)
            if np.size(edge[0]) != 0:

                # # lambda_5

                lambdad[ndxuse[edge]] = 2. / 3. / np.pi \
                    * np.arccos(1. - 2. * rprs) \
                    - 4. / 9. / np.pi * np.sqrt(rprs * (1. - rprs)) \
                    * (3. + 2. * rprs - 8. * rprs ** 2)
                if rprs > 0.5:
                    lambdad[ndxuse[edge]] -= 2. / 3.
                notused6 = np.where(z[ndxuse] != 1. - rprs)
                if np.size(notused6) == 0:
                    muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1
                                 + 2. * u2) * (lambdad + 2. / 3.
                                 * (rprs > z)) + u2 * etad) / omega
                    return muo1
                ndxuse = ndxuse[notused6[0]]

            # # Case 10 - origin of planet hits origin of star

            origin = np.where(z[ndxuse] == 0)
            if np.size(origin) != 0:

                # # lambda_6

                lambdad[ndxuse[origin]] = -2. / 3. * (1. - rprs ** 2) \
                    ** 1.5
                notused7 = np.where(z[ndxuse] != 0)
                if np.size(notused7) == 0:
                    muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1
                                 + 2. * u2) * (lambdad + 2. / 3.
                                 * (rprs > z)) + u2 * etad) / omega
                    return muo1
                ndxuse = ndxuse[notused7[0]]

            q = np.sqrt((x2[ndxuse] - x1[ndxuse]) / (1. - x1[ndxuse]))
            n = x2[ndxuse] / x1[ndxuse] - 1.
            (Ek, Kk) = ellke(q)

            # # Case 3, Case 9 - anywhere in between
            # # lambda_2

            lambdad[ndxuse] = 2. / 9. / np.pi / np.sqrt(1. - x1[ndxuse]) \
                * ((1. - 5. * z[ndxuse] ** 2 + rprs ** 2 + x3[ndxuse]
                   ** 2) * Kk + (1. - x1[ndxuse]) * (z[ndxuse] ** 2
                   + 7. * rprs ** 2 - 4.) * Ek - 3. * x3[ndxuse]
                   / x1[ndxuse] * ellpic_bulirsch(n, q))

        muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1 + 2. * u2)
                     * (lambdad + 2. / 3. * (rprs > z)) + u2 * etad) \
            / omega
    return muo1


def ellke(k):
    """
    Computes Hasting's polynomial approximation for the complete
    elliptic integral of the first (ek) and second (kk) find.

    Parameters
    ----------
    k : 1D array
        Intermediate value from trquad().

    Returns
    -------
    ek : 1D array
        elliptic integral of the first kind
    kk : 1D array
        elliptic integral of the second kind


    Notes
    -----
    History:

    - 2008-ish Jason Eastman
        Originally written in IDL (Eastman et al. 2013, PASP 125, 83)
    - 2010-ish Kevin Stevenson
        Converted to Python
    - 2024-01-29 Kevin B Stevenson
        Modified for Eureka!
    """
    m1 = 1. - k ** 2
    logm1 = np.log(m1)

    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639
    ee1 = 1. + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ee2 = m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * -logm1
    ek = ee1 + ee2

    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012
    ek1 = a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ek2 = (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * logm1
    kk = ek1 - ek2

    return [ek, kk]


def ellpic_bulirsch(n, k):
    """
    Computes the complete elliptical integral of the third kind using
    the algorithm of Bulirsch (1965).

    Parameters
    ----------
    n : float
        An intermediate value describing the shape of the transit at a point
        in time.
    k : float
        Another intermediate value describing the shape of the transit at a
        point in time.

    Returns
    -------
    ellpic : ndarray
        The elliptical integral

    Notes
    -----
    History:

    - 2008-ish Jason Eastman
        Originally written in IDL (Eastman et al. 2013, PASP 125, 83)
    - 2010-ish Kevin Stevenson
        Converted to Python
    - 2024-01-29 Kevin B Stevenson
        Modified for Eureka!
    """
    kc = np.sqrt(1. - k ** 2)
    p = n + 1.
    m0 = 1.
    c = 1.
    p = np.sqrt(p)
    d = 1. / p
    e = kc
    while 1:
        f = c
        c = d / p + c
        g = e / p
        d = 2. * (f * g + d)
        p = g + p
        g = m0
        m0 = kc + m0
        if max(np.abs(1. - kc / g)) > 1.e-8:
            kc = 2 * np.sqrt(e)
            e = kc * m0
        else:
            return 0.5 * np.pi * (c * m0 + d) / (m0 * (m0 + p))
