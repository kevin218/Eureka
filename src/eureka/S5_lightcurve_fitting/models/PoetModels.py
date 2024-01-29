from numpy import append, array, size, zeros, where, abs, arccos, sqrt, \
    arctan, pi, log, sin, cos, tan, bitwise_and, ones_like, all, in1d, \
    ones, any
import astropy.constants as const
import inspect
import batman as bm

from .Model import Model
from .KeplerOrbit import KeplerOrbit
from ..limb_darkening_fit import ld_profile
from ...lib.split_channels import split


class TransitParams():
    """
    Define transit parameters.
    """
    def __init__(self):
        self.midpt = None
        self.rprs = None
        self.i = None
        self.ars = None
        self.period = None
        self.u = None
        self.limb_dark = None
        self.e = None
        self.omega = None
        self.fpfs = None
        self.ecl_midpt = None


class PoetTransitModel(Model):
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

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        log = kwargs.get('log')

        # Store the ld_profile
        self.ld_from_S4 = kwargs.get('ld_from_S4')
        ld_func = ld_profile(self.parameters.limb_dark.value, 
                             use_gen_ld=self.ld_from_S4)
        len_params = len(inspect.signature(ld_func).parameters)
        self.coeffs = ['u{}'.format(n) for n in range(len_params)[1:]]

        self.ld_from_file = kwargs.get('ld_from_file')
        
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
                    index = where(array(self.paramtitles) == u)[0]
                    if len(index) != 0:
                        item = self.longparamlist[c][index[0]]
                        param = int(item.split('_')[0][-1])
                        ld_val = ld_array[chan][param-1]
                        log.writelog(f"{item}, {ld_val}")
                        # Use the file value as the starting guess
                        self.parameters.dict[item][0] = ld_val
                        # In a normal prior, center at the file value
                        if (self.parameters.dict[item][-1] == 'N' and
                                self.recenter_ld_prior):
                            self.parameters.dict[item][-3] = ld_val
                        # Update the non-dictionary form as well
                        setattr(self.parameters, item,
                                self.parameters.dict[item])

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

        # Initialize model
        poet_params = TransitParams()

        # Set all parameters
        lcfinal = array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            # Set all parameters
            for index, item in enumerate(self.longparamlist[chan]):
                setattr(poet_params, self.paramtitles[index],
                        self.parameters.dict[item][0])

            # Set limb darkening parameters
            uarray = []
            for u in self.coeffs:
                index = where(array(self.paramtitles) == u)[0]
                if len(index) != 0:
                    item = self.longparamlist[chan][index[0]]
                    uarray.append(self.parameters.dict[item][0])
            poet_params.u = uarray

            # Enforce physicality to avoid crashes by returning
            # something that should be a horrible fit
            if not ((0 < poet_params.period) and (0 < poet_params.i < 90) and
                    (1 < poet_params.ars)):
                # Returning nans or infs breaks the fits, so this was the
                # best I could think of
                lcfinal = append(lcfinal, 1e12*ones_like(time))
                continue

            # Check for out-of-bound values
            if self.parameters.limb_dark.value == '4-parameter':
                # Enforce small planet approximation
                if poet_params.rprs > 0.1:
                    # Return poor fit
                    lcfinal = append(lcfinal, 1e12*ones_like(time))
                    continue
            elif self.parameters.limb_dark.value == 'kipping2013':
                # Enforce physicality to avoid crashes
                if poet_params.u[0] <= 0:
                    # Return poor fit
                    lcfinal = append(lcfinal, 1e12*ones_like(time))
                    continue
                poet_params.limb_dark = 'quadratic'
                u1 = 2*sqrt(poet_params.u[0])*poet_params.u[1]
                u2 = sqrt(poet_params.u[0])*(1-2*poet_params.u[1])
                poet_params.u = array([u1, u2])

            # Make the transit model
            m_transit = TransitModel(poet_params, time,
                                     transittype='primary')

            lcfinal = append(lcfinal, m_transit.light_curve(poet_params))

        return lcfinal


class PoetEclipseModel(Model):
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

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        log = kwargs.get('log')

        # Get the parameters relevant to light travel time correction
        ltt_params = array(['midpt', 'ars', 'i', 'period', 'e', 'omega'])
        # Check if able to do ltt correction
        self.compute_ltt = (all(in1d(ltt_params, self.paramtitles))
                            and 'Rs' in self.parameters.dict.keys())
        if self.compute_ltt:
            # Check if we need to do ltt correction for each
            # wavelength or only one
            if self.nchannel_fitted > 1:
                # Check whether the parameters are all either fixed or shared
                once_type = ['shared', 'fixed']
                self.compute_ltt_once = \
                    all([self.parameters.dict.get(name)[1] in once_type
                         for name in ltt_params])
            else:
                self.compute_ltt_once = True
        else:
            missing_params = ltt_params[~any(ltt_params.reshape(-1, 1) == 
                                             array(self.paramtitles), axis=1)]
            if 'Rs' not in self.parameters.dict.keys():
                missing_params = append('Rs', missing_params)
            if 'ecl_midpt' not in self.parameters.dict.keys():
                log.writelog(f"WARNING: Missing parameters ["
                             f"{', '.join(missing_params)}] in your EPF which "
                             f"are required to account for light-travel time."
                             f"\n"
                             f"         You should either add these "
                             f"parameters, or you should be fitting for "
                             f"ecl_midpt\n"
                             f"         (but note that the fitted ecl_midpt "
                             f"will not be accounting for light-travel time).")
            else:
                log.writelog(f"WARNING: Missing parameters "
                             f"{', '.join(missing_params)} in your EPF which "
                             f"are required to account for light-travel time."
                             f"\n"
                             f"         While you are fitting for ecl_midpt"
                             f" which will help, note that the fitted "
                             f"ecl_midpt\n"
                             f"         will not be accounting for "
                             f"light-travel time).")

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

        # Initialize model
        poet_params = TransitParams()

        # Set all parameters
        lcfinal = array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            # Set all parameters
            for index, item in enumerate(self.longparamlist[chan]):
                setattr(poet_params, self.paramtitles[index],
                        self.parameters.dict[item][0])

            # Enforce physicality to avoid crashes by returning
            # something that should be a horrible fit
            if not ((0 < poet_params.period) and (0 < poet_params.i < 90) and
                    (1 < poet_params.ars) and (0 <= poet_params.e < 1) and
                    (0 <= poet_params.omega <= 360)):
                # Returning nans or infs breaks the fits, so this was the
                # best I could think of
                lcfinal = append(lcfinal, 1e12*ones_like(time))
                continue

            poet_params.limb_dark = 'uniform'
            poet_params.u = []

            # Compute light travel time
            if self.compute_ltt:
                if c == 0 or not self.compute_ltt_once:
                    self.adjusted_time = correct_light_travel_time(time,
                                                                   poet_params)
            else:
                self.adjusted_time = time

            if not any(['ecl_midpt' in key
                        for key in self.longparamlist[chan]]):
                # If not explicitly fitting for the time of eclipse, get
                # the time of eclipse from the time of transit, period,
                # eccentricity, and argument of periastron
                poet_params.ecl_midpt = get_ecl_midpt(poet_params)

            # Make the eclipse model
            m_eclipse = TransitModel(poet_params, self.adjusted_time,
                                     transittype='secondary')

            lcfinal = append(lcfinal, m_eclipse.light_curve(poet_params))

        return lcfinal
   

class TransitModel():
    """
    Class for generating model transit light curves.
    """
    def __init__(self, params, t, transittype="primary"):
        # Initializes model parameters
        self.t = t
        self.midpt = params.midpt
        self.rprs = params.rprs
        self.i = params.i
        self.ars = params.ars
        self.period = params.period
        self.u = params.u
        self.limb_dark = params.limb_dark
        self.transittype = transittype
        self.nthreads = 4

        # Handles the case of inverse transits (rp < 0)
        self.inverse = False
        if params.rprs < 0.:
            self.inverse = True

        # Compute distance, z, of planet and star midpoints
        self.z = self.ars \
            * sqrt(sin(2 * pi * (t - self.midpt) / self.period) ** 2 
                   + (cos(self.i * pi / 180) * cos(2 * pi * (t - self.midpt)
                      / self.period)) ** 2)

        # Ignore close approach near secondary eclipse
        self.z[where(bitwise_and((t - self.midpt) % self.period
               > self.period / 4., (t - self.midpt) % self.period
               < self.period * 3. / 4))] = self.ars

    def light_curve(self, params):
        """
        Calculate a model light curve.
        """
        # Update transit params
        self.midpt = params.midpt
        self.rprs = params.rprs
        self.i = params.i
        self.ars = params.ars
        self.period = params.period
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
            elif self.limb_dark == "4-parameter":
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
            lc = bm._eclipse._eclipse(self.z, abs(params.rprs), 
                                      params.fpfs, self.nthreads)
        return lc
    

def get_ecl_midpt(params):
    """
    Return the time of secondary eclipse center.
    """

    # Start with primary transit
    TA = pi / 2. - params.omega * pi / 180.
    E = 2. * arctan(sqrt((1. - params.e) / (1. + params.e)) * tan(TA
                    / 2.))
    M = E - params.e * sin(E)
    phase_tr = M / 2. / pi

    # Now do secondary eclipse
    TA = 3. * pi / 2. - params.omega * pi / 180.
    E = 2. * arctan(sqrt((1. - params.e) / (1. + params.e)) * tan(TA
                    / 2.))
    M = E - params.e * sin(E)
    phase_ecl = M / 2. / pi

    return params.midpt + params.period * (phase_ecl - phase_tr)


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

    Revisions
    ---------
    2010-11-27      Kevin Stevenson, UCF
                    Original version
    2024-01-28      Kevin Stevenson, APL
                    Updated for Eureka!
    """
    # INGRESS/EGRESS INDICES
    iingress = where(bitwise_and((1-rprs) < z, z <= (1+rprs)))[0]
    # COMPUTE k0 & k1
    k0 = arccos((rprs**2 + z[iingress]**2 - 1) / 2 / rprs / z[iingress])
    k1 = arccos((1 - rprs**2 + z[iingress]**2) / 2 / z[iingress])

    # CALCULATE TRANSIT SHAPE
    # Baseline
    y = ones(len(z))
    # Full transit
    y[where(z <= (1-rprs))] = 1.-rprs**2
    # Ingress/egress
    y[iingress] = 1. - 1./pi*(k0*rprs**2 + k1 - sqrt((4*z[iingress]**2
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

    Revisions
    ---------
    2010-12-15      Kevin Stevenson, UCF
                    Converted to Python
    2024-01-28      Kevin Stevenson, APL
                    Updated for Eureka!
    """

    # DEFINE PARAMETERS
    (u1, u2, u3, u4) = u
    Sigma4 = 1. - u1 / 5. - u2 / 3. - 3. * u3 / 7. - u4 / 2.

    # CALCULATE TRANSIT SHAPE WITH LIMB-DARKENING
    y = ones(len(z), dtype=float)
    if rprs == 0:
        return y

    # INGRESS/EGRESS
    iingress = where(bitwise_and(1 - rprs < z, z <= 1 + rprs))[0]
    x = 1. - (z[iingress] - rprs) ** 2
    I1star = 1. - u1 * (1. - 4. / 5. * sqrt(sqrt(x))) \
                - u2 * (1. - 2. / 3. * sqrt(x)) \
                - u3 * (1. - 4. / 7. * sqrt(sqrt(x * x * x))) \
                - u4 * (1. - 4. / 8. * x)
    y[iingress] = 1. - I1star \
        * (rprs ** 2 * arccos((z[iingress] - 1.) / rprs) - (z[iingress] - 1.)
           * sqrt(rprs ** 2 - (z[iingress] - 1.) ** 2)) / pi / Sigma4

    # Full transit (except @ z=0)
    itrans = where(bitwise_and(z <= 1 - rprs, z != 0.))
    sig1 = sqrt(sqrt(1. - (z[itrans] - rprs) ** 2))
    sig2 = sqrt(sqrt(1. - (z[itrans] + rprs) ** 2))
    I2star = 1. \
        - u1 * (1. + (sig2 ** 5 - sig1 ** 5) / 5. / rprs / z[itrans]) \
        - u2 * (1. + (sig2 ** 6 - sig1 ** 6) / 6. / rprs / z[itrans]) \
        - u3 * (1. + (sig2 ** 7 - sig1 ** 7) / 7. / rprs / z[itrans]) \
        - u4 * (rprs ** 2 + z[itrans] ** 2)
    y[itrans] = 1. - rprs ** 2 * I2star / Sigma4

    # z=0 (midpoint)
    y[where(z == 0.)] = 1. - rprs ** 2 / Sigma4

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

    Revisions
    ---------
    2012-08-13	    Kevin Stevenson, UChicago
                    Modified from Jason Eastman's version
    2024-01-28      Kevin Stevenson, APL
                    Updated for Eureka!
    '''

    nz = size(z)
    lambdad = zeros(nz)
    etad = zeros(nz)
    lambdae = zeros(nz)
    omega = 1. - u1 / 3. - u2 / 6.

    # # tolerance for double precision equalities
    # # special case integrations

    tol = 1e-14

    rprs = abs(rprs)

    z = where(abs(rprs - z) < tol, rprs, z)
    z = where(abs(rprs - 1 - z) < tol, rprs - 1., z)
    z = where(abs(1 - rprs - z) < tol, 1. - rprs, z)
    z = where(z < tol, 0., z)

    x1 = (rprs - z) ** 2.
    x2 = (rprs + z) ** 2.
    x3 = rprs ** 2. - z ** 2.

    # # trivial case of no planet

    if rprs <= 0.:
        muo1 = zeros(nz) + 1.
        return muo1

    # # Case 1 - the star is unocculted:
    # # only consider points with z < 1+rprs

    notusedyet = where(z < 1. + rprs)
    notusedyet = notusedyet[0]
    if size(notusedyet) == 0:
        muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1 + 2. * u2)
                     * (lambdad + 2. / 3. * (rprs > z)) + u2 * etad) \
            / omega
        return muo1

    # Case 11 - the source is completely occulted:

    if rprs >= 1.:
        occulted = where(z[notusedyet] <= rprs - 1.)  # ,complement=notused2)
        if size(occulted) != 0:
            ndxuse = notusedyet[occulted]
            etad[ndxuse] = 0.5  # corrected typo in paper
            lambdae[ndxuse] = 1.

            # lambdad = 0 already

            notused2 = where(z[notusedyet] > rprs - 1)
            if size(notused2) == 0:
                muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1 + 2.
                             * u2) * (lambdad + 2. / 3. * (rprs > z))
                             + u2 * etad) / omega
                return muo1
            notusedyet = notusedyet[notused2]

    # Case 2, 7, 8 - ingress/egress (uniform disk only)

    inegressuni = where((z[notusedyet] >= abs(1. - rprs))
                        & (z[notusedyet] < 1. + rprs))
    if size(inegressuni) != 0:
        ndxuse = notusedyet[inegressuni]
        tmp = (1. - rprs ** 2. + z[ndxuse] ** 2.) / 2. / z[ndxuse]
        tmp = where(tmp > 1., 1., tmp)
        tmp = where(tmp < -1., -1., tmp)
        kap1 = arccos(tmp)
        tmp = (rprs ** 2. + z[ndxuse] ** 2 - 1.) / 2. / rprs / z[ndxuse]
        tmp = where(tmp > 1., 1., tmp)
        tmp = where(tmp < -1., -1., tmp)
        kap0 = arccos(tmp)
        tmp = 4. * z[ndxuse] ** 2 - (1. + z[ndxuse] ** 2 - rprs ** 2) \
            ** 2
        tmp = where(tmp < 0, 0, tmp)
        lambdae[ndxuse] = (rprs ** 2 * kap0 + kap1 - 0.5 * sqrt(tmp)) \
            / pi

        # eta_1

        etad[ndxuse] = 1. / 2. / pi \
            * (kap1 + rprs ** 2 * (rprs ** 2 + 2. * z[ndxuse] ** 2) 
               * kap0 - (1. + 5. * rprs ** 2 + z[ndxuse] ** 2) 
               / 4. * sqrt((1. - x1[ndxuse]) * (x2[ndxuse] - 1.)))

    # Case 5, 6, 7 - the edge of planet lies at origin of star

    ocltor = where(z[notusedyet] == rprs)
    if size(ocltor) != 0:
        ndxuse = notusedyet[ocltor]
        if rprs < 0.5:

            # Case 5

            q = 2. * rprs  # corrected typo in paper (2k -> 2rprs)
            (Ek, Kk) = ellke(q)

            # lambda_4

            lambdad[ndxuse] = 1. / 3. + 2. / 9. / pi \
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

            lambdad[ndxuse] = 1. / 3. + 16. * rprs / 9. / pi \
                * (2. * rprs ** 2 - 1.) * Ek \
                - (32. * rprs ** 4 - 20. * rprs ** 2 + 3.) \
                / 9. / pi / rprs * Kk

        else:
            # etad = eta_1 already
            # Case 6

            lambdad[ndxuse] = 1. / 3. - 4. / pi / 9.
            etad[ndxuse] = 3. / 32.
        notused3 = where(z[notusedyet] != rprs)
        if size(notused3) == 0:
            muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1 + 2. * u2)
                         * (lambdad + 2. / 3. * (rprs > z)) + u2
                         * etad) / omega
            return muo1
        notusedyet = notusedyet[notused3]

    # Case 2, Case 8 - ingress/egress (with limb darkening)

    inegress = where((z[notusedyet] > 0.5 + abs(rprs - 0.5))
                     & (z[notusedyet] < 1. + rprs) | (rprs > 0.5)
                     & (z[notusedyet] > abs(1. - rprs))
                     & (z[notusedyet] < rprs))  # , complement=notused4)
    if size(inegress) != 0:
        ndxuse = notusedyet[inegress]
        q = sqrt((1. - x1[ndxuse]) / (x2[ndxuse] - x1[ndxuse]))
        (Ek, Kk) = ellke(q)
        n = 1. / x1[ndxuse] - 1.

        # lambda_1:

        lambdad[ndxuse] = 2. / 9. / pi / sqrt(x2[ndxuse] - x1[ndxuse]) \
            * (((1. - x2[ndxuse]) * (2. * x2[ndxuse] + x1[ndxuse] - 3.)
               - 3. * x3[ndxuse] * (x2[ndxuse] - 2.)) * Kk
               + (x2[ndxuse] - x1[ndxuse]) * (z[ndxuse] ** 2 + 7.
               * rprs ** 2 - 4.) * Ek - 3. * x3[ndxuse] / x1[ndxuse]
               * ellpic_bulirsch(n, q))
        notused4 = where(
            ((z[notusedyet] <= 0.5 + abs(rprs - 0.5)) | 
             (z[notusedyet] >= 1.0 + rprs))
            & ((rprs <= 0.5) | (z[notusedyet] <= abs(1.0 - rprs)) | 
               (z[notusedyet] >= rprs)))

        if size(notused4) == 0:
            muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1 + 2. * u2)
                         * (lambdad + 2. / 3. * (rprs > z)) + u2
                         * etad) / omega
            return muo1
        notusedyet = notusedyet[notused4]

    # Case 3, 4, 9, 10 - planet completely inside star

    if rprs < 1.:
        inside = where(z[notusedyet] <= 1. - rprs)  # , complement=notused5)
        if size(inside) != 0:
            ndxuse = notusedyet[inside]

            # # eta_2

            etad[ndxuse] = rprs ** 2 / 2. * (rprs ** 2 + 2. * z[ndxuse] ** 2)

            # # uniform disk

            lambdae[ndxuse] = rprs ** 2

            # # Case 4 - edge of planet hits edge of star

            edge = where(z[ndxuse] == 1. - rprs)  # , complement=notused6)
            if size(edge[0]) != 0:

                # # lambda_5

                lambdad[ndxuse[edge]] = 2. / 3. / pi \
                    * arccos(1. - 2. * rprs) \
                    - 4. / 9. / pi * sqrt(rprs * (1. - rprs)) \
                    * (3. + 2. * rprs - 8. * rprs ** 2)
                if rprs > 0.5:
                    lambdad[ndxuse[edge]] -= 2. / 3.
                notused6 = where(z[ndxuse] != 1. - rprs)
                if size(notused6) == 0:
                    muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1
                                 + 2. * u2) * (lambdad + 2. / 3.
                                 * (rprs > z)) + u2 * etad) / omega
                    return muo1
                ndxuse = ndxuse[notused6[0]]

            # # Case 10 - origin of planet hits origin of star

            origin = where(z[ndxuse] == 0)  # , complement=notused7)
            if size(origin) != 0:

                # # lambda_6

                lambdad[ndxuse[origin]] = -2. / 3. * (1. - rprs ** 2) \
                    ** 1.5
                notused7 = where(z[ndxuse] != 0)
                if size(notused7) == 0:
                    muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1
                                 + 2. * u2) * (lambdad + 2. / 3.
                                 * (rprs > z)) + u2 * etad) / omega
                    return muo1
                ndxuse = ndxuse[notused7[0]]

            q = sqrt((x2[ndxuse] - x1[ndxuse]) / (1. - x1[ndxuse]))
            n = x2[ndxuse] / x1[ndxuse] - 1.
            (Ek, Kk) = ellke(q)

            # # Case 3, Case 9 - anywhere in between
            # # lambda_2

            lambdad[ndxuse] = 2. / 9. / pi / sqrt(1. - x1[ndxuse]) \
                * ((1. - 5. * z[ndxuse] ** 2 + rprs ** 2 + x3[ndxuse]
                   ** 2) * Kk + (1. - x1[ndxuse]) * (z[ndxuse] ** 2
                   + 7. * rprs ** 2 - 4.) * Ek - 3. * x3[ndxuse]
                   / x1[ndxuse] * ellpic_bulirsch(n, q))

        muo1 = 1. - ((1. - u1 - 2. * u2) * lambdae + (u1 + 2. * u2)
                     * (lambdad + 2. / 3. * (rprs > z)) + u2 * etad) \
            / omega
    return muo1


def ellke(k):
    m1 = 1. - k ** 2
    logm1 = log(m1)

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
    kc = sqrt(1. - k ** 2)
    p = n + 1.
    m0 = 1.
    c = 1.
    p = sqrt(p)
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
        if max(abs(1. - kc / g)) > 1.e-8:
            kc = 2 * sqrt(e)
            e = kc * m0
        else:
            return 0.5 * pi * (c * m0 + d) / (m0 * (m0 + p))


def correct_light_travel_time(time, poet_params):
    '''Correct for the finite light travel speed.

    This function uses the KeplerOrbit.py file from the Bell_EBM package
    as that code includes a newer, faster method of solving Kepler's equation
    based on Tommasini+2018.

    Parameters
    ----------
    time : ndarray
        The times at which observations were collected
    poet_params : poet.TransitParams
        The POET TransitParams object that contains information on the orbit.

    Returns
    -------
    time : ndarray
        Updated times that can be put into POET transit and eclipse functions
        that will give the expected results assuming a finite light travel
        speed.

    Notes
    -----
    History:

    - 2022-03-31 Taylor J Bell
        Initial version based on the Bell_EMB KeplerOrbit.py file by
        Taylor J Bell and the light travel time calculations of SPIDERMAN's
        web.c file by Tom Louden
    - 2024-01-29 Kevin B Stevenson
        Modified for POET eclipses
    '''
    # Need to convert from a/Rs to a in meters
    a = poet_params.ars * (poet_params.Rs*const.R_sun.value)

    if poet_params.e > 0:
        # Need to solve Kepler's equation, so use the KeplerOrbit class
        # for rapid computation. In the SPIDERMAN notation z is the radial
        # coordinate, while for Bell_EBM the radial coordinate is x
        orbit = KeplerOrbit(a=a, Porb=poet_params.period, inc=poet_params.i,
                            t0=poet_params.midpt, e=poet_params.e, 
                            argp=poet_params.omega)
        old_x, _, _ = orbit.xyz(time)
        transit_x, _, _ = orbit.xyz(poet_params.midpt)
    else:
        # No need to solve Kepler's equation for circular orbits, so save
        # some computation time
        transit_x = a*sin(poet_params.i)
        old_x = transit_x*cos(2*pi*(time-poet_params.midpt)/poet_params.period)

    # Get the radial distance variations of the planet
    delta_x = transit_x - old_x

    # Compute for light travel time (and convert to days)
    delta_t = (delta_x/const.c.value)/(3600.*24.)

    # Subtract light travel time as a first-order correction
    # POET will then calculate the model at a slightly earlier time
    return time-delta_t.flatten()
