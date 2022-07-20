import numpy as np
import inspect
try:
    import batman
except ImportError:
    print("Could not import batman. Functionality may be limited.")

from ..limb_darkening_fit import ld_profile
from .Model import Model
from ...lib.readEPF import Parameters

from .KeplerOrbit import KeplerOrbit
import astropy.constants as const


class BatmanTransitModel(Model):
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
        # Inherit from Model calss
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        # Check for Parameters instance
        self.parameters = kwargs.get('parameters')

        # Generate parameters from kwargs if necessary
        if self.parameters is None:
            self.parameters = Parameters(**kwargs)

        # Set parameters for multi-channel fits
        self.longparamlist = kwargs.get('longparamlist')
        self.nchan = kwargs.get('nchan')
        self.paramtitles = kwargs.get('paramtitles')

        # Store the ld_profile
        self.ld_from_S4 = kwargs.get('ld_from_S4')
        ld_func = ld_profile(self.parameters.limb_dark.value, 
                             use_gen_ld=self.ld_from_S4)
        len_params = len(inspect.signature(ld_func).parameters)
        self.coeffs = ['u{}'.format(n) for n in range(len_params)[1:]]
        
        # Replace fixed u parameters with generated limb-darkening values
        if self.ld_from_S4 is not None:
            self.ld_S4_array = kwargs.get('ld_coeffs')[len_params-2]
            for c in np.arange(self.nchan):
                for u in self.coeffs:
                    index = np.where(np.array(self.paramtitles) == u)[0]
                    if len(index) != 0:
                        item = self.longparamlist[c][index[0]]
                        param = int(item[-1])
                        if self.parameters.dict[item][1] == 'fixed':
                            self.parameters.dict[item][0] = \
                                self.ld_S4_array[c][param-1]

    def eval(self, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            The value of the model at the times self.time.
        """
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Initialize model
        bm_params = batman.TransitParams()

        # Set all parameters
        lcfinal = np.array([])
        for c in np.arange(self.nchan):
            # Set all parameters
            for index, item in enumerate(self.longparamlist[c]):
                setattr(bm_params, self.paramtitles[index],
                        self.parameters.dict[item][0])

            # Set limb darkening parameters
            uarray = []
            for u in self.coeffs:
                index = np.where(np.array(self.paramtitles) == u)[0]
                if len(index) != 0:
                    item = self.longparamlist[c][index[0]]
                    uarray.append(self.parameters.dict[item][0])
            bm_params.u = uarray

            # Enforce physicality to avoid crashes from batman by returning
            # something that should be a horrible fit
            if not ((0 < bm_params.rp) and (0 < bm_params.per) and
                    (0 < bm_params.inc < 90) and (1 < bm_params.a) and
                    (0 <= bm_params.ecc < 1) and (0 <= bm_params.w <= 360)):
                # Returning nans or infs breaks the fits, so this was the
                # best I could think of
                lcfinal = np.append(lcfinal, 1e12*np.ones_like(self.time))
                continue

            # Use batman ld_profile name
            if self.parameters.limb_dark.value == '4-parameter':
                bm_params.limb_dark = 'nonlinear'
            elif self.parameters.limb_dark.value == 'kipping2013':
                # Enforce physicality to avoid crashes from batman by
                # returning something that should be a horrible fit
                if bm_params.u[0] <= 0:
                    # Returning nans or infs breaks the fits, so this was
                    # the best I could think of
                    lcfinal = np.append(lcfinal, 1e12*np.ones_like(self.time))
                    continue
                bm_params.limb_dark = 'quadratic'
                u1 = 2*np.sqrt(bm_params.u[0])*bm_params.u[1]
                u2 = np.sqrt(bm_params.u[0])*(1-2*bm_params.u[1])
                bm_params.u = np.array([u1, u2])

            # Make the transit model
            m_transit = batman.TransitModel(bm_params, self.time,
                                            transittype='primary')

            lcfinal = np.append(lcfinal, m_transit.light_curve(bm_params))

        return lcfinal


class BatmanEclipseModel(Model):
    """Eclipse Model"""
    def __init__(self, **kwargs):
        """Initialize the transit model

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
        # Inherit from Model calss
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        # Check for Parameters instance
        self.parameters = kwargs.get('parameters')

        # Generate parameters from kwargs if necessary
        if self.parameters is None:
            self.parameters = Parameters(**kwargs)

        # Set parameters for multi-channel fits
        self.longparamlist = kwargs.get('longparamlist')
        self.nchan = kwargs.get('nchan')
        self.paramtitles = kwargs.get('paramtitles')

        log = kwargs.get('log')

        # Get the parameters relevant to light travel time correction
        ltt_params = np.array(['a', 'per', 'inc', 't0', 'ecc', 'w'])
        # Check if able to do ltt correction
        self.compute_ltt = (np.all(np.in1d(ltt_params, self.paramtitles))
                            and 'Rs' in self.parameters.dict.keys())
        if self.compute_ltt:
            # Check if we need to do ltt correction for each
            # wavelength or only one
            if self.nchan > 1:
                # Check whether the parameters are all either fixed or shared
                once_type = ['shared', 'fixed']
                self.compute_ltt_once = \
                    all([self.parameters.dict.get(name)[1] in once_type
                         for name in ltt_params])
            else:
                self.compute_ltt_once = True
        else:
            missing_params = ltt_params[~np.any(ltt_params.reshape(-1, 1) ==
                                                np.array(self.paramtitles),
                                                axis=1)]
            if 'Rs' not in self.parameters.dict.keys():
                missing_params = np.append('Rs', missing_params)
            if 't_secondary' not in self.parameters.dict.keys():
                log.writelog(f"WARNING: Missing parameters ["
                             f"{', '.join(missing_params)}] in your EPF which "
                             f"are required to account for light-travel time."
                             f"\n"
                             f"         You should either add these "
                             f"parameters, or you should be fitting for "
                             f"t_secondary\n"
                             f"         (but note that the fitted t_secondary "
                             f"will not be accounting for light-travel time).")
            else:
                log.writelog(f"WARNING: Missing parameters "
                             f"{', '.join(missing_params)} in your EPF which "
                             f"are required to account for light-travel time."
                             f"\n"
                             f"         While you are fitting for t_secondary"
                             f" which will help, note that the fitted "
                             f"t_secondary\n"
                             f"         will not be accounting for "
                             f"light-travel time).")

    def eval(self, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : ndarray
            The value of the model at the times self.time.
        """
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Initialize model
        bm_params = batman.TransitParams()

        # Set all parameters
        lcfinal = np.array([])
        for c in np.arange(self.nchan):
            # Set all parameters
            for index, item in enumerate(self.longparamlist[c]):
                setattr(bm_params, self.paramtitles[index],
                        self.parameters.dict[item][0])

            # Enforce physicality to avoid crashes from batman by
            # returning something that should be a horrible fit
            if not ((bm_params.fp < 1) and (0 < bm_params.rp) and
                    (0 < bm_params.per) and (0 < bm_params.inc < 90) and
                    (1 < bm_params.a) and (0 <= bm_params.ecc < 1) and
                    (0 <= bm_params.w <= 360)):
                # Returning nans or infs breaks the fits, so this was
                # the best I could think of
                lcfinal = np.append(lcfinal, 1e12*np.ones_like(self.time))
                continue

            bm_params.limb_dark = 'uniform'
            bm_params.u = []

            if self.compute_ltt:
                if c == 0 or not self.compute_ltt_once:
                    self.adjusted_time = correct_light_travel_time(self.time,
                                                                   bm_params)
            else:
                self.adjusted_time = self.time

            if not np.any(['t_secondary' in key
                           for key in self.longparamlist[c]]):
                # If not explicitly fitting for the time of eclipse, get
                # the time of eclipse from the time of transit, period,
                # eccentricity, and argument of periastron
                m_transit = batman.TransitModel(bm_params, self.adjusted_time,
                                                transittype='primary')
                bm_params.t_secondary = m_transit.get_t_secondary(bm_params)

            # Make the eclipse model
            m_eclipse = batman.TransitModel(bm_params, self.adjusted_time,
                                            transittype='secondary')

            lcfinal = np.append(lcfinal, m_eclipse.light_curve(bm_params))

        return lcfinal


def correct_light_travel_time(time, bm_params):
    '''Correct for the finite light travel speed.

    This function uses the KeplerOrbit.py file from the Bell_EBM package
    as that code includes a newer, faster method of solving Kepler's equation
    based on Tommasini+2018.

    Parameters
    ----------
    time : ndarray
        The times at which observations were collected
    bm_params : batman.TransitParams
        The batman TransitParams object that contains information on the orbit.

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
    a = bm_params.a * (bm_params.Rs*const.R_sun.value)

    if bm_params.ecc > 0:
        # Need to solve Kepler's equation, so use the KeplerOrbit class
        # for rapid computation. In the SPIDERMAN notation z is the radial
        # coordinate, while for Bell_EBM the radial coordinate is x
        orbit = KeplerOrbit(a=a, Porb=bm_params.per, inc=bm_params.inc,
                            t0=bm_params.t0, e=bm_params.ecc, argp=bm_params.w)
        old_x, _, _ = orbit.xyz(time)
        transit_x, _, _ = orbit.xyz(bm_params.t0)
    else:
        # No need to solve Kepler's equation for circular orbits, so save
        # some computation time
        transit_x = a*np.sin(bm_params.inc)
        old_x = transit_x*np.cos(2*np.pi*(time-bm_params.t0)/bm_params.per)

    # Get the radial distance variations of the planet
    delta_x = transit_x - old_x

    # Compute for light travel time (and convert to days)
    delta_t = (delta_x/const.c.value)/(3600.*24.)

    # Subtract light travel time as a first-order correction
    # Batman will then calculate the model at a slightly earlier time
    return time-delta_t
