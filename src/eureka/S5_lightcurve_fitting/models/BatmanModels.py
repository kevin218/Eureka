import numpy as np
import astropy.constants as const
import inspect
try:
    import batman
except ImportError:
    print("Could not import batman. Functionality may be limited.")

from .Model import Model
from .KeplerOrbit import KeplerOrbit
from ..limb_darkening_fit import ld_profile
from ...lib.split_channels import split


class PlanetParams():
    """
    Define planet parameters.
    """
    def __init__(self, model, pid):
        # Planet ID
        self.pid = pid       
        # Set transit/eclipse parameters
        self.t0 = None
        self.rp = None
        self.inc = None
        self.a = None
        self.per = None
        self.ecc = None
        self.w = None
        self.fp = None
        self.t_secondary = None
        self.AmpCos1 = 0.
        self.AmpSin1 = 0.
        self.AmpCos2 = 0.
        self.AmpSin2 = 0.
        for item in self.__dict__.keys():
            if pid > 0:
                item0 = item + str(pid)
            else:
                item0 = item
            try:
                setattr(self, item, model.parameters.dict[item0][0])
            except:
                pass
        # Allow for rp or rprs
        if (self.rp is None) and ('rprs' in model.parameters.dict.keys()):
            if pid > 0:
                item0 = 'rprs' + str(pid)
            else:
                item0 = 'rprs'
            setattr(self, 'rp', model.parameters.dict[item0][0])
        # Allow for a or ars
        if (self.a is None) and ('ars' in model.parameters.dict.keys()):
            if pid > 0:
                item0 = 'ars' + str(pid)
            else:
                item0 = 'ars'
            setattr(self, 'a', model.parameters.dict[item0][0])
        # Allow for fp or fpfs
        if (self.fp is None) and ('fpfs' in model.parameters.dict.keys()):
            if pid > 0:
                item0 = 'fpfs' + str(pid)
            else:
                item0 = 'fpfs'
            setattr(self, 'fp', model.parameters.dict[item0][0])
        # Set stellar radius
        if 'Rs' in model.parameters.dict.keys():
            setattr(self, 'Rs', model.parameters.dict['Rs'][0])


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
        # Inherit from Model class
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        log = kwargs.get('log')

        # Set default to ignore light-travel correction if not specified
        if not hasattr(self, 'compute_ltt') or self.compute_ltt is None:
            self.compute_ltt = False
        
        # Get the parameters relevant to light travel time correction
        ltt_params = np.array(['per', 'inc', 't0', 'ecc', 'w'])
        ltt_par2 = np.array(['a', 'ars'])
        # Check if able to do ltt correction
        ltt_params_present = (np.all(np.in1d(ltt_params, self.paramtitles))
                              and 'Rs' in self.parameters.dict.keys()
                              and np.any(np.in1d(ltt_par2, self.paramtitles)))
        if self.compute_ltt and not ltt_params_present:
            missing_params = ltt_params[~np.any(ltt_params.reshape(-1, 1) ==
                                                np.array(self.paramtitles),
                                                axis=1)]
            if 'Rs' not in self.parameters.dict.keys():
                missing_params = np.append('Rs', missing_params)
            if ('a' not in self.parameters.dict.keys()) and \
                    ('ars' not in self.parameters.dict.keys()):
                missing_params = np.append('a', missing_params)

            log.writelog(f"WARNING: Missing parameters ["
                         f"{', '.join(missing_params)}] in your EPF which "
                         "are required to account for light-travel time.\n"
                         "         You should either add these "
                         "parameters, or you should set compute_ltt to "
                         "False in your ECF.\n"
                         "         Setting compute_ltt to False for now!")
            self.compute_ltt = False

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
                    index = np.where(np.array(self.paramtitles) == u)[0]
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

        # Set all parameters
        lcfinal = np.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            light_curve = np.ones_like(time)
            for pid in range(self.num_planets):
                # Initialize planet
                bm_params = PlanetParams(self, pid)

                # Set limb darkening parameters
                uarray = []
                for u in self.coeffs:
                    index = np.where(np.array(self.paramtitles) == u)[0]
                    if len(index) != 0:
                        item = self.longparamlist[chan][index[0]]
                        uarray.append(self.parameters.dict[item][0])
                bm_params.u = uarray
                bm_params.limb_dark = self.parameters.dict['limb_dark'][0]

                # Enforce physicality to avoid crashes from batman by returning
                # something that should be a horrible fit
                if not ((0 < bm_params.per) and (0 < bm_params.inc < 90) and
                        (1 < bm_params.a) and (0 <= bm_params.ecc < 1) and
                        (0 <= bm_params.w <= 360)):
                    # Returning nans or infs breaks the fits, so this was the
                    # best I could think of
                    light_curve = 1e12*np.ones_like(time)
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
                        light_curve = 1e12*np.ones_like(time)
                        continue
                    bm_params.limb_dark = 'quadratic'
                    u1 = 2*np.sqrt(bm_params.u[0])*bm_params.u[1]
                    u2 = np.sqrt(bm_params.u[0])*(1-2*bm_params.u[1])
                    bm_params.u = np.array([u1, u2])

                if self.compute_ltt:
                    self.adjusted_time = correct_light_travel_time(time, 
                                                                   bm_params)
                else:
                    self.adjusted_time = time

                # Make the transit model
                m_transit = batman.TransitModel(bm_params, self.adjusted_time,
                                                transittype='primary')
                light_curve *= m_transit.light_curve(bm_params)

            lcfinal = np.append(lcfinal, light_curve)

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

        log = kwargs.get('log')

        # Set default to turn light-travel correction on if not specified
        if not hasattr(self, 'compute_ltt') or self.compute_ltt is None:
            self.compute_ltt = True

        # Get the parameters relevant to light travel time correction
        ltt_params = np.array(['per', 'inc', 't0', 'ecc', 'w'])
        ltt_par2 = np.array(['a', 'ars'])
        # Check if able to do ltt correction
        ltt_params_present = (np.all(np.in1d(ltt_params, self.paramtitles))
                              and 'Rs' in self.parameters.dict.keys()
                              and np.any(np.in1d(ltt_par2, self.paramtitles)))
        if self.compute_ltt and not ltt_params_present:
            missing_params = ltt_params[~np.any(ltt_params.reshape(-1, 1) ==
                                                np.array(self.paramtitles),
                                                axis=1)]
            if 'Rs' not in self.parameters.dict.keys():
                missing_params = np.append('Rs', missing_params)
            if ('a' not in self.parameters.dict.keys()) and \
                    ('ars' not in self.parameters.dict.keys()):
                missing_params = np.append('a', missing_params)

            log.writelog("WARNING: Missing parameters ["
                         f"{', '.join(missing_params)}] in your EPF which "
                         "are required to account for light-travel time.\n")

            if 't_secondary' not in self.parameters.dict.keys():
                log.writelog("         You should either add these parameters,"
                             " fit for t_secondary (but note that the\n"
                             "         fitted t_secondary value will not have "
                             "accounted for light-travel time), or you\n"
                             "         should set compute_ltt to False in your"
                             " ECF.")
            else:
                log.writelog("         While you are fitting for t_secondary "
                             "which will help, note that the fitted\n"
                             "         t_secondary value will not have "
                             "accounted for light-travel time. You should\n"
                             "         either add the missing parameters or "
                             "set compute_ltt to False in your ECF.")

            log.writelog("         Setting compute_ltt to False for now!")
            self.compute_ltt = False

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
        lcfinal = np.array([])
        self.adjusted_time = []
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            light_curve = np.ones_like(time)
            for pid in range(self.num_planets):
                # Initialize planet
                bm_params = PlanetParams(self, pid)

                # Set limb darkening parameters
                bm_params.u = []
                bm_params.limb_dark = 'uniform'

                # Enforce physicality to avoid crashes from batman by
                # returning something that should be a horrible fit
                if not ((0 < bm_params.per) and (0 < bm_params.inc < 90) and
                        (1 < bm_params.a) and (0 <= bm_params.ecc < 1) and
                        (0 <= bm_params.w <= 360)):
                    # Returning nans or infs breaks the fits, so this was
                    # the best I could think of
                    light_curve = 1e12*np.ones_like(time)
                    continue

                # Compute light travel time
                if self.compute_ltt and (c == 0):
                    self.adjusted_time.append(
                        correct_light_travel_time(time, bm_params))
                elif c == 0:
                    self.adjusted_time.append(time)

                if not np.any(['t_secondary' in key
                              for key in self.longparamlist[chan]]):
                    # If not explicitly fitting for the time of eclipse, get
                    # the time of eclipse from the time of transit, period,
                    # eccentricity, and argument of periastron
                    m_transit = batman.TransitModel(bm_params, 
                                                    self.adjusted_time[pid],
                                                    transittype='primary')
                    bm_params.t_secondary = m_transit.get_t_secondary(bm_params)

                # Make the eclipse model
                m_eclipse = batman.TransitModel(bm_params, 
                                                self.adjusted_time[pid],
                                                transittype='secondary')
                light_curve *= m_eclipse.light_curve(bm_params)

            lcfinal = np.append(lcfinal, light_curve)

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
    return time-delta_t.flatten()
