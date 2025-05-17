import numpy as np
from astropy import constants as const
import inspect
import jax.numpy as jnp

from jaxoplanet.orbits.keplerian import Central, System, Body
from jaxoplanet.light_curves import limb_dark_light_curve

from . import JaxModel
from .AstroModel import PlanetParams
from ..limb_darkening_fit import ld_profile
from ...lib.split_channels import split


class JaxoplanetModel(JaxModel):
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
        self.name = 'jaxoplanet transit'

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        log = kwargs.get('log')

        # Store the ld_profile
        self.ld_from_S4 = kwargs.get('ld_from_S4')
        ld_func = ld_profile(self.parameters.limb_dark.value,
                             use_gen_ld=self.ld_from_S4)
        len_params = len(inspect.signature(ld_func).parameters)
        self.coeffs = ['u{}'.format(n) for n in range(1, len_params)]

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

    def setup(self, newparams):
        """Setup a model for evaluation and fitting.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        """
        self.systems = []
        for chan in range(self.nchannel_fitted):
            # Initialize PlanetParams object
            pl_params = PlanetParams(self, 0, chan, eval=False)

            # Initialize star object
            star = Central(radius=pl_params.Rs, mass=0.)
            # Instantiate the system
            system = System(star)

            if pl_params.limb_dark not in ['uniform', 'linear', 'quadratic',
                                           'kipping2013']:
                message = (f'ERROR: Our JaxoplanetModel is not yet able to '
                           f'handle {self.parameters.limb_dark.value} '
                           f'limb darkening.\n'
                           f'       limb_dark must be one of uniform, '
                           f'linear, quadratic, or kipping2013.')
                raise ValueError(message)

            # Setup each planet
            planets = []
            for pid in range(self.num_planets):
                # Initialize PlanetParams object for this planet
                pl_params = PlanetParams(self, pid, chan, eval=False)

                # Solve Keplerian orbital period equation for system mass
                # (otherwise jaxoplanet is going to mess with P or a...)
                a = pl_params.a*pl_params.Rs*const.R_sun.value
                p = pl_params.per*(24*3600)
                Mp = ((2*np.pi*a**(3/2))/p)**2/const.G.value/const.M_sun.value

                # Initialize planet object
                planet = Body(
                    mass=Mp,
                    # Convert radius ratio to R_star units
                    radius=pl_params.rp*pl_params.Rs,
                    semimajor=pl_params.a*pl_params.Rs,
                    inclination=pl_params.inc*np.pi/180,
                    # period=pl_params.per,
                    time_transit=pl_params.t0,
                    eccentricity=pl_params.ecc,
                    omega_peri=pl_params.w*np.pi/180,
                )
                planets.append(planet)
                system = system.add_body(planet)
            self.systems.append(system)

        self.update(newparams)

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

        # Can't separately evaluate jaxoplanet models to allow for
        # mutual occultations
        # pid_iter = range(self.num_planets)

        if eval:
            lib = np
            systems = self.fit.systems
        else:
            lib = jnp
            systems = self.systems

        lcfinal = lib.array([])
        for c in range(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            pl_params = PlanetParams(self, 0, chan, eval=eval)
            system = systems[chan]

            light_curve = 1. + limb_dark_light_curve(system, pl_params.u)(time)

            lcfinal = lib.append(lcfinal, light_curve)

        return lcfinal

    def update(self, newparams, **kwargs):
        """Update parameters and update the self.fit.systems list.

        Parameters
        ----------
        newparams : ndarray
            New parameter values.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.differentiable_models.PyMC3Model.update().
        """
        super().update(newparams, **kwargs)

        self.fit.systems = []
        for chan in range(self.nchannel_fitted):
            # Initialize PlanetParams object
            pl_params = PlanetParams(self, 0, chan, eval=True)

            # Initialize star object
            star = Central(radius=pl_params.Rs, mass=0.)
            # Instantiate the system
            system = System(star)

            if pl_params.limb_dark not in ['uniform', 'linear', 'quadratic',
                                           'kipping2013']:
                message = (f'ERROR: Our JaxoplanetModel is not yet able to '
                           f'handle {self.parameters.limb_dark.value} '
                           f'limb darkening.\n'
                           f'       limb_dark must be one of uniform, '
                           f'linear, quadratic, or kipping2013.')
                raise ValueError(message)

            # Setup each planet
            planets = []
            for pid in range(self.num_planets):
                # Initialize PlanetParams object for this planet
                pl_params = PlanetParams(self, pid, chan, eval=True)

                # Solve Keplerian orbital period equation for system mass
                # (otherwise jaxoplanet is going to mess with P or a...)
                a = pl_params.a*pl_params.Rs*const.R_sun.value
                p = pl_params.per*(24*3600)
                Mp = ((2*np.pi*a**(3/2))/p)**2/const.G.value/const.M_sun.value

                # Initialize planet object
                planet = Body(
                    mass=Mp,
                    # Convert radius ratio to R_star units
                    radius=pl_params.rp*pl_params.Rs,
                    semimajor=pl_params.a*pl_params.Rs,
                    inclination=pl_params.inc*np.pi/180,
                    # period=pl_params.per,
                    time_transit=pl_params.t0,
                    eccentricity=pl_params.ecc,
                    omega_peri=pl_params.w*np.pi/180,
                )
                planets.append(planet)
                system = system.add_body(planet)

            self.fit.systems.append(system)
