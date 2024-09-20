import numpy as np
import astropy.constants as const
try:
    import spiderman as sp
except ImportError:
    print("Could not import spiderman. Functionality may be limited.")

from . import Model
from .AstroModel import PlanetParams
from ...lib.split_channels import split


class SpidermanModel(Model):
    """Eclipse/Phasecurve Model"""
    def __init__(self, **kwargs):
        """Initialize the model

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
        # Inherit from Model calss
        super().__init__(**kwargs)
        self.name = 'spiderman eclipse'

        # Define model type (physical, systematic, other)
        self.modeltype = 'physical'

        # Get the bandpass of the current channel
        self.l1 = kwargs.get('l1')
        self.l2 = kwargs.get('l2')

        # Allow an assumed blackbody stellar_model
        if not hasattr(self.parameters, 'stellar_model'):
            self.parameters.stellar_model = ['blackbody', 'independent']

    def eval(self, channel=None, pid=None, **kwargs):
        """Evaluate the function with the given values.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one of the channels. Defaults to None.
        pid : int; optional
            Planet ID, default is None which combines the models from
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

        if pid is None:
            pid_iter = range(self.num_planets)
        else:
            pid_iter = [pid,]

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Set all parameters
        lcfinal = np.array([])
        for c in np.arange(nchan):
            if self.nchannel_fitted > 1:
                chan = channels[c]
            else:
                chan = 0

            time = self.time
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time = split([time, ], self.nints, chan)[0]

            light_curve = np.ma.ones(len(time))
            for pid in pid_iter:
                # Initialize planet
                pl_params = PlanetParams(self, pid, chan)

                # Set wavelengths
                pl_params.l1 = self.l1[c]
                pl_params.l2 = self.l2[c]

                # Don't require users to explicitly specify zero planetary
                # limb darkening
                if pl_params.p_u1 is None:
                    pl_params.p_u1 = 0
                if pl_params.p_u2 is None:
                    pl_params.p_u2 = 0

                # Allow an assumed default la0 of zero
                if not hasattr(pl_params, 'la0'):
                    pl_params.la0 = 0

                # Allow users to fit R* and a/R* instead of a/R* and a
                if (not hasattr(pl_params, 'a_abs')
                        and hasattr(pl_params, 'Rs')):
                    pl_params.a_abs = (pl_params.a*pl_params.Rs *
                                       const.R_sun.value/const.au.value)

                # Enforce physicality to avoid crashes
                if not ((0 < pl_params.per) and (0 < pl_params.inc < 90) and
                        (1 < pl_params.a) and (-1 <= pl_params.ecosw <= 1) and
                        (-1 <= pl_params.esinw <= 1)):
                    # Returning nans or infs breaks the fits, so this was
                    # the best I could think of
                    light_curve = 1e6*np.ma.ones(time.shape)
                    break

                # Initialize spiderman model
                self.spider_params = sp.ModelParams(
                    brightness_model=self.parameters.brightness_model.value,
                    stellar_model=self.parameters.stellar_model.value,
                    **pl_params.__dict__)

                # Evaluate the spiderman model
                if hasattr(pl_params, 'spiderman_npts'):
                    spiderman_times = np.linspace(time.data[0],
                                                  time.data[-1],
                                                  pl_params.spiderman_npts)
                else:
                    spiderman_times = time
                lc = self.spider_params.lightcurve(spiderman_times)
                lc = np.interp(time, spiderman_times, lc)
                light_curve += lc

            lcfinal = np.ma.append(lcfinal, light_curve)

        return lcfinal
