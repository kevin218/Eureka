import numpy as np

import theano
theano.config.gcc__cxxflags += " -fexceptions"
import theano.tensor as tt

# Avoid tonnes of "Cannot construct a scalar test value" messages
import logging
logger = logging.getLogger("theano.tensor.opt")
logger.setLevel(logging.ERROR)

from . import PyMC3Model
from .AstroModel import PlanetParams, get_ecl_midpt, true_anomaly
from ...lib.split_channels import split


class SinusoidPhaseCurveModel(PyMC3Model):
    def __init__(self, **kwargs):
        """Initialize the model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.differentiable_models.PyMC3Model.__init__().
        """  # NOQA: E501
        # Inherit from PyMC3Model class
        super().__init__(**kwargs,
                         modeltype='physical',
                         name='sinusoid phase curve')

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
        else:
            lib = tt
            model = self.model

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
                    phaseVars = np.ma.ones(time.shape)
                    continue

                if pl_params.t_secondary is None:
                    # If not explicitly fitting for the time of eclipse, get
                    # the time of eclipse from the time of transit, period,
                    # eccentricity, and argument of periastron
                    pl_params.t_secondary = get_ecl_midpt(pl_params)

                # Compute orbital phase
                if pl_params.ecc == 0:
                    # the planet is on a circular orbit
                    t = time - pl_params.t_secondary
                    phi = 2*np.pi/pl_params.per*t
                else:
                    # the planet is on an eccentric orbit
                    anom = true_anomaly(pl_params, self.time, lib)
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

            lcfinal = lib.concatenate([lcfinal, phaseVars])

        return lcfinal
