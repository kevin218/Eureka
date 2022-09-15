import numpy as np
import astropy.constants as const
try:
    import spiderman as sp
except ImportError:
    print("Could not import spiderman. Functionality may be limited.")

from .Model import Model
from ...lib.readEPF import Parameters


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

        self.l1 = kwargs.get('l1')
        self.l2 = kwargs.get('l2')

        # Allow an assumed blackbody stellar_model
        if not hasattr(self.parameters, 'stellar_model'):
            self.parameters.stellar_model = ['blackbody', 'independent']
        
        # Initialize model
        self.spider_params = sp.ModelParams(
            brightness_model=self.parameters.brightness_model.value,
            stellar_model=self.parameters.stellar_model.value)

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

        # Set all parameters
        lcfinal = np.array([])
        for c in np.arange(self.nchan):
            # Set all parameters
            for index, item in enumerate(self.longparamlist[c]):
                setattr(self.spider_params, self.paramtitles[index],
                        self.parameters.dict[item][0])

            # Set wavelengths
            self.spider_params.l1 = self.l1[c]
            self.spider_params.l2 = self.l2[c]

            # Don't require users to explicitly specify zero planetary
            # limb darkening
            if self.spider_params.p_u1 is None:
                self.spider_params.p_u1 = 0
            if self.spider_params.p_u2 is None:
                self.spider_params.p_u2 = 0

            # Allow an assumed default la0 of zero
            if not hasattr(self.spider_params, 'la0'):
                self.spider_params.la0 = 0
            
            # Allow users to fit R* and a/R* instead of a/R* and a
            if (not hasattr(self.parameters, 'a_abs')
                    and hasattr(self.parameters, 'Rs')):
                self.spider_params.a_abs = (self.parameters.a.value *
                                            self.parameters.Rs.value *
                                            const.R_sun.value /
                                            const.au.value)

            # Enforce physicality to avoid crashes from spiderman by
            # returning something that should be a horrible fit
            if not ((0 < self.spider_params.rp) and
                    (0 < self.spider_params.per) and
                    (0 < self.spider_params.inc < 90) and
                    (1 < self.spider_params.a) and
                    (0 <= self.spider_params.ecc < 1) and
                    (0 <= self.spider_params.w <= 360)):
                # Returning nans or infs breaks the fits, so this was
                # the best I could think of
                lcfinal = np.append(lcfinal, 1e12*np.ones_like(self.time))
                continue

            # Evaluate the eclipse model
            lcfinal = np.append(lcfinal,
                                self.spider_params.lightcurve(self.time))

        return lcfinal
