import numpy as np

from .jax_models import JaxModel
from . import lightcurve


class LightCurve(JaxModel, lightcurve.LightCurve):
    def __init__(self, time, flux, channel, nchannel, log, longparamlist,
                 parameters, freenames, unc=None, time_units='BJD',
                 name='My Light Curve', share=False, white=False,
                 multwhite=False, nints=[], **kwargs):
        """
        A class to store the actual light curve

        Parameters
        ----------
        time : sequence
            The time axis in days.
        flux : sequence
            The flux in electrons (not ADU).
        channel : int
            The channel number.
        nChannel : int
            The total number of channels.
        log : logedit.Logedit
            The open log in which notes from this step can be added.
        unc : sequence
            The uncertainty on the flux
        parameters : eureka.lib.readEPF.Parameters
            The Parameters object containing the fitted parameters
            and their priors.
        freenames : list
            The specific names of all fitted parameters (e.g., including _ch#)
        time_units : str; optional
            The time units.
        name : str; optional
            A name for the object.
        share : bool; optional
            Whether the fit shares parameters between spectral channels.
        white : bool; optional
            Whether the current fit is for a white-light light curve
        multwhite : bool; optional
            Whether the current fit is for a multi white-light lightcurve fit.
        nints : bool; optional
            Number of exposures of each white lightcurve for splitting
            up time array.
        **kwargs : dict
            Parameters to set in the LightCurve object.
            Any parameter named log will not be loaded into the
            LightCurve object as Logedit objects cannot be pickled
            which is required for multiprocessing.
        """
        # Inherit from the Model-based LightCurve class
        lightcurve.LightCurve.__init__(
            self, time, flux, channel, nchannel, log, longparamlist,
            parameters, freenames, unc, time_units, name, share, white,
            multwhite, nints, **kwargs)

    @property
    def unc(self):
        """A getter for the unc array."""
        return self._unc

    @unc.setter
    def unc(self, unc_array):
        """A setter for the unc array.

        Parameters
        ----------
        unc_array : sequence
            The unc array.
        """
        # Check the type
        if not isinstance(unc_array, (np.ndarray, tuple, list, type(None))):
            raise TypeError("unc axis must be a tuple, list, or numpy array.")

        if isinstance(unc_array, np.ma.core.MaskedArray):
            # Convert to a numpy array with NaN masking
            unc_array = unc_array.filled(np.nan)

        # Set the array
        self._unc = unc_array

    @property
    def unc_fit(self):
        """A getter for the unc_fit array."""
        return self._unc_fit

    @unc_fit.setter
    def unc_fit(self, unc_fit_array):
        """A setter for the unc_fit array.

        Parameters
        ----------
        unc_fit_array : sequence
            The unc_fit array.
        """
        # Check the type
        if not isinstance(unc_fit_array, (np.ndarray, tuple, list,
                                          type(None))):
            raise TypeError("unc_fit axis must be a tuple, list, or "
                            "numpy array.")

        if isinstance(unc_fit_array, np.ma.core.MaskedArray):
            # Convert to a numpy array with NaN masking
            unc_fit_array = unc_fit_array.filled(np.nan)

        # Set the array
        self._unc_fit = unc_fit_array
