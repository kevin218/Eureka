import numpy as np

from .Model import Model
from ...lib.split_channels import split, get_trim
from ...lib import astropytable


class CommonModeModel(Model):
    """Common Mode Model"""
    def __init__(self, meta, log, **kwargs):
        """Initialize the common mode model.

        Parameters
        ----------
        meta : eureka.lib.readECF.MetaClass
            The current metadata object. Must have ``common_mode_file``
            and ``common_mode_name`` attributes.
        log : logedit.Logedit
            The current log in which to output messages from this function.
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
        super().__init__(**kwargs)
        self.name = 'common_mode'

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

        # Read common-mode systematics (typically from Stage 5 white LC)
        if (not hasattr(meta, 'common_mode_file') or
                not hasattr(meta, 'common_mode_name')):
            raise ValueError("meta must define 'common_mode_file' and "
                             "'common_mode_name'.")

        msg = (f"Reading {meta.common_mode_name} values from common-mode "
               f"systematics file: {meta.common_mode_file}.")
        log.writelog(msg, mute=(not meta.verbose))

        lc_table = astropytable.readtable(meta.common_mode_file)
        self.cm_flux = np.ma.masked_invalid(lc_table[meta.common_mode_name])
        self.cm_flux -= self.cm_flux.mean()

    @property
    def time(self):
        """A getter for the time."""
        return self._time

    @time.setter
    def time(self, time_array):
        """A setter for the time."""
        if time_array is None:
            self._time = None
            self.time_local = None
            return

        self._time = np.ma.masked_invalid(time_array)
        # Convert to local time
        if self.multwhite:
            self.time_local = np.ma.zeros(self._time.shape)
            for chan in self.fitted_channels:
                trim1, trim2 = get_trim(self.nints, chan)
                piece = self._time[trim1:trim2]
                self.time_local[trim1:trim2] = piece - piece.mean()
        else:
            self.time_local = self._time - self._time.mean()

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
        np.ma.MaskedArray
            The value of the model at self.time.
        """
        nchan, channels = self._channels(channel)

        if self.time is None:
            self.time = kwargs.get('time')

        pieces = []
        for chan in channels:
            time = self.time_local
            cm_flux = self.cm_flux
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                time, cm_flux = split([time, cm_flux], self.nints, chan)

            # Get the coefficients for this channel
            cm1 = self._get_param_value('cm1', 0.0, chan=chan)
            cm2 = self._get_param_value('cm2', 0.0, chan=chan)
            lcpiece = 1. + cm1*cm_flux + cm2*cm_flux**2

            # Respect any time mask
            lcpiece = np.ma.masked_where(np.ma.getmaskarray(time), lcpiece)
            pieces.append(lcpiece)

        if len(pieces) == 1:
            return pieces[0]
        else:
            return np.ma.concatenate(pieces)
