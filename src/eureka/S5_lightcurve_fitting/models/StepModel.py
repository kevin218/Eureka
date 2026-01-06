import numpy as np

from .Model import Model
from ...lib.split_channels import split, get_trim


class StepModel(Model):
    """Model for step-functions in time"""
    def __init__(self, **kwargs):
        """Initialize the step-function model.

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
        """
        # Inherit from Model class
        super().__init__(**kwargs)
        self.name = 'step'

        # Define model type (physical, systematic, other)
        self.modeltype = 'systematic'

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
                # Split the arrays that have lengths
                # of the original time axis
                trim1, trim2 = get_trim(self.nints, chan)
                piece = self._time[trim1:trim2]
                self.time_local[trim1:trim2] = piece - piece.data[0]
        else:
            self.time_local = self._time - self._time.data[0]

    def _index_set_for_chan(self, chan):
        """Discover usable step indices for a given channel.

        Enumerate integer indices ``N`` present in any ``step{N}*`` or
        ``steptime{N}*`` key. For each candidate ``N``, use
        ``_get_param_value(..., default=None, chan=chan)`` to apply the
        standard precedence (``wl > ch > base``) and keep those where *both*
        a step and a steptime resolve to defined values.

        Parameters
        ----------
        chan : int
            Real channel id.

        Returns
        -------
        list of int
            Sorted indices where both step and steptime are defined after
            resolution for this (chan, wl).
        """
        if getattr(self, "parameters", None) is None:
            return []

        keys = list(self.parameters.dict.keys())

        # Collect every integer N appearing after 'step' or 'steptime'
        # and before any suffix (first non-digit).
        cand = set()
        for k in keys:
            if k.startswith('step'):
                rest = k[4:]
            elif k.startswith('steptime'):
                rest = k[8:]
            else:
                continue
            digits = []
            for ch_ in rest:
                if ch_.isdigit():
                    digits.append(ch_)
                else:
                    break
            if digits:
                cand.add(int(''.join(digits)))

        out = []
        for n in sorted(cand):
            has_step = self._get_param_value(f'step{n}', default=None,
                                             chan=chan) is not None
            has_time = self._get_param_value(f'steptime{n}', default=None,
                                             chan=chan) is not None
            if has_step and has_time:
                out.append(n)
        return out

    def _read_steps_for_chan(self, chan):
        """Read and sort step pairs for a given channel.

        For each index ``N`` discovered by ``_index_set_for_chan``,
        read values via ``_get_param_value`` using the same key rules
        as in ``_match_and_index``. Pairs with zero amplitude are
        skipped. The result is sorted by step time.

        Parameters
        ----------
        chan : int
            Real channel id.

        Returns
        -------
        list of tuple
            A list of ``(t_step, step)`` pairs sorted by ``t_step``.
        """
        idxs = self._index_set_for_chan(chan)
        pairs = []

        for n in idxs:
            # Resolve values for this channel (None if missing)
            step = self._get_param_value(f'step{n}', default=None,
                                         chan=chan)
            tstep = self._get_param_value(f'steptime{n}', default=None,
                                          chan=chan)
            if step is None or tstep is None or step == 0.0:
                continue
            pairs.append((tstep, step))

        # Ensure deterministic application order.
        pairs.sort(key=lambda x: x[0])
        return pairs

    def eval(self, channel=None, **kwargs):
        """Evaluate the step model.

        Parameters
        ----------
        channel : int; optional
            If not None, only consider one channel. Defaults to None.
        **kwargs : dict
            Must pass in the time array here if not already set.

        Returns
        -------
        lcfinal : np.ma.MaskedArray
            The model values at self.time.
        """
        nchan, channels = self._channels(channel)

        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        pieces = []
        for chan in channels:
            t = self.time_local
            if self.multwhite:
                # Split the arrays that have lengths of the original time axis
                t = split([t], self.nints, chan)[0]

            lcpiece = np.ma.ones(t.shape)
            for tstep, step in self._read_steps_for_chan(chan):
                mask = t >= tstep
                lcpiece[mask] += step

            lcpiece = np.ma.masked_where(np.ma.getmaskarray(t), lcpiece)
            pieces.append(lcpiece)

        if len(pieces) == 1:
            return pieces[0]
        else:
            return np.ma.concatenate(pieces)
