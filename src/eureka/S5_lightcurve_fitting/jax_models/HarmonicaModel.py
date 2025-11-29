from typing import Any, Dict, List, Optional

import inspect
import numpy as np
from jax.typing import ArrayLike

import harmonica.jax as harmonica
import jax.numpy as jnp

from . import JaxModel
from .AstroModel import compute_astroparams
from ..limb_darkening_fit import ld_profile
from ...lib.split_channels import split


def evaluate_harmonica_transit_model_jax(
    time_global: ArrayLike,
    nints: List[int],
    multwhite: bool,
    fitted_channels: List[int],
    num_planets: int,
    param_dict: Dict[str, Any],
    max_fourier_term: int,
    limb_dark_name: str,
) -> ArrayLike:
    """
    Evaluate the Harmonica transit model in a JAX-pure way.

    This mirrors :meth:`HarmonicaTransitModel.eval` but takes all required
    inputs explicitly, making it suitable for use inside a composite JAX
    model or NumPyro log-prob function.

    Parameters
    ----------
    time_global : array-like
        Full time array for the observation.
    nints : list[int]
        Number of integrations per channel.
    multwhite : bool
        Whether this is a multwhite fit.
    fitted_channels : list[int]
        Channel indices to evaluate.
    num_planets : int
        Number of planets in the system.
    param_dict : dict
        Flat parameter dictionary (decoded from the free parameter vector).
    max_fourier_term : int
        Number of Fourier coefficients to keep from the ``ab`` vector.
    limb_dark_name : str
        Name of the limb-darkening law ('uniform', 'linear', 'quadratic',
        'kipping2013', or 'nonlinear').

    Returns
    -------
    jnp.ndarray
        The model light curve concatenated across all selected channels.
    """
    if harmonica is None or jnp is None:
        raise ImportError(
            "harmonica.jax could not be imported, but "
            "HarmonicaTransitModel was requested. Please install "
            "the JAX-enabled version of harmonica."
        )

    time_global = jnp.array(time_global)
    light_curves: List[ArrayLike] = []

    for chan in fitted_channels:
        if multwhite:
            # Split the arrays that have lengths of the original time axis
            time_chan = split([time_global], nints, chan)[0]
        else:
            time_chan = time_global

        time_chan = jnp.array(time_chan)
        lc_chan = jnp.ones(time_chan.shape[0])

        for pid in range(num_planets):
            astro = compute_astroparams(param_dict, channel=chan, pid=pid)

            # `ab` is the Fourier coefficient vector; truncate to the
            # highest-order terms we actually want to use.
            ab = jnp.asarray(astro['ab'])[:max_fourier_term]

            if limb_dark_name in [
                'uniform',
                'linear',
                'quadratic',
                'kipping2013',
            ]:
                lc_planet = harmonica.harmonica_transit_quad_ld(
                    time_chan,
                    t0=astro['t0'],
                    period=astro['per'],
                    a=astro['a'],
                    inc=astro['inc_rad'],
                    ecc=astro['ecc'],
                    omega=astro['w_rad'],
                    u1=astro['u1'],
                    u2=astro['u2'],
                    r=ab,
                )
            elif limb_dark_name == 'nonlinear':
                lc_planet = harmonica.harmonica_transit_nonlinear_ld(
                    time_chan,
                    t0=astro['t0'],
                    period=astro['per'],
                    a=astro['a'],
                    inc=astro['inc_rad'],
                    ecc=astro['ecc'],
                    omega=astro['w_rad'],
                    u1=astro['u1'],
                    u2=astro['u2'],
                    u3=astro['u3'],
                    u4=astro['u4'],
                    r=ab,
                )
            else:
                raise NotImplementedError(
                    'The requested limb-darkening model '
                    f'"{limb_dark_name}" is not supported by '
                    'HarmonicaTransitModel.'
                )

            lc_chan = lc_chan * lc_planet

        light_curves.append(lc_chan)

    return jnp.concatenate(light_curves)


class HarmonicaTransitModel(JaxModel):
    """
    Transit model wrapper for the Harmonica JAX backend.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the transit model

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            ``eureka.S5_lightcurve_fitting.models.Model.__init__()``.
            Can pass in the ``parameters``, ``longparamlist``, ``nchan``,
            and ``paramtitles`` arguments here.
        """
        super().__init__(**kwargs)
        self.name = 'harmonica transit'
        self.modeltype = 'physical'

        log = kwargs.get('log')

        # Store the ld_profile metadata and ensure consistency
        self.ld_from_S4 = kwargs.get('ld_from_S4')
        ld_name = self.parameters.limb_dark.value
        ld_func = ld_profile(ld_name, use_gen_ld=self.ld_from_S4)
        len_params = len(inspect.signature(ld_func).parameters)
        self.coeffs = [f'u{n}' for n in range(1, len_params)]

        self.ld_from_file = kwargs.get('ld_from_file')

        # Replace u parameters with generated limb-darkening values
        if self.ld_from_S4 or self.ld_from_file:
            if log is not None:
                log.writelog('Using the following limb-darkening values:')
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
                        if log is not None:
                            log.writelog(f'{item}: {ld_val}')
                        # Use the file value as the starting guess
                        self.parameters.dict[item][0] = ld_val
                        # In a normal prior, center at the file value
                        if (
                            self.parameters.dict[item][-1] == 'N'
                            and self.recenter_ld_prior
                        ):
                            self.parameters.dict[item][-3] = ld_val
                        # Update the non-dictionary form as well
                        setattr(self.parameters, item,
                                self.parameters.dict[item])

        # Determine how many Fourier terms to keep (a1, b1, a2, b2, ...)
        # We'll store this as `self.max_fourier_term` to be used in eval
        self.max_fourier_term = 1
        for term in ['a3', 'b3', 'a2', 'b2', 'a1', 'b1']:
            if any(term in p for p in self.paramtitles):
                # e.g., "a3" → 2*3+1 = 7 (a0, a1, b1, a2, b2, a3, b3)
                self.max_fourier_term = 2 * int(term[1]) + 1
                break  # keep the highest used term only

        msg = (
            f'Using {self.max_fourier_term} Fourier terms in the '
            'Harmonica transit model.'
        )
        if log is not None:
            log.writelog(msg)

        self._limb_dark_name = ld_name

    def eval(
        self,
        param_dict: Optional[Dict[str, Any]] = None,
        channel: Optional[int] = None,
        **kwargs: Any,
    ) -> ArrayLike:
        """
        Evaluate the Harmonica transit model.

        Parameters
        ----------
        param_dict : dict; optional
            If None, uses values from ``self.parameters`` (i.e., fitted mode).
        channel : int; optional
            If provided, only evaluate for the specified channel.
        **kwargs : dict
            Reserved for future keyword arguments (unused).

        Returns
        -------
        jnp.ndarray
            The model light curve evaluated at observation times.
        """
        if harmonica is None or jnp is None:
            raise ImportError(
                "harmonica.jax could not be imported, but "
                "HarmonicaTransitModel was requested. Please install "
                "the JAX-enabled version of harmonica."
            )

        if param_dict is None:
            param_dict = self._get_param_dict()

        if channel is None:
            fitted_channels = list(self.fitted_channels)
        else:
            fitted_channels = [channel]

        return evaluate_harmonica_transit_model_jax(
            time_global=self.time,
            nints=self.nints,
            multwhite=self.multwhite,
            fitted_channels=fitted_channels,
            num_planets=self.num_planets,
            param_dict=param_dict,
            max_fourier_term=self.max_fourier_term,
            limb_dark_name=self._limb_dark_name,
        )
