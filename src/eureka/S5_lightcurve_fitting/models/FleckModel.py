import numpy as np
import astropy.units as unit

import fleck

from .BatmanModels import BatmanTransitModel


class FleckTransitModel(BatmanTransitModel):
    """Transit Model with Star Spots"""
    def __init__(self, **kwargs):
        """Initialize the fleck model

        Parameters
        ----------
        **kwargs : dict
            Additional parameters to pass to
            eureka.S5_lightcurve_fitting.models.Model.__init__().
            Can pass in the parameters, longparamlist, nchan, and
            paramtitles arguments here.
        """
        # Inherit from BatmanTransitModel class
        super().__init__(**kwargs)
        self.name = 'fleck transit'
        # Define transit model to be used
        self.transit_model = TransitModel

        self.spotcon_file = kwargs.get('spotcon_file')
        if self.spotcon_file:
            # Load spot contrast coefficients from a custom file
            try:
                spot_coeffs = np.genfromtxt(self.spotcon_file)
            except FileNotFoundError:
                raise Exception(f"The spot contrast file {self.spotcon_file}"
                                " could not be found.")

            # Load all spot contrasts into the parameters object
            log = kwargs.get('log')
            log.writelog("Using the following spot contrast values:")
            for c in range(self.nchannel_fitted):
                chan = self.fitted_channels[c]
                if c == 0 or self.nchannel_fitted == 1:
                    chankey = ''
                else:
                    chankey = f'_ch{chan}'
                item = f'spotcon{chankey}'
                if item in self.paramtitles:
                    contrast_val = spot_coeffs[chan]
                    log.writelog(f"{item}: {contrast_val}")
                    # Use the file value as the starting guess
                    self.parameters.dict[item][0] = contrast_val
                    # In a normal prior, center at the file value
                    if (self.parameters.dict[item][-1] == 'N' and
                            self.recenter_spotcon_prior):
                        self.parameters.dict[item][-3] = contrast_val
                    # Update the non-dictionary form as well
                    setattr(self.parameters, item,
                            self.parameters.dict[item])


class TransitModel():
    """
    Class for generating model transit light curves with fleck.
    """
    def __init__(self, pl_params, t, transittype="primary"):
        """
        Does some initial pre-checks and saves some parameters.

        Parameters
        ----------
        pl_params : object
            Contains the physical parameters for the transit model.
        t : array
            Array of times.
        transittype : str; optional
            Options are primary or secondary.  Default is primary.
        """
        if transittype != "primary":
            raise ValueError('The fleck transit model only allows transits and'
                             ' not eclipses.')

        if pl_params.limb_dark not in ['quadratic', 'kipping2013']:
            raise ValueError('limb_dark was set to "'
                             f'{self.parameters.limb_dark.value}" in '
                             'your EPF, while the fleck transit model only '
                             'allows "quadratic" or "kipping2013".')

        # store t for later use
        self.t = t

    def light_curve(self, pl_params):
        """
        Calculate a model light curve.

        Parameters
        ----------
        pl_params : object
            Contains the physical parameters for the transit model.

        Returns
        -------
        lc : ndarray
            Light curve.
        """
        # create arrays to hold values
        spotrad = np.zeros(0)
        spotlat = np.zeros(0)
        spotlon = np.zeros(0)

        for n in range(pl_params.nspots):
            # read radii, latitudes, longitudes, and contrasts
            if n > 0:
                spot_id = f'{n}'
            else:
                spot_id = ''
            spotrad = np.concatenate([
                spotrad, [getattr(pl_params, f'spotrad{spot_id}'),]])
            spotlat = np.concatenate([
                spotlat, [getattr(pl_params, f'spotlat{spot_id}'),]])
            spotlon = np.concatenate([
                spotlon, [getattr(pl_params, f'spotlon{spot_id}'),]])

        if np.any((np.abs(spotlat) > 90) | (np.abs(spotlon) > 180) |
                  (spotrad > 1)):
            # Returning nans or infs breaks the fits, so this was the
            # best I could think of
            return 1e6*np.ma.ones(self.t.shape)

        if pl_params.spotnpts is None:
            # Have a default spotnpts for fleck
            pl_params.spotnpts = 300

        inverse = False
        if pl_params.rp < 0:
            # The planet's radius is negative, so need to do some tricks to
            # avoid errors
            inverse = True
            pl_params.rp *= -1

        # Make the star object
        star = fleck.Star(spot_contrast=pl_params.spotcon,
                          u_ld=pl_params.u,
                          rotation_period=pl_params.spotrot)

        # Make the transit model
        fleck_times = np.linspace(self.t.data[0], self.t.data[-1],
                                  pl_params.spotnpts)
        fleck_transit = star.light_curve(
            spotlon[:, None]*unit.deg,
            spotlat[:, None]*unit.deg,
            spotrad[:, None],
            pl_params.spotstari*unit.deg,
            planet=pl_params, times=fleck_times,
            fast=pl_params.fleck_fast).flatten()
        lc = np.interp(self.t, fleck_times, fleck_transit)

        # Re-normalize to avoid degenaricies with c0
        lc /= lc[0]

        if inverse:
            # Invert the transit feature if rp<0
            lc = 2. - lc

        return lc
