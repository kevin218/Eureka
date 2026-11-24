from copy import copy
from functools import partial

try:
    import catwoman
except ImportError:
    print("Could not import catwoman. Functionality may be limited.")

from .BatmanModels import BatmanTransitModel


class CatwomanTransitModel(BatmanTransitModel):
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
        # Inherit from BatmanTransitModel class
        super().__init__(**kwargs)
        self.name = 'catwoman transit'
        # Define transit model to be used
        self.transit_model = partial(self._make_transit_model,
                                     max_err=kwargs['max_err'],
                                     fac=kwargs['fac'])

        if ('rp2' not in self.longparamlist[0]
                and 'rprs2' not in self.longparamlist[0]):
            raise AssertionError('You must include an rp2 parameter in your '
                                 'EPF when using catwoman.')

    @staticmethod
    def _make_transit_model(params, time, **kwargs):
        """Initialize Catwoman with a separate parameter object."""
        # The constructor makes negative rp positive, but light_curve needs
        # the original sign to produce an inverse transit.
        return catwoman.TransitModel(copy(params), time, **kwargs)
