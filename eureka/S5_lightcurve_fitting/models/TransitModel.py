import inspect
try:
    import batman
except ImportError:
    print("Could not import batman. Functionality may be limited.")

from ..limb_darkening_fit import ld_profile
from .Model import Model
from ..parameters import Parameters

class TransitModel(Model):
    """Transit Model"""
    def __init__(self, **kwargs):
        """Initialize the transit model
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

        # Store the ld_profile
        self.ld_func = ld_profile(self.parameters.limb_dark.value)
        len_params = len(inspect.signature(self.ld_func).parameters)
        self.coeffs = ['u{}'.format(n) for n in range(len_params)[1:]]

    def eval(self, **kwargs):
        """Evaluate the function with the given values"""
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')

        # Generate with batman
        bm_params = batman.TransitParams()

        # Set all parameters
        for arg, val in self.parameters.dict.items():
            setattr(bm_params, arg, val[0])
        #for p in self.parameters.list:
        #    setattr(bm_params, p[0], p[1])

        # Combine limb darkening coeffs
        bm_params.u = [getattr(self.parameters, u).value for u in self.coeffs]

        # Use batman ld_profile name
        if self.parameters.limb_dark.value == '4-parameter':
            bm_params.limb_dark = 'nonlinear'

        # Make the eclipse
        tt = self.parameters.transittype.value
        m_eclipse = batman.TransitModel(bm_params, self.time, transittype=tt)

        # Evaluate the light curve
        return m_eclipse.light_curve(bm_params)

    def update(self, newparams, names, **kwargs):
        """Update parameter values"""
        for ii,arg in enumerate(names):
            if hasattr(self.parameters,arg):
                val = getattr(self.parameters,arg).values[1:]
                val[0] = newparams[ii]
                setattr(self.parameters, arg, val)
        # ii = 0
        # for arg, val in self.parameters.dict.items():
        #     val[0] = newparams[ii]
        #     setattr(self.parameters, arg, val)
        #     ii += 1
        return