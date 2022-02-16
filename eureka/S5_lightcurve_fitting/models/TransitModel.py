import numpy as np
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
        
        # Set parameters for multi-channel fits
        self.longparamlist = kwargs.get('longparamlist')
        self.nchan = kwargs.get('nchan')
        self.paramtitles = kwargs.get('paramtitles')

        # Store the ld_profile
        self.ld_func = ld_profile(self.parameters.limb_dark.value)
        len_params = len(inspect.signature(self.ld_func).parameters)
        self.coeffs = ['u{}'.format(n) for n in range(len_params)[1:]]

    def eval(self, **kwargs):
        """Evaluate the function with the given values"""
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')
        
        longparamlist=self.longparamlist
        nchan=self.nchan
        paramtitles=self.paramtitles

        #Initialize model
        bm_params = batman.TransitParams()
        
        # Set all parameters
        lcfinal=np.array([])
        for c in np.arange(nchan):
            # Set all parameters
            for index,item in enumerate(longparamlist[c]):
                setattr(bm_params,paramtitles[index],self.parameters.dict[item][0])
            
            #Set limb darkening parameters
            uarray=[]
            for u in self.coeffs:
                index=np.where(np.array(paramtitles)==u)[0]
                if len(index)!=0:
                    item=longparamlist[c][index[0]]
                    uarray.append(self.parameters.dict[item][0])
            bm_params.u = uarray

            # Use batman ld_profile name
            if self.parameters.limb_dark.value == '4-parameter':
                bm_params.limb_dark = 'nonlinear'

            # Make the eclipse
            tt = self.parameters.transittype.value
            m_eclipse = batman.TransitModel(bm_params, self.time, transittype=tt)

            lcfinal = np.append(lcfinal,m_eclipse.light_curve(bm_params))

        return lcfinal

    def update(self, newparams, names, **kwargs):
        """Update parameter values"""
        for ii,arg in enumerate(names):
            if hasattr(self.parameters,arg):
                val = getattr(self.parameters,arg).values[1:]
                val[0] = newparams[ii]
                setattr(self.parameters, arg, val)
        return
