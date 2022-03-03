import numpy as np
import inspect
try:
    import batman
except ImportError:
    print("Could not import batman. Functionality may be limited.")

from ..limb_darkening_fit import ld_profile
from .Model import Model
from ..parameters import Parameters

class BatmanTransitModel(Model):
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
        
        #Initialize model
        bm_params = batman.TransitParams()
        
        # Set all parameters
        lcfinal=np.array([])
        for c in np.arange(self.nchan):
            # Set all parameters
            for index,item in enumerate(self.longparamlist[c]):
                setattr(bm_params, self.paramtitles[index], self.parameters.dict[item][0])
            
            #Set limb darkening parameters
            uarray=[]
            for u in self.coeffs:
                index=np.where(np.array(self.paramtitles)==u)[0]
                if len(index)!=0:
                    item=self.longparamlist[c][index[0]]
                    uarray.append(self.parameters.dict[item][0])
            bm_params.u = uarray

            # Use batman ld_profile name
            if self.parameters.limb_dark.value == '4-parameter':
                bm_params.limb_dark = 'nonlinear'
            elif self.parameters.limb_dark.value == 'kipping2013':
                bm_params.limb_dark = 'quadratic'
                u1  = 2*np.sqrt(bm_params.u[0])*bm_params.u[1]
                u2  = np.sqrt(bm_params.u[0])*(1-2*bm_params.u[1])
                bm_params.u = np.array([u1, u2])

            # Make the transit model
            m_transit = batman.TransitModel(bm_params, self.time, transittype='primary')

            lcfinal = np.append(lcfinal, m_transit.light_curve(bm_params))

        return lcfinal

class BatmanEclipseModel(Model):
    """Eclipse Model"""
    def __init__(self, **kwargs):
        """Initialize the eclipse model
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
        self.ld_func = ld_profile('uniform')

    def eval(self, **kwargs):
        """Evaluate the function with the given values"""
        # Get the time
        if self.time is None:
            self.time = kwargs.get('time')
        
       #Initialize model
        bm_params = batman.TransitParams()

        # Set all parameters
        lcfinal=np.array([])
        for c in np.arange(self.nchan):
            # Set all parameters
            for index,item in enumerate(self.longparamlist[c]):
                setattr(bm_params, self.paramtitles[index], self.parameters.dict[item][0])

            bm_params.limb_dark = 'uniform'
            bm_params.u = []

            if not np.any(['t_secondary' in key for key in self.longparamlist[c]]):
                # If not explicitly fitting for the time of eclipse, get the time of eclipse from the time of transit, period, eccentricity, and argument of periastron
                m_transit = batman.TransitModel(bm_params, self.time, transittype='primary')
                bm_params.t_secondary = m_transit.get_t_secondary(bm_params)
            
            # Make the eclipse model
            m_eclipse = batman.TransitModel(bm_params, self.time, transittype='secondary')

            lcfinal = np.append(lcfinal, m_eclipse.light_curve(bm_params))

        return lcfinal
