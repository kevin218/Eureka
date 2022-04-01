import numpy as np
import george
from george import kernels

from .Model import Model
from ...lib.readEPF import Parameters

###########for future tinygp version - tinygp needs python >= 3.7
###########!! careful when activating - did not test the tinygp code yet (didn't have python >= 3.7)
#import jax.numpy as jnp
#import tinygp
#from tinygp import kernels, GaussianProcess
#from jax.config import config
#config.update("jax_enable_x64", True)

class GPModel(Model):
    """Model for Gaussian Process (GP)"""
    def __init__(self, kernel_classes, kernel_inputs, lc, gp_code='george', **kwargs):
        """Initialize the GP model
        """
        # Inherit from Model class
        super().__init__(**kwargs)

        # Define model type (physical, systematic, other)
        self.modeltype = 'GP'

        # Check for Parameters instance
        self.parameters = kwargs.get('parameters')

        # Generate parameters from kwargs if necessary
        self.gp_code_name = gp_code
        self.kernel_types = kernel_classes
        self.kernel_input_names = kernel_inputs
        self.kernel_input_arrays = []
        self.nkernels = len(kernel_classes)
        self.lc = lc
        if self.parameters is None:
            coeff_dict = kwargs.get('coeff_dict')
            ## think about how to get coeff_dict ... 
            #params = {rN: coeff for rN, coeff in coeff_dict.items()
                      #if rN.startswith('r') and rN[1:].isdigit()}
            self.parameters = Parameters(**params)

        # Update coefficients
        self._parse_coeffs()

    def _parse_coeffs(self):
        """Convert dict of coefficients into a list
        of coefficients in increasing order

        Parameters
        ----------
        None

        Returns
        -------
        dictionary of coefficients for GP kernels
        """
        # Parse keyword arguments as coefficients
        self.coeffs = {}
        kernel_no = 0

        for k, v in self.parameters.dict.items():
            if k.startswith('A'):
                self.coeffs['A'] = v[0]
            if k.lower().startswith('m_') and k[2:].isdigit():
                self.coeffs[self.kernel_types[kernel_no]] = v[0]
                kernel_no += 1
            if k.startswith('WN'):
                self.coeffs['WN'] = v[0]
                if 'fixed' in v:
                    self.fit_white_noise = False
                else:
                    self.fit_white_noise = True
    
    def eval(self, fit, gp=None, **kwargs):
        """Compute GP with the given parameters
        Parameters 
        ----------
        fit - current model (i.e. transit model)

        Returns
        -------
        predicted systematics model 
        """
        # Get the time
        #if self.time is None:
            #self.time = kwargs.get('time')

        # Create the GP object with current parameters
        if gp==None:
            gp = self.setup_GP()       
        if self.nkernels > 1:
            if self.gp_code_name == 'george':
                gp.compute(self.kernel_input_arrays.T,self.lc.unc)
                mu,cov = gp.predict(self.lc.flux-fit,self.kernel_input_arrays.T)
            if self.gp_code_name == 'tinygp':
                cond_gp = gp.condition(self.kernel_input_arrays.T,self.lc.unc).gp
                mu, cov = cond_gp.loc, cond_gp.variance
        else:
            #print(y, y_err, fit)
            if self.gp_code_name == 'george':
                gp.compute(self.kernel_input_arrays[0],self.lc.unc)
                mu,cov = gp.predict(self.lc.flux-fit,self.kernel_input_arrays[0])
            if self.gp_code_name == 'tinygp':
                cond_gp = gp.condition(self.kernel_inputs[0],self.lc.unc).gp
                mu, cov = cond_gp.loc, cond_gp.variance
        #std = np.sqrt(np.diag(cov))

        #gp.evaluate() #tinygp           
        return mu#, std

    def set_inputs(self, normalise=False):
        """Setting up kernel inputs as array and standardizing them 
        see e.g. Evans et al. 2017
        Parameters
        ----------
        kernel input as strings e.g. 'time' 

        Returns
        ----------
        kernel inputs as a np.array
        """
        kernel_inputs = []
        for i in self.kernel_input_names:
            if i == 'time':
                x = self.lc.time
                #x = np.linspace(0.10, 0.22, num=1287)
                kernel_inputs.append(x)
                #kernel_inputs.append(self.time)
            ##add more input options here 

        if normalise: 
            #print(kernel_inputs)
            norm_kernel_inputs = [(i-i.mean())/i.std() for i in kernel_inputs]
            self.kernel_input_arrays =  np.array(norm_kernel_inputs)
            return
        else:
            self.kernel_input_arrays = np.array(kernel_inputs)
        return

    def setup_GP(self, **kwargs):
        """Set up GP kernels and GP object.
        
        Parameters
        ----------
        None

        Returns
        -------
        GP object

        """
        if len(self.kernel_input_arrays) == 0:
            self.set_inputs()

        for i in range(self.nkernels):
            if i == 0:
                kernel = self.get_kernel(self.kernel_types[i],i)
            else:
                kernel += self.get_kernel(self.kernel_types[i],i)

        kernel = kernels.ConstantKernel(self.coeffs['A'],ndim=self.nkernels,axes=np.arange(self.nkernels))*kernel

        if self.gp_code_name == 'george': 
            gp = george.GP(kernel, white_noise=self.coeffs['WN'],fit_white_noise=self.fit_white_noise, mean=0, fit_mean=False)#, solver=george.solvers.HODLRSolver)
        if self.gp_code_name == 'tinygp':
            gp = tinygp.GaussianProcess(kernel, diag=self.coeffs['WN']**2, mean=0)
        #tiny gp code
        #kernel = tinygp.kernels.ConstantKernel(self.coeffs['A'],ndim=self.nkernels,axes=np.arange(self.nkernels))*kernel
        return gp

    def loglikelihood(self, fit):
        """Compute log likelihood of GP
        Parameters 
        ----------
        None

        Returns
        -------
        log likelihood of the GP evaluated by george/tinygp
        """
        gp = self.setup_GP()
        #gp_evaluated = self.eval(lc, fit, gp=gp)
        # Create the GP object with current parameters
        if self.gp_code_name == 'george':
            if self.nkernels > 1:
                gp.compute(self.kernel_input_arrays.T,self.lc.unc)
            else:
                gp.compute(self.kernel_input_arrays[0],self.lc.unc)
            loglike = gp.lnlikelihood(self.lc.flux - fit,quiet=True)

        if self.gp_code_name == 'tinygp':
            if self.nkernels > 1:
                loglike = gp.condition(self.lc.flux - fit, X_test = self.kernel_input_arrays.T).log_probability
            else:
                loglike = gp.condition(self.lc.flux - fit, X_test = self.kernel_input_arrays[0]).log_probability  
        return loglike

    def get_kernel(self, kernel_name, i):
        """get individual kernels"""
        metric = ( 1./np.exp(self.coeffs[kernel_name]) )**2
        if self.gp_code_name == 'george':
            if kernel_name == 'Matern32':
                kernel = kernels.Matern32Kernel(metric,ndim=self.nkernels,axes=i)
            if kernel_name == 'ExpSquared':
                kernel = kernels.ExpSquaredKernel(metric,ndim=self.nkernels,axes=i)
            if kernel_name == 'RationalQuadratic':
                kernel = kernels.RationalQuadraticKernel(log_alpha=1,metric=metric,ndim=self.nkernels,axes=i)
            if kernel_name == 'Exp':
                kernel = kernels.ExpKernel(metric,ndim=self.nkernels,axes=i)   

        if self.gp_code_name == 'tinygp':
            if kernel_name == 'Matern32':
                kernel = tinygp.kernels.Matern32(metric,ndim=self.nkernels,axes=i)
            if kernel_name == 'ExpSquared':
                kernel = tinygp.kernels.ExpSquared(metric,ndim=self.nkernels,axes=i)
            if kernel_name == 'RationalQuadratic':
                kernel = tinygp.kernels.RationalQuadratic(alpha=1,scale=metric,ndim=self.nkernels,axes=i)
            if kernel_name == 'Exp':
                kernel = tinygp.kernels.Exp(metric,ndim=self.nkernels,axes=i) 

        return kernel

    def update(self, newparams, names, **kwargs):
        """Update parameter values"""
        for ii,arg in enumerate(names):
            if hasattr(self.parameters,arg):
                val = getattr(self.parameters,arg).values[1:]
                val[0] = newparams[ii]
                setattr(self.parameters, arg, val)
        self._parse_coeffs()
        return
