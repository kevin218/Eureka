
import george as gg
import numpy as np
import scipy.optimize as spo
import pickle

def gp_exp2(params, t, etc = []):
    """
    This function applies a squared exponential Gaussian Process.

    Parameters
    ----------
    a       : max amplitude
    b       : scale length
    t       : array of time/phase points

    Returns
    -------
    returns an array of flux values

    Revisions
    ---------
    Kevin Stevenson       Dec 2014
    """
    a,b         = np.exp(params[:2])    #Max amplitude, scale length
    nsamp       = params[2]
    flux        = etc[0]            #Flux
    ferr        = etc[1]            #Uncertainty in flux
    if len(etc) >= 3:
        isoptimize  = etc[2]        #Optimize hyperparameters
        savefile    = etc[3]
        loadfile    = etc[4]
    else:
        isoptimize  = False  
        savefile    = None
        loadfile    = None 
    
    if loadfile == None:
        # Set up the Gaussian process in log space.
        gp  = gg.GP(a * gg.kernels.ExpSquaredKernel(b), mean=np.mean(flux))
        # Compute the factorization of the matrix.
        gp.compute(t, ferr)
    else:
        #Load gp object
        handle  = open(loadfile, 'r')
        gp      = pickle.load(handle)
        handle.close()
    #print(gp.lnlikelihood(flux))
    if isoptimize == True:
        bestp, results = gp.optimize(t, flux, ferr, verbose=False)
        #results = spo.minimize(nll, gp.kernel.vector, args=(gp, flux), jac=grad_nll)
        #print(bestp, gp.kernel[:], results.x)
        gp.kernel[:] = bestp
        params[:2]   = bestp
        #print(gp.kernel.vector, np.log(gp.kernel.pars))
        #print(gp.lnlikelihood(flux))
        # Evaluate at times t
        #y   = gp.sample_conditional(flux, t)
        y   = np.mean(gp.sample_conditional(flux, t, 1000), axis=0)
        if savefile != None:
            #Save gp object
            handle    = open(savefile, 'w')
            pickle.dump(gp, handle)
            handle.close()
        return (y, params)
    else:
        gp.kernel[:] = params[:2]
        # Evaluate at times t
        if int(nsamp) == 1:
            y   = gp.sample_conditional(flux, t)
        else:
            y   = np.mean(gp.sample_conditional(flux, t, int(nsamp)), axis=0)
        return y
    

'''
# Define the objective function (negative log-likelihood in this case).
def nll(p, gp, flux): #, aplev):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = p
    ll = gp.lnlikelihood(flux, quiet=True)

    # The scipy optimizer doesn't play well with infinities.
    return -ll if np.isfinite(ll) else 1e25

# And the gradient of the objective function.
def grad_nll(p, gp, flux):    #, aplev):
    # Update the kernel parameters and compute the likelihood.
    gp.kernel[:] = p
    return -gp.grad_lnlikelihood(flux, quiet=True)
'''
