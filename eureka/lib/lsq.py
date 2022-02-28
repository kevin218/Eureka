
import numpy as np
import scipy.optimize as op
from copy import deepcopy

def modelfunc(freepars, lc, model, pmin, pmax, freenames, indep_vars):
    ilow            = np.where(freepars < pmin)
    ihi             = np.where(freepars > pmax)
    freepars[ilow]    = pmin[ilow]
    freepars[ihi]     = pmax[ihi]
    model.update(freepars, freenames)
    model_lc        = model.eval()
    if "scatter_ppm" in freenames:
        # ind = np.where(freenames=="scatter_ppm")
        ind = [i for i in np.arange(len(freenames)) if freenames[i][0:11] == "scatter_ppm"]
        lc.unc_fit = np.ones_like(lc.flux) * freepars[ind[0]] * 1e-6
        
        if len(ind)>1:
            for chan in np.arange(lc.flux.size//lc.time.size):
                lc.unc_fit[chan*lc.time.size:(chan+1)*lc.time.size] = freepars[ind[chan]] * 1e-6

    else:
        lc.unc_fit = deepcopy(lc.unc)
    residuals       = (lc.flux - model_lc)/lc.unc_fit
    '''
    #Apply priors, if they exist
    if len(fit[j].ipriors) > 0:
        pbar   = fit[j].priorvals[:,0]  #prior mean
        psigma = np.zeros(len(pbar))    #prior standard deviation
        # Determine psigma based on which side of asymmetric Gaussian nextp is on
        for i in range(len(fit[j].ipriors)):
            if params[fit[j].ipriors[i]] < pbar[i]:
                psigma[i] = fit[j].priorvals[i,1]
            else:
                psigma[i] = fit[j].priorvals[i,2]
            #print(params[fit[j].ipriors[i]],pbar[i],psigma[i])
            #print(sum(residuals**2),(np.sqrt(fit[j].nobj)*(params[fit[j].ipriors[i]] - pbar[i])/psigma[i])**2)
            #plt.pause(1)
            residuals = np.concatenate((residuals,[np.sqrt(fit[j].nobj)*(params[fit[j].ipriors[i]] - pbar[i])/psigma[i]]),axis=0)
    '''
    return residuals

def minimize(lc, model, freepars, pmin, pmax, freenames, indep_vars):
    return op.least_squares(modelfunc, freepars,
                            args=(lc, model, pmin, pmax, freenames, indep_vars),
                            ftol=1e-15, xtol=1e-15, gtol=1e-15)
    # return op.leastsq(modelfunc, freepars,
    #                   args=(lc, model, pmin, pmax, freenames, indep_vars),
    #                   ftol=1e-16, xtol=1e-15, gtol=1e-16, full_output=True,
    #                   factor=100, diag=1000./(freepars), maxfev=1000)
                      #diag=100./(pmax-pmin))
