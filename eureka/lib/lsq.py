
import numpy as np
import scipy.optimize as op

def modelfunc(freepars, lc, model, pmin, pmax, freenames, indep_vars):
    #params[ifreepars] = freepars
    ilow            = np.where(freepars < pmin)
    ihi             = np.where(freepars > pmax)
    freepars[ilow]    = pmin[ilow]
    freepars[ihi]     = pmax[ihi]
    model.update(freepars, freenames)
    #model.time = time
    #model.components[0].time = time
    model_lc        = model.eval()
    residuals       = (lc.flux - model_lc)/lc.unc
    '''
    residuals       = []
    for j in range(numevents):
        fit0 = np.ones(fit[j].nobj)
        k    = 0
        for i in range(cummodels[j],cummodels[j+1]):
            if functype[i] == 'ipmap':
                fit[j].etc[k] = fit0
            fit0 *= myfuncs[i](params[iparams[i]], funcx[i], fit[j].etc[k])
            k    += 1
        residuals = np.concatenate((residuals,(fit0 - flux[j])/uncertainty[j]),axis=0)
    '''
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
    #modelfunc = lambda freepars, params, uncertainty: callmodelfunc(freepars, params, uncertainty)
    return op.leastsq(modelfunc, freepars,
                      args=(lc, model, pmin, pmax, freenames, indep_vars),
                      factor=100, ftol=1e-16, xtol=1e-16, gtol=1e-16)#, diag=1./stepsize[ifreepars])

#params, pmin, pmax, uncertainty, ifreepars, numevents, fit[j].nobj, cummodels, functype, fit[j].etc, myfuncs, iparams, funcx, flux
