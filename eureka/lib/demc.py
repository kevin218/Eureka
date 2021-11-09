
import numpy as np
import numpy.random as npr
import time#, timer
from . import gelmanrubin as gr
#reload(gr)
#import python_models as mc
#import models_c as mc
import multiprocessing as mp


def calcModel(nchains, functype, myfuncs, pedit, nextp, iortholist, funcx, cummodels, numparams, j, iblock=None, chains=None):
    '''
    Compute model light curve by combining model components.  Also returns correlated noise parameters.
    '''
    #Build final model from model components
    ymodels     = np.ones((nchains, fit[j].nobj))
    noisepars   = [[] for i in range(nchains)]
    k           = 0
    if chains == None:
        chains = range(nchains)
    if iblock == None:
        iblock = range(cummodels[j],cummodels[j+1])
    for i in range(cummodels[j],cummodels[j+1]):
        if iblock.__contains__(i):
            for n in chains:
                if   functype[i] == 'ortho':
                    #MODIFY COPY OF nextp ONLY
                    pedit[n,iortholist] = myfuncs[i](pedit[n,iortholist], funcx[i], fit[j].etc[k])
                elif (functype[i] == 'ipmap') or (functype[i] == 'spline'):
                    ymodels[n] *= myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], ymodels[n])
                elif functype[i] == 'posoffset':
                    # Record change in Position 0 => cannot orthogonalize position parameters
                    ymodels[n] *= myfuncs[i](nextp[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
                elif hasattr(fit[j], 'timebins') and (functype[i] == 'ecl/tr'
                                                  or  functype[i] == 'ramp'
                                                  or  functype[i] == 'sinusoidal'):
                    # Average over high-resolution model
                    hiresmodel = myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
                    if len(fit[j].timebins) == fit[j].nobj:
                        for tb in range(len(fit[j].timebins)):
                            ymodels[n,tb] *= np.mean(hiresmodel[fit[j].timebins[tb]])
                    else:
                        for tb in range(len(fit[j].timebinsuc)):
                            ymodels[n,tb] *= np.mean(hiresmodel[fit[j].timebinsuc[tb]])
                elif functype[i] == 'noise':
                    noisepars[n]  = pedit[n,numparams[i]:numparams[i+1]]
                else:
                    ymodels[n] *= myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
        k += 1
    return ymodels, noisepars

# Calculate chi^2
def calcChisq(y, sigma, ymodels, nchains, nextp, j, noisepars, isrednoise, wavelet, noisefunc, chains=None):
    '''
    Compute chi-squared with priors.
    '''
    if chains == None:
        chains = range(nchains)
    chi2 = np.zeros(nchains)
    for n in chains:
        if isrednoise == False:
            #chi2[n] = mc.chisq(ymodels[n], y, sigma)
            chi2[n]  += np.sum((ymodels[n] - y)**2 / sigma**2)
        else:
            chi2[n] = noisefunc(noisepars[n], ymodels[n]-y, wavelet)
        # Apply prior, if one exists
        if len(fit[j].ipriors) > 0:
            pbar   = fit[j].priorvals[:,0]  #prior mean
            psigma = np.zeros(len(pbar))    #prior standard deviation
            # Determine psigma based on which side of asymmetric Gaussian nextp is on
            for i in range(len(fit[j].ipriors)):
                if nextp[n,fit[j].ipriors[i]] < pbar[i]:
                    psigma[i] = fit[j].priorvals[i,1]
                else:
                    psigma[i] = fit[j].priorvals[i,2]
                #chi2[n] += fit[j].nobj*((nextp[n,fit[j].ipriors[i]] - pbar[i])/psigma[i])**2
                chi2[n] += ((nextp[n,fit[j].ipriors[i]] - pbar[i])/psigma[i])**2

    return chi2

def demc_block(y, pars, pmin, pmax, stepsize, numit, sigma, numparams, cummodels, functype, myfuncs, funcx, iortholist, fits, gamma=None, isGR=True, ncpu=1):
    """
    This function uses a differential evolution Markov chain with block updating to assess uncertainties.

    PARAMETERS
    ----------
    y:         Array containing dependent data
    Params:    Array of initial guess for parameters
    #Pmin:      Array of parameter minimum values
    #Pmax:      Array of parameter maximum values
    stepsize:  Array of 1-sigma change in parameter per iteration
    Numit:	   Number of iterations to perform
    Sigma:	   Standard deviation of data noise in y
    Numparams: Number of parameters for each model
    Cummodels: Cumulative number of models used
    Functype:  Define function type (eclipse, ramp, ip, etc), see models.py
    Myfuncs:   Pointers to model functions
    Funcx:	   Array of x-axis values for myfuncs
    fit:       List of fit objects
    gamma:     Multiplcation factor in parameter differential, establishes acceptance rate

    OUTPUTS
    -------
    This function returns an array of the best fitting parameters,
    an array of all parameters over all iterations, and numaccept.

    REFERENCES
    ----------
    Cajo J. F. Ter Braak, "Genetic algorithms and Markov Chain Monte Carlo: Differential Evolution Markov Chain makes Bayesian computing easy," Biometrics, 2006.

    HISTORY
    -------
    Adapted from mcmc.py
        Kevin Stevenson, UChicago   August 2012
    """
    global fit
    fit   = fits

    params          = np.copy(pars)
    nchains, nump   = params.shape
    nextp           = np.copy(params)       #Proposed parameters
    bestp           = np.copy(params[0])    #Best-fit parameters
    pedit           = np.copy(params)       #Editable parameters

    numaccept       = 0
    allparams       = np.zeros((nump, nchains, numit))
    inotfixed       = np.where(stepsize != 0)[0]
    ishare          = np.where(stepsize < 0)[0]
    #ifree           = np.where(stepsize > 0)[0]
    outside         = np.zeros((nchains, nump))
    numevents       = len(fit)
    intsteps        = np.min((numit/5,1e5))
    isrednoise      = False
    wavelet         = None
    noisefunc       = None

    #UPDATE PARAMTER(S) EQUAL TO OTHER PARAMETER(S)
    if (ishare.size > 0):
        for s in range(ishare.size):
            params[:,ishare[s]] = params[:,int(abs(stepsize[ishare[s]])-1)]

    #Define blocks
    blocks = []
    for j in range(numevents):
        #Build list of blocks
        blocks = np.concatenate((blocks, fit[j].blocks))
        for i in range(cummodels[j],cummodels[j+1]):
            if functype[i] == 'noise':
                # Set up for modified chi-squared calculation using correlated noise
                isrednoise   = True
                wavelet      = fit[j].etc[k]
                noisefunc    = myfuncs[i]

    blocks    = blocks.astype(int)
    iblocks   = []
    eps       = []
    numblocks = blocks.max() + 1
    numbp     = np.zeros(numblocks)
    ifree     = [[] for i in range(numblocks)]
    for b in range(numblocks):
        #Map block indices
        whereb = np.where(blocks == b)[0]
        iblocks.append(whereb)
        #Locate indices of free parameters in each block
        for w in whereb:
            ifree[b] = np.concatenate((ifree[b],numparams[w]+np.where(stepsize[numparams[w]:numparams[w+1]] > 0)[0])).astype(int)
        #Calculate number of free parameters per block
        numbp[b] += len(ifree[b])
        eps.append(npr.normal(0, stepsize[ifree[b]]/100., [numit,numbp[b]]))

    print("Number of free parameters per block:")
    print(numbp)
    numa        = np.zeros(numblocks)
    if gamma == None:
        gamma   = 2.38/np.sqrt(2.*numbp)
    print("gamma:")
    print(gamma)

    #Calc chi-squared for model type using current params
    currchisq   = np.zeros(nchains)
    currmodel   = [[] for i in range(numevents)]
    for j in range(numevents):
        currmodel[j], noisepars = calcModel(nchains, functype, myfuncs, pedit, params, iortholist[j],
                                            funcx, cummodels, numparams, j)
        currchisq += calcChisq(y[j], sigma[j], currmodel[j], nchains, params, j, noisepars, isrednoise, wavelet, noisefunc)

    bestchisq = currchisq[0]

    #GENERATE RANDOM NUMBERS FOR MCMC
    numnotfixed = len(inotfixed)
    unif        = npr.rand(numit,nchains)
    randchains  = npr.randint(0,nchains,[numit,nchains,2])

    #START TIMER
    clock = timer.Timer(numit,progress = np.arange(0.05,1.01,0.05))

    #Run Differential Evolution Monte Carlo algorithm 'numit' times
    for m in range(numit):
        #Select next event (block) to update
        b           = m % numblocks
        #Remove model component(s) that are taking a step
        pedit       = np.copy(params)
        nextmodel   = currmodel[:]
        for j in range(numevents):
            ymodels, noisepars = calcModel(nchains, functype, myfuncs, pedit, params, iortholist[j],
                                           funcx, cummodels, numparams, j, iblocks[b])
            nextmodel[j] = np.divide(currmodel[j],ymodels)
        #Generate next step using differential evolution
        for n in range(nchains):
            rand1, rand2 = randchains[m,n]
            while rand1 == n or rand2 == n or rand1 == rand2:
                rand1, rand2 = npr.randint(0,nchains,2)
            nextp[n,ifree[b]] = params[n,ifree[b]] + gamma[b]*(params[rand1,ifree[b]]-params[rand2,ifree[b]]) + eps[b][m]
            #CHECK FOR NEW STEPS OUTSIDE BOUNDARIES
            ioutside     = np.where(np.bitwise_or(nextp[n] < pmin, nextp[n] > pmax))[0]
            if (len(ioutside) > 0):
                nextp[n,ioutside]    = np.copy(params[n,ioutside])
                outside[n,ioutside] += 1
        #UPDATE PARAMTER(S) EQUAL TO OTHER PARAMETER(S)
        if (ishare.size > 0):
            for s in range(ishare.size):
                nextp[:,ishare[s]] = nextp[:,int(abs(stepsize[ishare[s]])-1)]
        #COMPUTE NEXT CHI SQUARED AND ACCEPTANCE VALUES
        pedit       = np.copy(nextp)
        nextchisq   = np.zeros(nchains)
        for j in range(numevents):
            ymodels, noisepars = calcModel(nchains, functype, myfuncs, pedit, params, iortholist[j], funcx, cummodels, numparams, j, iblocks[b])
            nextmodel[j] = np.multiply(nextmodel[j],ymodels)
            nextchisq   += calcChisq(y[j], sigma[j], nextmodel[j], nchains, params, j, noisepars, isrednoise, wavelet, noisefunc)
        #CALCULATE ACCEPTANCE PROBABILITY
        accept = np.exp(0.5 * (currchisq - nextchisq))
        #print(b,currchisq[0], nextchisq[0], accept[0])
        for n in range(nchains):
            if accept[n] >= 1:
                #ACCEPT BETTER STEP
                numaccept    += 1
                numa[b]      += 1
                params[n]     = np.copy(nextp[n])
                currchisq[n]  = np.copy(nextchisq[n])
                if (currchisq[n] < bestchisq):
                    bestp     = np.copy(params[n])
                    bestchisq = np.copy(currchisq[n])
            elif unif[m,n] <= accept[n]:
                #ACCEPT WORSE STEP
                numaccept    += 1
                numa[b]      += 1
                params[n]     = np.copy(nextp[n])
                currchisq[n]  = np.copy(nextchisq[n])

        allparams[:,:,m] = params.T
        #PRINT INTERMEDIATE INFO
        if ((m+1) % intsteps == 0) and (m > 0):
            print("\n" + time.ctime())
            #print("Number of times parameter tries to step outside its prior:")
            #print(outside)
            print("Current Best Parameters: ")
            print(bestp)

            #Apply Gelman-Rubin statistic
            if isGR:
                #Check for no accepted steps in each chain
                #stdev   = np.std(allparams[inotfixed],axis=1)
                #ichain  = np.where(stdev > 0.)[0]
                #Call test
                #psrf, meanpsrf = gr.convergetest(allparams[inotfixed,ichain,:m+1], len(ichain))
                psrf, meanpsrf = gr.convergetest(allparams[inotfixed,:,:m+1], nchains)
                numconv = np.sum(np.bitwise_and(psrf < 1.01, psrf >= 1.00))
                print("Gelman-Rubin statistic for free parameters:")
                print(psrf)
                if numconv == numnotfixed: #and m >= 1e4:
                    print("All parameters have converged to within 1% of unity. Halting MCMC.")
                    allparams = allparams[:,:,:m+1]
                    break
        clock.check(m+1)

    print("Acceptance rate per block (%):")
    print(100.*numa*numblocks/numit/nchains)
    allparams = np.reshape(allparams,(nump, (m+1)*nchains))
    return allparams, bestp, numaccept, (m+1)*nchains

#****************************************************************

def calcChi2(nchains, functype, myfuncs, pedit, nextp, iortholist, funcx, cummodels, numparams, j, isrednoise, wavelet, noisefunc, systematics, chains=None):
    '''
    Compute model light curve by combining model components.
    '''
    #Build final model from model components
    ymodels     = np.ones((nchains, fit[j].nobj))
    noisepars   = [[] for i in range(nchains)]
    k           = 0
    if chains == None:
        chains = range(nchains)
    for i in range(cummodels[j],cummodels[j+1]):
        for n in chains:
            if   functype[i] == 'ortho':
                #MODIFY COPY OF nextp ONLY
                pedit[n,iortholist] = myfuncs[i](pedit[n,iortholist], funcx[i], fit[j].etc[k])
            elif (functype[i] == 'ipmap') or (functype[i] == 'spline'):
                ymodels[n] *= myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], ymodels[n])
            elif functype[i] == 'posoffset':
                # Record change in Position 0 => cannot orthogonalize position parameters
                ymodels[n] *= myfuncs[i](nextp[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
            elif hasattr(fit[j], 'timebins') and (functype[i] == 'ecl/tr'
                                              or  functype[i] == 'ramp'
                                              or  functype[i] == 'sinusoidal'):
                # Average over high-resolution model
                hiresmodel = myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
                if len(fit[j].timebins) == fit[j].nobj:
                    for tb in range(len(fit[j].timebins)):
                        ymodels[n,tb] *= np.mean(hiresmodel[fit[j].timebins[tb]])
                else:
                    for tb in range(len(fit[j].timebinsuc)):
                        ymodels[n,tb] *= np.mean(hiresmodel[fit[j].timebinsuc[tb]])
            elif functype[i] == 'noise':
                noisepars[n]  = pedit[n,numparams[i]:numparams[i+1]]
            else:
                ymodels[n] *= myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
        k += 1

    # Calculate chi^2
    chi2 = np.zeros(nchains)
    for n in chains:
        if isrednoise == False:
            #chi2[n] = mc.chisq(ymodels[n]*systematics[n][j], data[j], unc[j])
            chi2[n] = np.sum((ymodels[n]*systematics[n][j] - data[j])**2 / unc[j]**2)
        else:
            chi2[n] = noisefunc(noisepars[n], ymodels[n]*systematics[n][j]-data[j], wavelet)
        # Apply prior, if one exists
        if len(fit[j].ipriors) > 0:
            pbar   = fit[j].priorvals[:,0]  #prior mean
            psigma = np.zeros(len(pbar))    #prior standard deviation
            # Determine psigma based on which side of asymmetric Gaussian nextp is on
            for i in range(len(fit[j].ipriors)):
                if nextp[n,fit[j].ipriors[i]] < pbar[i]:
                    psigma[i] = fit[j].priorvals[i,1]
                else:
                    psigma[i] = fit[j].priorvals[i,2]
                #chi2[n] += fit[j].nobj*((nextp[n,fit[j].ipriors[i]] - pbar[i])/psigma[i])**2
                chi2[n] += ((nextp[n,fit[j].ipriors[i]] - pbar[i])/psigma[i])**2

    return chi2

#
def writeChi2(chi2):
    '''
    Write models after multiprocessing.
    '''
    global nextchisq
    nextchisq += chi2
    return

def demc(y, pars, pmin, pmax, stepsize, numit, sigma, numparams, cummodels, functype, myfuncs, funcx, iortholist, nights, fits, gamma=None, isGR=True, ncpu=1):
    """
    This function uses a differential evolution Markov chain to assess uncertainties.

    PARAMETERS
    ----------
    y:         Array containing dependent data
    Params:    Array of initial guess for parameters
    #Pmin:      Array of parameter minimum values
    #Pmax:      Array of parameter maximum values
    stepsize:  Array of 1-sigma change in parameter per iteration
    Numit:	   Number of iterations to perform
    Sigma:	   Standard deviation of data noise in y
    Numparams: Number of parameters for each model
    Cummodels: Cumulative number of models used
    Functype:  Define function type (eclipse, ramp, ip, etc), see models.py
    Myfuncs:   Pointers to model functions
    Funcx:	   Array of x-axis values for myfuncs
    fit:       List of fit objects
    gamma:     Multiplcation factor in parameter differential, establishes acceptance rate

    OUTPUTS
    -------
    This function returns an array of the best fitting parameters,
    an array of all parameters over all iterations, and numaccept.

    REFERENCES
    ----------
    Cajo J. F. Ter Braak, "Genetic algorithms and Markov Chain Monte Carlo: Differential Evolution Markov Chain makes Bayesian computing easy," Biometrics, 2006.

    HISTORY
    -------
    Adapted from mcmc.py
        Kevin Stevenson, UChicago   August 2012
    Multiplied prior by number of points in fit
                                    January 2014
    """
    global nextchisq, fit, data, unc
    fit   = fits
    data  = y
    unc   = sigma

    params          = np.copy(pars)
    nchains, nump   = params.shape
    nextp           = np.copy(params)       #Proposed parameters
    bestp           = np.copy(params[0])    #Best-fit parameters
    pedit           = np.copy(params)       #Editable parameters

    numaccept       = 0
    #allparams must be 64-bit!
    allparams       = np.zeros((nump, nchains, numit))
    inotfixed       = np.where(stepsize != 0)[0]
    ishare          = np.where(stepsize < 0)[0]
    ifree           = np.where(stepsize > 0)[0]
    outside         = np.zeros((nchains, nump))
    numevents       = len(fit)
    intsteps        = np.min((numit/5,1e5))
    isrednoise      = False
    wavelet         = None
    noisefunc       = None
    numfree         = len(ifree)
    print("Number of free parameters:")
    print(len(ifree))
    if gamma == None:
        gamma     = 2.38/np.sqrt(2*numfree)
    print('Gamma = ' + str(gamma))

    #UPDATE PARAMTER(S) EQUAL TO OTHER PARAMETER(S)
    if (ishare.size > 0):
        for s in range(ishare.size):
            params[:,ishare[s]] = params[:,int(abs(stepsize[ishare[s]])-1)]

    # Construct non-analytic systematic model
    for nn in np.unique(nights):
        tonight   = np.where(nights == nn)[0]
        if hasattr(fit[tonight[0]], 'whiteparams') and fit[tonight[0]].whiteparams != None:
            if type(fit[tonight[0]].whiteparams) == type(np.array([])):
                #Only 1 model in model for white LC, grandfathered code
                #print("WARNING: You are using grandfathered code.  Update whiteparams to handle multiple models.")
                #whitemodel  = np.zeros((nchains,len(fit[tonight[0]].good)))
                i = int(fit[tonight[0]].whiteparams[0])
                whitemodel = myfuncs[i](fit[tonight[0]].whiteparams[1:], fit[tonight[0]].tuall, None)
            else:
                #Any number of models can be used to build white LC
                whitemodel  = np.ones((nchains,len(fit[tonight[0]].good)))
                for k in range(len(fit[tonight[0]].whiteparams)):
                    i = int(fit[tonight[0]].whiteparams[k][0])
                    whitemodel *= myfuncs[i](fit[tonight[0]].whiteparams[k][1:], fit[tonight[0]].tuall, None)
                    #whitemodel *= myfuncs[i](fit[tonight[0]].whiteparams[k][1:], funcxuc[i], None)
            for j in tonight:
                fit[j].whitemodel = whitemodel
        elif hasattr(fit[tonight[0]], 'iswhitelc') and fit[tonight[0]].iswhitelc != False:
            whitemodel = np.zeros((nchains,len(fit[tonight[0]].good)))
            weight     = np.zeros((nchains,len(fit[tonight[0]].good)))
            for n in range(nchains):
                for j in tonight:
                    k = 0
                    for i in range(cummodels[j],cummodels[j+1]):
                        if functype[i] == 'ecl/tr':
                            specmodel = myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
                            specmodeluc = np.zeros(len(fit[j].clipmask))
                            specmodeluc[fit[j].isclipmask] = specmodel
                            whitemodel[n,fit[j].isgood] += specmodeluc
                            weight    [n,fit[j].isgood] += specmodel[0]
                        k    += 1
                whitemodel[n] /= weight[n]
                #FINDME: Need to determine exact anchor point
                #slope      = fit[0].iswhitelc / (1-whitemodel[n].min())
                #offset     = 1 - slope
                #whitemodel[n] = slope*whitemodel[n] + offset
            for j in tonight:
                fit[j].whitemodel = whitemodel
        else:
            for j in tonight:
                fit[j].whitemodel = np.ones((nchains,len(fit[j].good)))

    #Calc chi-squared for model type using current params
    currchisq   = np.zeros(nchains)
    noisepars   = [[] for i in range(nchains)]
    for j in range(numevents):
        #Build final model from model components
        ymodels     = np.ones((nchains, fit[j].nobj))
        k           = 0
        for i in range(cummodels[j],cummodels[j+1]):
            for n in range(nchains):
                if   functype[i] == 'ortho':
                    #MODIFY COPY OF nextp ONLY
                    pedit[n,iortholist[j]] = myfuncs[i](pedit[n,iortholist[j]], funcx[i], fit[j].etc[k])
                elif (functype[i] == 'ipmap') or (functype[i] == 'spline'):
                    ymodels[n] *= myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], ymodels[n])
                elif functype[i] == 'posoffset':
                    # Record change in Position 0 => cannot orthogonalize position parameters
                    ymodels[n] *= myfuncs[i](nextp[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
                elif hasattr(fit[j], 'timebins') and (functype[i] == 'ecl/tr'
                                                  or  functype[i] == 'ramp'
                                                  or  functype[i] == 'sinusoidal'):
                    # Average over high-resolution model
                    hiresmodel = myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
                    if len(fit[j].timebins) == fit[j].nobj:
                        for tb in range(len(fit[j].timebins)):
                            ymodels[n,tb] *= np.mean(hiresmodel[fit[j].timebins[tb]])
                    else:
                        for tb in range(len(fit[j].timebinsuc)):
                            ymodels[n,tb] *= np.mean(hiresmodel[fit[j].timebinsuc[tb]])
                elif functype[i] == 'noise':
                    # Set up for modified chi-squared calculation using correlated noise
                    isrednoise   = True
                    wavelet      = fit[j].etc[k]
                    noisefunc    = myfuncs[i]
                    noisepars[n] = pedit[n,numparams[i]:numparams[i+1]]
                else:
                    ymodels[n] *= myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
            k += 1
            #Multiply analytic model by non-analytic systematics model
            systematics = fit[j].whitelc*fit[j].refspeclc/(fit[j].whitemodel[n][fit[j].isgood].flatten())[fit[j].isclipmask]
            ymodels[n] *= systematics
        # Calculate chi^2
        for n in range(nchains):
            if isrednoise == False:
                currchisq[n]  += np.sum((ymodels[n] - y[j])**2 / sigma[j]**2)
                #currchisq[n]  += mc.chisq(ymodels[n], y[j], sigma[j])
            else:
                currchisq[n]  += noisefunc(noisepars[n], ymodels[n]-y[j], wavelet)
            # Apply prior, if one exists
            if len(fit[j].ipriors) > 0:
                pbar   = fit[j].priorvals[:,0]  #prior mean
                psigma = np.zeros(len(pbar))    #prior standard deviation
                # Determine psigma based on which side of asymmetric Gaussian nextp is on
                for i in range(len(fit[j].ipriors)):
                    if nextp[n,fit[j].ipriors[i]] < pbar[i]:
                        psigma[i] = fit[j].priorvals[i,1]
                    else:
                        psigma[i] = fit[j].priorvals[i,2]
                    #currchisq[n] += fit[j].nobj*((nextp[n,fit[j].ipriors[i]] - pbar[i])/psigma[i])**2
                    currchisq[n] += ((nextp[n,fit[j].ipriors[i]] - pbar[i])/psigma[i])**2

    bestchisq = currchisq[0]

    #GENERATE RANDOM NUMBERS FOR MCMC
    numnotfixed = len(inotfixed)
    unif        = npr.rand(numit,nchains)

    #START TIMER
    clock = timer.Timer(numit,progress = np.arange(0.05,1.01,0.05))

    #Run Differential Evolution Monte Carlo algorithm 'numit' times
    b = gamma*stepsize[ifree]/100.
    for m in range(numit):
        for n in range(nchains):
            #Generate next step using differential evolution
            rand1, rand2 = npr.randint(0,nchains,2)
            while rand1 == n or rand2 == n or rand1 == rand2:
                rand1, rand2 = npr.randint(0,nchains,2)
            nextp[n,ifree] = params[n,ifree] + gamma*(params[rand1,ifree]-params[rand2,ifree]) \
                                             + npr.normal(0, b, numfree)
            #CHECK FOR NEW STEPS OUTSIDE BOUNDARIES
            ioutside     = np.where(np.bitwise_or(nextp[n] < pmin, nextp[n] > pmax))[0]
            if (len(ioutside) > 0):
                nextp[n,ioutside]    = np.copy(params[n,ioutside])
                outside[n,ioutside] += 1
        #UPDATE PARAMTER(S) EQUAL TO OTHER PARAMETER(S)
        if (ishare.size > 0):
            for s in range(ishare.size):
                nextp[:,ishare[s]] = nextp[:,int(abs(stepsize[ishare[s]])-1)]
        # Construct non-analytic systematic model
        for nn in np.unique(nights):
            tonight   = np.where(nights == nn)[0]
            if hasattr(fit[tonight[0]], 'whiteparams') and fit[tonight[0]].whiteparams != None:
                pass
            elif hasattr(fit[tonight[0]], 'iswhitelc') and fit[tonight[0]].iswhitelc != False:
                whitemodel = np.zeros((nchains,len(fit[tonight[0]].good)))
                weight     = np.zeros((nchains,len(fit[tonight[0]].good)))
                for n in range(nchains):
                    for j in tonight:
                        k = 0
                        for i in range(cummodels[j],cummodels[j+1]):
                            if functype[i] == 'ecl/tr':
                                specmodel = myfuncs[i](nextp[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
                                specmodeluc = np.zeros(len(fit[j].clipmask))
                                specmodeluc[fit[j].isclipmask] = specmodel
                                whitemodel[n,fit[j].isgood] += specmodeluc
                                weight    [n,fit[j].isgood] += specmodel[0]
                            k    += 1
                    whitemodel[n] /= weight[n]
                    #FINDME: Need to determine exact anchor point
                    #Also modify statement in w6model.py
                    #slope      = fit[0].iswhitelc / (1-whitemodel[n].min())
                    #offset     = 1 - slope
                    #whitemodel[n] = slope*whitemodel[n] + offset
                for j in tonight:
                    fit[j].whitemodel = whitemodel
            else:
                for j in tonight:
                    fit[j].whitemodel = np.ones((nchains,len(fit[j].good)))
        # Assemble systematics models
        systematics = [[] for n in range(nchains)]
        for n in range(nchains):
            for j in range(numevents):
                systematics[n].append(fit[j].whitelc*fit[j].refspeclc/(fit[j].whitemodel[n][fit[j].isgood].flatten())[fit[j].isclipmask])
                #systematics[n].append(((fit[j].whitelc/whitemodel[n])[fit[j].isgood].flatten())[fit[j].isclipmask])
        #COMPUTE NEXT CHI SQUARED AND ACCEPTANCE VALUES
        pedit        = np.copy(nextp)
        nextchisq    = np.zeros(nchains)
        if ncpu == 1:
            # Only 1 CPU
            for j in range(numevents):
                nextchisq += calcChi2(nchains, functype, myfuncs, pedit, nextp, iortholist[j], funcx, cummodels, numparams, j, isrednoise=isrednoise, wavelet=wavelet, noisefunc=noisefunc, systematics=systematics)
        else:
            # Multiple CPUs
            # Code works but is less efficient
            pool = mp.Pool(ncpu)
            for j in range(numevents):
                res = pool.apply_async(calcChi2, args=(nchains, functype, myfuncs, pedit, nextp, iortholist[j], funcx, cummodels, numparams, j, isrednoise, wavelet, noisefunc, systematics), callback=writeChi2)

            pool.close()
            pool.join()
            res.wait()

        #CALCULATE ACCEPTANCE PROBABILITY
        accept = np.exp(0.5 * (currchisq - nextchisq))
        for n in range(nchains):
            if (accept[n] >= 1) or (unif[m,n] <= accept[n]):
                #ACCEPT STEP
                numaccept    += 1
                params[n]     = np.copy(nextp[n])
                currchisq[n]  = nextchisq[n]
                if (currchisq[n] < bestchisq):
                    bestp     = np.copy(params[n])
                    bestchisq = np.copy(currchisq[n])

        allparams[:,:,m] = params.T
        #PRINT INTERMEDIATE INFO
        if ((m+1) % intsteps == 0) and (m > 0):
            print("\n" + time.ctime())
            #print("Number of times parameter tries to step outside its prior:")
            #print(outside)
            print("Current Best Parameters: ")
            print(bestp)

            #Apply Gelman-Rubin statistic
            if isGR:
                #Check for no accepted steps in each chain
                stdev   = np.std(allparams[inotfixed[0],:,:m+1],axis=1)
                ichain  = np.where(stdev > 1e-8)[0]
                #Call test
                foo = allparams[inotfixed]
                psrf, meanpsrf = gr.convergetest(foo[:,ichain,:m+1], len(ichain))
                #psrf, meanpsrf = gr.convergetest(allparams[inotfixed,:,:m+1], nchains)
                numconv = np.sum(np.bitwise_and(psrf < 1.01, psrf >= 1.00))
                print("Gelman-Rubin statistic for free parameters:")
                print(psrf)
                if numconv == numnotfixed: #and j >= 1e4:
                    print("All parameters have converged to within 1% of unity. Halting MCMC.")
                    allparams = allparams[:,:,:m+1]
                    break
        clock.check(m+1)

    #Check for no accepted steps in each chain
    stdev   = np.std(allparams[inotfixed[0]],axis=1)
    ichain  = np.where(stdev > 1e-8)[0]
    print("Number of good chains: " + str(len(ichain)))
    #print(len(ichain), ichain)
    #print(stdev)
    allparams = allparams[:,ichain]
    allparams = np.reshape(allparams,(nump, (m+1)*len(ichain)))
    return allparams, bestp, numaccept, (m+1)*len(ichain)



def demcz(y, pars, stdpburnin, pmin, pmax, stepsize, numit, sigma, numparams, cummodels, functype, myfuncs, funcx, iortholist, nights, fits, gamma=None, isGR=True, ncpu=1):
    """
    This function uses a differential evolution Markov chain with fewer chains to assess uncertainties.

    PARAMETERS
    ----------
    y:         Array containing dependent data
    Params:    Array of initial guess for parameters
    stdpburnin:Standard deviation of allparams from burn-in
    Pmin:      Array of parameter minimum values
    Pmax:      Array of parameter maximum values
    stepsize:  Array of 1-sigma change in parameter per iteration
    Numit:	   Number of iterations to perform
    Sigma:	   Standard deviation of data noise in y
    Numparams: Number of parameters for each model
    Cummodels: Cumulative number of models used
    Functype:  Define function type (eclipse, ramp, ip, etc), see models.py
    Myfuncs:   Pointers to model functions
    Funcx:	   Array of x-axis values for myfuncs
    fit:       List of fit objects
    gamma:     Multiplication factor in parameter differential, establishes acceptance rate

    OUTPUTS
    -------
    This function returns an array of the best fitting parameters,
    an array of all parameters over all iterations, and numaccept.

    REFERENCES
    ----------
    Cajo J. F. Ter Braak, "Differential Evolution Markov Chain with snooker updater and fewer chains" Stat Comput, 2008.

    HISTORY
    -------
    Adapted from mcmc.py                            August 2012
        Kevin Stevenson, UChicago
    Multiplied prior by number of points in fit     January 2014
    Adapted from demc()                             August 2014

    """
    global nextchisq, fit, data, unc
    fit   = fits
    data  = y
    unc   = sigma

    params          = np.copy(pars)
    nchains, nump   = params.shape
    nextp           = np.copy(params)       #Proposed parameters
    bestp           = np.copy(params[0])    #Best-fit parameters
    pedit           = np.copy(params)       #Editable parameters

    numaccept       = 0
    ifixed          = np.where(stepsize == 0)[0]    #Indices of fixed parameters
    inotfixed       = np.where(stepsize != 0)[0]    #Indices of non-fixed parameters
    ishare          = np.where(stepsize < 0)[0]     #Indices of shared parameters
    ifree           = np.where(stepsize > 0)[0]     #Indices of free parameters
    #outside         = np.zeros((nchains, nump))
    numevents       = len(fit)
    intsteps        = np.min((numit/5,1e5))         #Number of steps before checking G-R statistic
    isrednoise      = False
    wavelet         = None
    noisefunc       = None
    numfree         = len(ifree)
    print("Number of free parameters: " + str(len(ifree)))
    if gamma == None:
        gamma     = 2.38/np.sqrt(2*numfree)
    print('Gamma = ' + str(gamma))

    #UPDATE PARAMETER(S) EQUAL TO OTHER PARAMETER(S)
    if (ishare.size > 0):
        ishareptr = []
        for s in range(ishare.size):
            ishareptr.append(int(abs(stepsize[ishare[s]])-1))   #Pointer to where parameter is shared from
            params[:,ishare[s]] = params[:,ishareptr[s]]

    #INITIALIZE FIRST 10*numfree PARAMETERS IN allparams
    ninit   = 10*numfree
    numit  += ninit
    #if numit < (ninit + 10):
    #    numit == 1ninit + 10
    allparams       = np.zeros((nump, nchains, numit))  #allparams must be 64-bit!
    #Populate fixed parameters
    allparams[ifixed,:,:ninit] = params[:,ifixed].T[:,:,np.newaxis]
    #Populate free parameters
    for p in ifree:
        allparams[p,:,:ninit] = np.random.normal(params[0,p],stdpburnin[p],[nchains,ninit])
    #Update shared parameters
    if (ishare.size > 0):
        allparams[ishare,:,:ninit] = allparams[ishareptr,:,:ninit]

    # Construct non-analytic systematic model
    for nn in np.unique(nights):
        tonight   = np.where(nights == nn)[0]
        if hasattr(fit[tonight[0]], 'whiteparams') and fit[tonight[0]].whiteparams != None:
            if type(fit[tonight[0]].whiteparams) == type(np.array([])):
                #Only 1 model in model for white LC, grandfathered code
                #print("WARNING: You are using grandfathered code.  Update whiteparams to handle multiple models.")
                #whitemodel  = np.zeros((nchains,len(fit[tonight[0]].good)))
                i = int(fit[tonight[0]].whiteparams[0])
                whitemodel = myfuncs[i](fit[tonight[0]].whiteparams[1:], fit[tonight[0]].tuall, None)
            else:
                #Any number of models can be used to build white LC
                whitemodel  = np.ones((nchains,len(fit[tonight[0]].good)))
                for k in range(len(fit[tonight[0]].whiteparams)):
                    i = int(fit[tonight[0]].whiteparams[k][0])
                    whitemodel *= myfuncs[i](fit[tonight[0]].whiteparams[k][1:], fit[tonight[0]].tuall, None)
                    #whitemodel *= myfuncs[i](fit[tonight[0]].whiteparams[k][1:], funcxuc[i], None)
            for j in tonight:
                fit[j].whitemodel = whitemodel
        elif hasattr(fit[tonight[0]], 'iswhitelc') and fit[tonight[0]].iswhitelc != False:
            whitemodel = np.zeros((nchains,len(fit[tonight[0]].good)))
            weight     = np.zeros((nchains,len(fit[tonight[0]].good)))
            for n in range(nchains):
                for j in tonight:
                    k = 0
                    for i in range(cummodels[j],cummodels[j+1]):
                        if functype[i] == 'ecl/tr':
                            specmodel = myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
                            specmodeluc = np.zeros(len(fit[j].clipmask))
                            specmodeluc[fit[j].isclipmask] = specmodel
                            whitemodel[n,fit[j].isgood] += specmodeluc
                            weight    [n,fit[j].isgood] += specmodel[0]
                        k    += 1
                whitemodel[n] /= weight[n]
                #FINDME: Need to determine exact anchor point
                #slope      = fit[0].iswhitelc / (1-whitemodel[n].min())
                #offset     = 1 - slope
                #whitemodel[n] = slope*whitemodel[n] + offset
            for j in tonight:
                fit[j].whitemodel = whitemodel
        else:
            for j in tonight:
                fit[j].whitemodel = np.ones((nchains,len(fit[j].good)))

    #Calc chi-squared for model type using current params
    currchisq   = np.zeros(nchains)
    noisepars   = [[] for i in range(nchains)]
    for j in range(numevents):
        #Build final model from model components
        ymodels     = np.ones((nchains, fit[j].nobj))
        k           = 0
        for i in range(cummodels[j],cummodels[j+1]):
            for n in range(nchains):
                if   functype[i] == 'ortho':
                    #MODIFY COPY OF nextp ONLY
                    pedit[n,iortholist[j]] = myfuncs[i](pedit[n,iortholist[j]], funcx[i], fit[j].etc[k])
                elif (functype[i] == 'ipmap') or (functype[i] == 'spline'):
                    ymodels[n] *= myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], ymodels[n])
                elif functype[i] == 'posoffset':
                    # Record change in Position 0 => cannot orthogonalize position parameters
                    ymodels[n] *= myfuncs[i](nextp[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
                elif hasattr(fit[j], 'timebins') and (functype[i] == 'ecl/tr'
                                                  or  functype[i] == 'ramp'
                                                  or  functype[i] == 'sinusoidal'):
                    # Average over high-resolution model
                    hiresmodel = myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
                    if len(fit[j].timebins) == fit[j].nobj:
                        for tb in range(len(fit[j].timebins)):
                            ymodels[n,tb] *= np.mean(hiresmodel[fit[j].timebins[tb]])
                    else:
                        for tb in range(len(fit[j].timebinsuc)):
                            ymodels[n,tb] *= np.mean(hiresmodel[fit[j].timebinsuc[tb]])
                elif functype[i] == 'noise':
                    # Set up for modified chi-squared calculation using correlated noise
                    isrednoise   = True
                    wavelet      = fit[j].etc[k]
                    noisefunc    = myfuncs[i]
                    noisepars[n] = pedit[n,numparams[i]:numparams[i+1]]
                else:
                    ymodels[n] *= myfuncs[i](pedit[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
            k += 1
            #Multiply analytic model by non-analytic systematics model
            systematics = fit[j].whitelc*fit[j].refspeclc/(fit[j].whitemodel[n][fit[j].isgood].flatten())[fit[j].isclipmask]
            ymodels[n] *= systematics
        # Calculate chi^2
        for n in range(nchains):
            if isrednoise == False:
                #currchisq[n]  += mc.chisq(ymodels[n], y[j], sigma[j])
                currchisq[n]  += np.sum((ymodels[n] - y[j])**2 / sigma[j]**2)
            else:
                currchisq[n]  += noisefunc(noisepars[n], ymodels[n]-y[j], wavelet)
            # Apply prior, if one exists
            if len(fit[j].ipriors) > 0:
                pbar   = fit[j].priorvals[:,0]  #prior mean
                psigma = np.zeros(len(pbar))    #prior standard deviation
                # Determine psigma based on which side of asymmetric Gaussian nextp is on
                for i in range(len(fit[j].ipriors)):
                    if nextp[n,fit[j].ipriors[i]] < pbar[i]:
                        psigma[i] = fit[j].priorvals[i,1]
                    else:
                        psigma[i] = fit[j].priorvals[i,2]
                    #currchisq[n] += fit[j].nobj*((nextp[n,fit[j].ipriors[i]] - pbar[i])/psigma[i])**2
                    currchisq[n] += ((nextp[n,fit[j].ipriors[i]] - pbar[i])/psigma[i])**2

    bestchisq = currchisq[0]

    #GENERATE RANDOM NUMBERS FOR MCMC
    unif        = npr.rand(numit,nchains)           #Acceptance
    snooker     = npr.rand(numit,nchains)           #If <0.1, set gamma = 1 during step
    randchain   = npr.randint(0,nchains,[2,numit,nchains])  #Pairs of chains
    b = gamma*stepsize[ifree]/100.
    epsilon     = npr.normal(0, b, [numit,nchains,numfree])

    #START TIMER
    clock = timer.Timer(numit-ninit,progress = np.arange(0.05,1.01,0.05))

    #Run Differential Evolution Monte Carlo algorithm 'numit' times
    numnotfixed = len(inotfixed)
    for m in range(ninit,numit):
        '''
        #Code below is slower, possibly because array copies are made???
        #Generate next step using differential evolution
        randstep    = npr.randint(0,m,[2,nchains])
        nextp[:,ifree] = params[:,ifree] + gamma*(allparams[ifree][:,randchain[0,m],randstep[0]] \
                                                - allparams[ifree][:,randchain[1,m],randstep[1]]).T \
                                                + epsilon[m]
        '''
        randstep    = npr.randint(0,m,[2,nchains])
        for n in range(nchains):
            #Generate next step using differential evolution
            if snooker[m,n] < 0.1:
                #Set gamma = 1 to jump between modes (bimodal distribution)
                nextp[n,ifree] = params[n,ifree] + (allparams[ifree,randchain[0,m,n],randstep[0,n]] \
                                                  - allparams[ifree,randchain[1,m,n],randstep[1,n]]) \
                                                  + epsilon[m,n]
            else:
                nextp[n,ifree] = params[n,ifree] + gamma*(allparams[ifree,randchain[0,m,n],randstep[0,n]] \
                                                        - allparams[ifree,randchain[1,m,n],randstep[1,n]]) \
                                                        + epsilon[m,n]
        #CHECK FOR NEW STEPS OUTSIDE BOUNDARIES
        ioutside     = np.where(np.bitwise_or(nextp < pmin, nextp > pmax))
        if (len(ioutside) > 0):
            nextp[ioutside]    = np.copy(params[ioutside])
        #UPDATE PARAMTER(S) EQUAL TO OTHER PARAMETER(S)
        if (ishare.size > 0):
            nextp[:,ishare] = nextp[:,ishareptr]
        # Construct non-analytic systematic model
        for nn in np.unique(nights):
            tonight   = np.where(nights == nn)[0]
            if hasattr(fit[tonight[0]], 'whiteparams') and fit[tonight[0]].whiteparams != None:
                pass
            elif hasattr(fit[tonight[0]], 'iswhitelc') and fit[tonight[0]].iswhitelc != False:
                print("***WARNING: whiteparams not defined.***")
                whitemodel = np.zeros((nchains,len(fit[tonight[0]].good)))
                weight     = np.zeros((nchains,len(fit[tonight[0]].good)))
                for n in range(nchains):
                    for j in tonight:
                        k = 0
                        for i in range(cummodels[j],cummodels[j+1]):
                            if functype[i] == 'ecl/tr':
                                specmodel = myfuncs[i](nextp[n,numparams[i]:numparams[i+1]], funcx[i], fit[j].etc[k])
                                specmodeluc = np.zeros(len(fit[j].clipmask))
                                specmodeluc[fit[j].isclipmask] = specmodel
                                whitemodel[n,fit[j].isgood] += specmodeluc
                                weight    [n,fit[j].isgood] += specmodel[0]
                            k    += 1
                    whitemodel[n] /= weight[n]
                    #FINDME: Need to determine exact anchor point
                    #Also modify statement in w6model.py
                    #slope      = fit[0].iswhitelc / (1-whitemodel[n].min())
                    #offset     = 1 - slope
                    #whitemodel[n] = slope*whitemodel[n] + offset
                for j in tonight:
                    fit[j].whitemodel = whitemodel
            else:
                for j in tonight:
                    fit[j].whitemodel = np.ones((nchains,len(fit[j].good)))
        # Assemble systematics models
        systematics = [[] for n in range(nchains)]
        for n in range(nchains):
            for j in range(numevents):
                systematics[n].append(fit[j].whitelc*fit[j].refspeclc/(fit[j].whitemodel[n][fit[j].isgood].flatten())[fit[j].isclipmask])
                #systematics[n].append(((fit[j].whitelc/whitemodel[n])[fit[j].isgood].flatten())[fit[j].isclipmask])
        #COMPUTE NEXT CHI SQUARED AND ACCEPTANCE VALUES
        pedit        = np.copy(nextp)
        nextchisq    = np.zeros(nchains)
        if ncpu == 1:
            # Only 1 CPU
            for j in range(numevents):
                nextchisq += calcChi2(nchains, functype, myfuncs, pedit, nextp, iortholist[j], funcx, cummodels, numparams, j, isrednoise=isrednoise, wavelet=wavelet, noisefunc=noisefunc, systematics=systematics)
        else:
            # Multiple CPUs
            # Code works but is less efficient
            pool = mp.Pool(ncpu)
            for j in range(numevents):
                res = pool.apply_async(calcChi2, args=(nchains, functype, myfuncs, pedit, nextp, iortholist[j], funcx, cummodels, numparams, j, isrednoise, wavelet, noisefunc, systematics), callback=writeChi2)

            pool.close()
            pool.join()
            res.wait()

        #CALCULATE ACCEPTANCE PROBABILITY
        accept = np.exp(0.5 * (currchisq - nextchisq))
        for n in range(nchains):
            if (accept[n] >= 1) or (unif[m,n] <= accept[n]):
                #ACCEPT STEP
                numaccept    += 1
                params[n]     = np.copy(nextp[n])
                currchisq[n]  = nextchisq[n]
                if (currchisq[n] < bestchisq):
                    bestp     = np.copy(params[n])
                    bestchisq = np.copy(currchisq[n])

        allparams[:,:,m] = params.T
        #PRINT INTERMEDIATE INFO
        if ((m+1-ninit) % intsteps == 0) and (m > ninit):
            print("\n" + time.ctime())
            #print("Number of times parameter tries to step outside its prior:")
            #print(outside)
            print("Current Best Parameters: ")
            print(bestp)

            #Apply Gelman-Rubin statistic
            if isGR:
                #Check for no accepted steps in each chain
                stdev   = np.std(allparams[inotfixed[0],:,ninit:m+1],axis=1)
                ichain  = np.where(stdev > 1e-8)[0]
                if len(ichain) > 1:
                    #Call test
                    foo = allparams[inotfixed]
                    psrf, meanpsrf = gr.convergetest(foo[:,ichain,ninit:m+1], len(ichain))
                    #psrf, meanpsrf = gr.convergetest(allparams[inotfixed,:,:m+1], nchains)
                    numconv = np.sum(np.bitwise_and(psrf < 1.01, psrf >= 1.00))
                    print("Gelman-Rubin statistic for free parameters:")
                    print(psrf)
                    if numconv == numnotfixed: #and j >= 1e4:
                        print("All parameters have converged to within 1% of unity. Halting MCMC.")
                        allparams = allparams[:,:,:m+1]
                        break
        clock.check(m+1-ninit)

    #Check for no accepted steps in each chain
    stdev   = np.std(allparams[inotfixed[0]],axis=1)
    ichain  = np.where(stdev > 1e-8)[0]
    print("Number of good chains: " + str(len(ichain)))
    #FINDME
    allparams = allparams[:,ichain,ninit:]
    allparams = np.reshape(allparams,(nump, (m+1-ninit)*len(ichain)))
    #allparams = allparams[:,ichain]
    #allparams = np.reshape(allparams,(nump, (m+1)*len(ichain)))
    return allparams, bestp, numaccept, (m+1-ninit)*len(ichain)
