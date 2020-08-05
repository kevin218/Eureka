
#Non-Analytic Systematic Corrector
import numpy as np
import scipy.interpolate as spi

def refStarSpecLC(ev, fit, subtonight, tonight):
    '''
    Calculate spectroscopic correction using a single reference star.
    
    PARAMETERS
    ----------
    subtonight  : List of indices indicating clean light curve
    tonight     : List of indices of events belonging to the same night
    '''
    
    if hasattr(ev[tonight[0]].params, 'nrefStar') and ev[tonight[0]].params.nrefStar != None:
        numevents  = len(tonight)
        numpts     = len(ev[tonight[0]].aplev)
        rflux      = np.zeros((numevents,numpts))   #Array of reference star flux for each channel
        
        # Assign ref star flux values
        i = 0
        for k in tonight:
            rflux[i]    = ev[k].photflux[ev[0].params.nrefStar+ev[k].chip]
            #rfluxerr[i] = ev[k].photfluxerr[nref+ev[k].chip]
            i          += 1
        
        # Calculate ref star white LC
        refwhitelc  = np.sum(rflux[subtonight-tonight[0]],axis=0)
        refwhitelc /= np.mean(refwhitelc)
        # Calculate ref star spec component of LC
        refspeclc   = rflux/refwhitelc
        # Normalize and assign to fit
        i = 0
        for k in tonight:
            ev[k].refspeclc    = refspeclc[i]/np.mean(refspeclc[i])
            fit[k].refspeclcuc = ev[k].refspeclc[fit[k].isgood]
            fit[k].refspeclc   = fit[k].refspeclcuc[fit[k].isclipmask]
            i                 += 1
        '''
        plt.figure(1)
        plt.clf()
        for i in range(numevents):
            plt.subplot(3,6,i+1)
            plt.plot(refspeclc[i],'o-')
        '''
    else:
        for k in tonight:
            ev[k].refspeclc    = np.ones(len(ev[tonight[0]].aplev))
            fit[k].refspeclcuc = ev[k].refspeclc[fit[k].isgood]
            fit[k].refspeclc   = fit[k].refspeclcuc[fit[k].isclipmask]
        
    return


def refStarSpecLC2(ev, fit, tonight):
    '''
    Calculate spectroscopic correction using multiple reference stars.
    
    PARAMETERS
    ----------
    subtonight  : List of indices indicating clean light curve
    tonight     : List of indices of events belonging to the same night
    '''
    if hasattr(ev[tonight[0]].params, 'nrefStar') and ev[tonight[0]].params.nrefStar != None:
        irefwhitelc = ev[tonight[0]].params.irefwhitelc
        irefspeclc  = ev[tonight[0]].params.irefspeclc
        numevents  = len(tonight)
        numpts     = len(ev[tonight[0]].aplev)
        numref     = len(ev[tonight[0]].params.nrefStar)
        rflux      = np.zeros((numref,numevents,numpts))   #Array of reference star flux for each channel
        refspeclc  = np.zeros((numref,numevents,numpts))   #Array of reference star flux for each channel
        
        # Assign ref star flux values
        for j in range(numref):
            i = 0
            for k in tonight:
                if hasattr(ev[k], 'chip'):
                    rflux[j,i]  = ev[k].photflux[ev[tonight[0]].params.nrefStar[j]+ev[k].chip]
                else:
                    rflux[j,i]  = ev[k].photflux[ev[tonight[0]].params.nrefStar[j]]
                #rfluxerr[i] = ev[k].photfluxerr[nref+ev[k].chip]
                i          += 1
            # Calculate ref star white LC
            refwhitelc   = np.sum(rflux[j,irefwhitelc[j]-tonight[0]],axis=0)
            #refwhitelc  /= np.mean(refwhitelc)
            # Calculate ref star spec component of LC
            refwhitelc[np.where(refwhitelc==0)[0]] = -1.
            refspeclc[j,irefspeclc[j]] = rflux[j,irefspeclc[j]]/refwhitelc*np.mean(refwhitelc)
        
        # Normalize and assign to fit
        i = 0
        #refspeclc[np.where(refspeclc==0)] = 1e-6
        for k in tonight:
            ev[k].refspeclc    = np.sum(refspeclc[:,i],axis=0)/np.mean(refspeclc[:,i])
            fit[k].refspeclcuc = ev[k].refspeclc[fit[k].isgood]
            fit[k].refspeclc   = fit[k].refspeclcuc[fit[k].isclipmask]
            i                 += 1
        '''
        import matplotlib.pyplot as plt
        plt.figure(1,figsize=(16,10))
        plt.clf()
        for i in range(numevents):
            plt.subplot(3,6,i+1)
            plt.plot(refspeclc[0,i],'o-')
        plt.savefig('fig1.png')
        plt.pause(5)
        plt.figure(2,figsize=(16,10))
        plt.clf()
        for i in range(numevents):
            plt.subplot(3,6,i+1)
            plt.plot(refspeclc[1,i],'o-')
        plt.savefig('fig2.png')
        plt.pause(5)
        '''
    else:
        for k in tonight:
            ev[k].refspeclc    = np.ones(len(ev[tonight[0]].aplev))
            fit[k].refspeclcuc = np.ones(len(fit[k].fluxuc))
            fit[k].refspeclc   = np.ones(len(fit[k].flux))
        
    return

def constructWhiteLC(ev, fit, tonight, night):
    '''
    Construct white light curve from weighted spectroscopic lights curves
    '''
    if hasattr(ev[tonight[0]].params, 'whiteparams') and ev[tonight[0]].params.whiteparams != None:
        print("Using given parameters to construct white transit model for night " + str(night) + ".")
        for j in tonight:
            fit[j].whiteparams = ev[tonight[0]].params.whiteparams
    if hasattr(ev[tonight[0]].params, 'iswhitelc') and ev[tonight[0]].params.iswhitelc != False:
        #irefwhitelc = ev[tonight[0]].params.irefwhitelc
        #print("Constructing white light curve from good data for night " + str(k) + ".")
        # Construct white LC from good data (not clipped data)
        whitelc = np.zeros(len(ev[tonight[0]].aplev))
        weight  = np.zeros(len(ev[tonight[0]].aplev))
        for j in tonight:
            whitelc[fit[j].isgood] += fit[j].fluxuc
            weight [fit[j].isgood] += np.mean(fit[j].fluxuc)
        whitelc[np.where(weight>0)] = whitelc[np.where(weight>0)]/weight[np.where(weight>0)]
        for k in tonight:
            fit[k].iswhitelc = ev[tonight[0]].params.iswhitelc
            ev[k].whitelc    = whitelc
            fit[k].whitelcuc = ev[k].whitelc[fit[k].isgood]
            fit[k].whitelc   = fit[k].whitelcuc[fit[k].isclipmask]
        """
        #FINDME: Plot
        plt.figure(1)
        plt.clf()
        plt.plot(ev[0].bjdtdb, whitelc, 'o')
        """
    else:
        for k in tonight:
            fit[k].iswhitelc = False
            ev[k].whitelc    = np.ones(len(ev[tonight[0]].aplev))
            fit[k].whitelcuc = np.ones(len(fit[k].fluxuc))
            fit[k].whitelc   = np.ones(len(fit[k].flux))
    return

def nasmodel(event, fit, params, iparams, cummodels, functype, myfuncs, funcxuc, tonight):
    '''
    Construct non-analytic systematic model through division of white light curve by white transit model.
    Construct best white LC model for each fit
    '''
    if hasattr(event[tonight[0]].params, 'whiteparams') and event[tonight[0]].params.whiteparams != None:
        if type(event[tonight[0]].params.whiteparams) == type(np.array([])):
            #Only 1 model in model for white LC, grandfathered code
            print("WARNING: You are using grandfathered code.  Update whiteparams to handle multiple models.")
            i = int(event[tonight[0]].params.whiteparams[0])
            whitemodel = myfuncs[i](event[tonight[0]].params.whiteparams[1:], fit[tonight[0]].tuall, None)
            #whitemodel = myfuncs[i](event[0].params.whiteparams[1:], funcxuc[i], None)
            for j in tonight:
                event[j].bestsys = event[j].whitelc/whitemodel*event[j].refspeclc
                fit[j].bestsysuc = fit[j].whitelcuc/whitemodel[fit[j].isgood]*fit[j].refspeclcuc
                fit[j].bestsys   = fit[j].bestsysuc[fit[j].isclipmask]
        else:
            #Any number of models can be used to build white LC
            whitemodel  = np.ones(len(event[tonight[0]].aplev))
            for k in range(len(event[tonight[0]].params.whiteparams)):
                i = int(event[tonight[0]].params.whiteparams[k][0])
                whitemodel *= myfuncs[i](event[tonight[0]].params.whiteparams[k][1:], fit[tonight[0]].tuall, None)
                #whitemodel *= myfuncs[i](event[tonight[0]].params.whiteparams[k][1:], funcxuc[i], None)
            for j in tonight:
                event[j].bestsys = event[j].whitelc/whitemodel*event[j].refspeclc
                fit[j].bestsysuc = fit[j].whitelcuc/whitemodel[fit[j].isgood]*fit[j].refspeclcuc
                fit[j].bestsys   = fit[j].bestsysuc[fit[j].isclipmask]
    elif hasattr(event[tonight[0]].params, 'iswhitelc') and event[tonight[0]].params.iswhitelc != False:
        whitemodel = np.zeros(len(event[tonight[0]].aplev))
        weight     = np.zeros(len(event[tonight[0]].aplev))
        for j in tonight:
            k = 0
            for i in range(cummodels[j],cummodels[j+1]):
                if functype[i] == 'ecl/tr':
                    specmodel = myfuncs[i](params[iparams[i]], funcxuc[i], fit[j].etc[k])
                    whitemodel[fit[j].isgood] += specmodel
                    weight    [fit[j].isgood] += specmodel[0]
                k    += 1
        for j in tonight:
            event[j].bestsys = event[j].whitelc/whitemodel*weight*event[j].refspeclc
            fit[j].bestsysuc = fit[j].whitelcuc*(weight/whitemodel)[fit[j].isgood]*fit[j].refspeclcuc
            fit[j].bestsys   = fit[j].bestsysuc[fit[j].isclipmask]
        #whitemodel /= weight
        #FINDME: Need to determine exact anchor point
        #Also modify statements in demc.py
        #slope      = event[0].params.iswhitelc / (1-whitemodel.min())
        #offset     = 1 - slope
        #whitemodel = slope*whitemodel + offset
    else:
        for j in tonight:
            event[j].bestsys = np.ones(len(event[tonight[0]].aplev))*event[j].refspeclc
            fit[j].bestsysuc = np.ones(fit[j].nobjuc)*fit[j].refspeclcuc
            fit[j].bestsys   = np.ones(fit[j].nobj)
        #whitemodel = np.ones(len(event[tonight[0]].aplev))
    
    #fit[k].systematics = fit[k].whitelc/whitemodel*weight
    return 

def whiteLCmodel3(event, fit, params, iparams, numevents, cummodels, functype, myfuncs, funcxuc, tonight):
    '''
    Construct white LC model for HST plots
    '''
    if hasattr(event[tonight[0]].params, 'whiteparams') and event[tonight[0]].params.whiteparams != None:
        if type(event[tonight[0]].params.whiteparams) == type(np.array([])):
            #Only 1 model in model for white LC, grandfathered code
            #print("WARNING: You are using grandfathered code.  Update whiteparams to handle multiple models.")
            i = event[tonight[0]].params.whiteparams[0]
            whitemodel2 = myfuncs[i](event[tonight[0]].params.whiteparams[1:], fit[tonight[0]].modelfuncx, None)
        else:
            #Any number of models can be used to build white LC
            whitemodel2  = np.ones(len(fit[tonight[0]].modelfuncx))
            for k in range(len(event[tonight[0]].params.whiteparams)):
                i = int(event[tonight[0]].params.whiteparams[k][0])
                whitemodel2 *= myfuncs[i](event[tonight[0]].params.whiteparams[k][1:], fit[tonight[0]].modelfuncx, None)
    elif hasattr(event[tonight[0]].params, 'iswhitelc') and event[tonight[0]].params.iswhitelc != False:
        whitemodel2 = np.zeros(fit[tonight[0]].nmodelpts)
        weight2     = np.zeros(fit[tonight[0]].nmodelpts)
        for j in range(numevents):
            k = 0
            for i in range(cummodels[j],cummodels[j+1]):
                if functype[i] == 'ecl/tr':
                    specmodel = myfuncs[i](params[iparams[i]], fit[j].modelfuncx, fit[j].etc[k])
                    whitemodel2 += specmodel
                    weight2     += specmodel[0]
                k    += 1
        whitemodel2 /= weight2
    else:
        whitemodel2 = np.ones(fit[tonight[0]].nmodelpts)
    
    # Interpolate white LC over modelfuncx
    for j in tonight:
        print("If the numbers below don't match and the code breaks, it's because one or more nights listed in the params file are wrong.")
        print(j,fit[j].tuall.shape,event[j].whitelc.shape)
        fit[j].systematics2 = spi.interp1d(fit[j].tuall, event[j].whitelc)(fit[j].modelfuncx)/whitemodel2
    
    return

"""
def whiteLCmodel2(event, fit, params, iparams, numevents, cummodels, functype, myfuncs, funcxuc):
    '''
    Construct white LC model for HST plots
    '''
    if hasattr(event[0].params, 'whiteparams') and event[0].params.whiteparams != None:
        if type(event[0].params.whiteparams) == type(np.array([])):
            #Only 1 model in model for white LC, grandfathered code
            #print("WARNING: You are using grandfathered code.  Update whiteparams to handle multiple models.")
            i = event[0].params.whiteparams[0]
            whitemodel2 = myfuncs[i](event[0].params.whiteparams[1:], fit[0].modelfuncx, None)
        else:
            #Any number of models can be used to build white LC
            whitemodel2  = np.ones(len(fit[0].modelfuncx))
            for k in range(len(event[0].params.whiteparams)):
                i = int(event[0].params.whiteparams[k][0])
                whitemodel2 *= myfuncs[i](event[0].params.whiteparams[k][1:], fit[0].modelfuncx, None)
    elif hasattr(event[0].params, 'iswhitelc') and event[0].params.iswhitelc != False:
        whitemodel2 = np.zeros(fit[0].nmodelpts)
        weight2     = np.zeros(fit[0].nmodelpts)
        for j in range(numevents):
            k = 0
            for i in range(cummodels[j],cummodels[j+1]):
                if functype[i] == 'ecl/tr':
                    specmodel = myfuncs[i](params[iparams[i]], fit[j].modelfuncx, fit[j].etc[k])
                    whitemodel2 += specmodel
                    weight2     += specmodel[0]
                k    += 1
        whitemodel2 /= weight2
    else:
        whitemodel2 = np.ones(fit[0].nmodelpts)
    
    # Interpolate white LC over modelfuncx
    whitelc2 = spi.interp1d(fit[0].tuall, event[0].whitelc)(fit[0].modelfuncx)
    
    return whitelc2/whitemodel2
"""
