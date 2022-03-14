
import numpy as np
import time

def ballardip(ipparams, position, etc=[]):
    """
  This function fits the intra-pixel sensitivity effect using the method described by Ballard et al. (2010).

    Parameters
    ----------
	ipparams :  tuple
                unused
    y :         1D array, size = # of measurements
                Pixel position along y
    x :         1D array, size = # of measurements
                Pixel position along x
    flux :      1D array, size = # of measurements
                Observed flux at each position
    
    Returns
    -------
    weight :    1D array, size = # of measurements
                Normalized intrapixel-corrected flux multiplier
    
    Revisions
    ---------
    2010-12-20  Kevin Stevenson, UCF
			    kevin218@knights.ucf.edu
                Original version
    2011-01-01  kevin
                Modified with if statement
    """
    
    sigmay, sigmax, nbins = ipparams
    [y, x, q], weight, flux = position
    nbins = int(nbins)
    if len(etc) == 0:
        #IP EFFECT HAS ALREADY BEEN REMOVED, RETURN weight
        return weight
    else:
        f     = flux/etc[0]
        nobj  = y.size
        #CALCULATE IP EFFECT
        for i in range(nbins):
            start   = int(1.*i*nobj/nbins)
            end     = int(1.*(i+1)*nobj/nbins)
            s       = np.ones(nobj)
            s[start:end] = 0
            #EXCLUDE ECLIPSE REGION
            #s[np.where(fit[j].phase >= (params[fit[j].i.midpt] - params[fit[j].i.width]/2.))[0][0]:  \
            #  np.where(fit[j].phase >= (params[fit[j].i.midpt] + params[fit[j].i.width]/2.))[0][0]] = 0
            biny = np.mean(y[start:end])
            binx = np.mean(x[start:end])
            weight[start:end] = sum(np.exp(-0.5*((x-binx)/sigmax)**2) * \
                                    np.exp(-0.5*((y-biny)/sigmay)**2)*flux*s) / \
                                sum(np.exp(-0.5*((x-binx)/sigmax)**2) * \
                                    np.exp(-0.5*((y-biny)/sigmay)**2)*s)
            if (i % (nbins / 5.) == 0): 
                print(str(int(100.*i/nbins)) + "% complete at " + time.ctime())
    
    weight /= np.mean(weight)
    return weight
    '''
    for i in range(nbins):
        start   = int(1.*i*nobj/nbins)
        end     = int(1.*(i+1)*nobj/nbins)
        s       = np.ones(nobj)
        s[start:end] = 0
        biny = np.mean(y[start:end])
        binx = np.mean(x[start:end])
        weight[start:end] = sum(np.exp(-0.5*((x-binx)/sigmax)**2) * \
                                np.exp(-0.5*((y-biny)/sigmay)**2)*flux*s) / \
                            sum(np.exp(-0.5*((x-binx)/sigmax)**2) * \
                                np.exp(-0.5*((y-biny)/sigmay)**2)*s)
    '''

