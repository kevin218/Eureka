
import numpy as np
import scipy.interpolate as spi

def unispline(params, timeflux, etc = []):
    """
    This function fits the stellar flux using a univariate spline.

    Parameters
    ----------
    nknots      : Number of knots
    k           : degree polynomial
    time        : BJD_TDB or phase
    flux        : flux
    etc         : model flux

    Returns
    -------
    This function returns an array of flux values...

    Revisions
    ---------
    2016-04-10	Kevin Stevenson, UChicago  
		        kbs@uchicago.edu
	            Original Version
    """
    nknots, k   = params
    time, flux  = timeflux
    iknots      = np.linspace(time[2], time[-3], nknots)
    splflux     = flux / etc
    
    sp          = spi.LSQUnivariateSpline(time, splflux, iknots, k=k)#, w=good)
    saplev      = sp(time)
    
    #tck = spi.bisplrep(xknots.flatten(), yknots.flatten(), ipparams, kx=3, ky=3)
    #print(tck)
    #tck = [yknots, xknots, ipparams, 3, 3]
    #func = spi.interp2d(xknots, yknots, ipparams, kind='cubic'
    #output = np.ones(time.size)
    #for i in range(time.size):
    #    output[i] = spi.bisplev(x[i], y[i], tck, dx=0, dy=0)
    
    return saplev/np.mean(saplev)
    '''
    #Splines
    ev[i].sp.nknots         = 70
    ev[i].sp.nsigma         = 8
    ev[i].sp.maxiter        = 500
    '''
        
