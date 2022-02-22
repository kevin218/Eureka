import numpy as np

def heqramp(rampparams, t, etc):
    """
    This function creates a model that fits the HST 'hook' using a rising exponential.

    Parameters
    ----------
    goal:  goal as x -> inf
    r0:	   rise exp
    x0:	   time offset
    x:	   Array of time/phase points

    Returns
    -------
    This function returns an array of y values by combining an eclipse and a rising exponential

    Revisions
    ---------
    2014-06-09	Kevin Stevenson 
	            kbs@uchicago.edu
                Modified from hook2.py
    """

    t0      = rampparams[0]
    r0      = rampparams[1]
    r1      = rampparams[2]
    r2      = rampparams[3]
    r3      = rampparams[4]
    pm      = rampparams[5]
    period  = rampparams[6]
    
    t_batch     = (t-t[0]-t0) % period
    #for i in range(len(istartbatch)-1):
    #    t_batch[istartbatch[i]:istartbatch[i+1]] = t[istartbatch[i]:istartbatch[i+1]]-t[istartbatch[i]]
    
    return 1 + pm*np.exp(-r0*t_batch + r1) + r2*t_batch + r3*t_batch**2
