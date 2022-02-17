import numpy as np


def pcaramp(rampparams, t, etc = []):
    """
    This function creates a model that fits a ramp using a principal component analysis.

    Parameters
    ----------
    goal:  goal as t -> inf
    r0:	   curvature
    r1:	   time offset * curvature
    t:	   Array of time/phase points

    Returns
    -------
    This function returns an array of y values by combining an eclipse and a rising exponential

    Revisions
    ---------
    201-07-09	Kevin Stevenson, UCF  
	            kevin218@knights.ucf.edu
                Original version
    """
    if len(etc) > 0:
        #print(np.asarray(etc*rampparams[0:3][:, np.newaxis])[0].T)
        #print(np.asarray(etc*rampparams[0:3][:, np.newaxis]).T + rampparams[4:7])
        goal, r0, r1 = np.asarray(etc*rampparams[0:3][:, np.newaxis]).T[0] + rampparams[4:7]
    else:
        goal, r0, r1 = rampparams[0:3] + rampparams[4:7]
    
    pm    = rampparams[3]

    return goal + pm*np.exp(-r0*t + r1)

