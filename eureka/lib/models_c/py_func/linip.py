
import numpy as np

def linip(ipparams, x, etc):
    """
  This function creates a model that fits the median flux at each mosition

  Parameters
  ----------
    ipparams:   Intrapixel parameters at each position
    nobj:       Number of points
    wherepos:	Array of position point locations

  Returns
  -------
    This function returns an array of y values...

  Revisions
  ---------
  2011-07-29	Kevin Stevenson, UCF 
                kevin218@knights.ucf.edu
                Original Version
    """
    [y, x, q], nobj, wherepos = x
    y0, x0 = etc
    output = np.ones(nobj)
    # Cycle through positions
    for i in range(len(wherepos)):
        output[wherepos[i]] = ipparams[2*i]*(y[wherepos[i]]-y0[i]) + ipparams[2*i+1]*(x[wherepos[i]]-x0[i]) + 1
    
    return output
