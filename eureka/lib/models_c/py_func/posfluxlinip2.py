
import numpy as np

def posfluxlinip(params, x, etc):
    """
  This function creates a model that fits the median flux at each mosition

  Parameters
  ----------
    posparams:  Position parameters
    nobj:       Number of points
    wherepos:	Array of position point locations

  Returns
  -------
    This function returns an array of y values...

  Revisions
  ---------
  2010-01-20	Kevin Stevenson, UCF 
                kevin218@knights.ucf.edu
                Original Version
  2010-08-12    Kevin
                Updated for speed & posparams[0]
    """
    [y, x, q], nobj, wherepos = x
    y0, x0 = etc
    npos        = len(wherepos)
    posparams   = params[0:npos]
    ipparams    = params[9:]    #HARD-CODED VALUE!!!
    normfactors = np.ones(nobj)
    #SET posparams[0] = npos - (SUM OF posparams[1:])
    posparams[0] = npos - np.sum(posparams[1:npos])
    for i in range(npos):
        normfactors[wherepos[i]] = \
        ipparams[2*i]*(y[wherepos[i]]-y0[i]) + ipparams[2*i+1]*(x[wherepos[i]]-x0[i]) + posparams[i]
    
    return normfactors
