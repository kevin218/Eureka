
import numpy as np

def posflux(posparams, x, etc = []):
    """
  This function creates a model that fits the flux at each position

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
    nobj, wherepos = x
    normfactors = np.ones(nobj)
    npos        = len(wherepos)
    #SET posparams[0] = 1/ (PRODUCT OF posparams[1:])
    posparams[0] = 1/np.product(posparams[1:npos])
    for i in range(npos):
        normfactors[wherepos[i]] = posparams[i]
    
    return normfactors
