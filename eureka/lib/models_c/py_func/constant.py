import batman
import numpy as np

def constant(rampparams, t, etc = []):
    '''
    Parameters
    ----------
    c:          Constant normalization 

    Returns
    ---------
    Constant vector with same length as t. 

    Revisions
    ---------
    2019-02-21  Laura Kreidberg
                laura.kreidberg@gmail.com
    '''
    #DEFINE PARAMETERS
    c = rampparams[0]

    return c*np.ones_like(t) 

