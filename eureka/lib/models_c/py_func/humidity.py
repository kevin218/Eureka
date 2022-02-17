import numpy as np

def humidity(params, x, etc):
    """
    This function creates a model that fits the change in relative humidity.

    Parameters
    ----------
    rha:    multiplier
    rhb:    offset

    Returns
    -------
    This function returns an array of y values.

    Revisions
    ---------
    2015-01-27	Kevin Stevenson 
	            kbs@uchicago.edu
    """

    rha = params[0]
    rhb = params[1]
    rh  = x[2]
    
    rhmean = rh.mean()
    
    return rha + rhb*(rh-rhmean)
