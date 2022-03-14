import numpy as np


def orthoInvTrans(params, invtrans, etc):
    """
    This function uses principal component analysis to modify parameter values.

    Parameters
    ----------
    params:     Array of params to be modified
    invtrans:   Inverse transformation matrix, np.matrix() type
    origin:     Array of len(params) indicating the reference frame origin
    sigma:      Array of len(params) indicating the uncertainties

    Returns
    -------
    This function returns the modified parameter values

    Revisions
    ---------
    2011-07-22	Kevin Stevenson, UCF  
	            kevin218@knights.ucf.edu
                Original version
    2011-07-27  kevin
                Added sigma, 2D params
    """
    origin, sigma = etc
    
    if params.ndim == 1:
        params = params[:, np.newaxis]
    
    return (np.squeeze(np.asarray(invtrans*params).T)*sigma+ origin).T

