import numpy as np


def orthoTrans(params, trans, etc):
    """
    This function uses principal component analysis to modify parameter values.

    Parameters
    ----------
    params:     Array of params to be modified, length is npars for 1D
                If 2D, shape is npars x nsteps
    invtrans:   Inverse transformation matrix, np.matrix() type, shape is npars x npars
    origin:     Array of length npars indicating the reference frame origin
    sigma:      Array of length npars indicating the uncertainties

    Returns
    -------
    This function returns the modified parameter values of shape params

    Revisions
    ---------
    2011-07-22	Kevin Stevenson, UCF  
	            kevin218@knights.ucf.edu
                Original version
    2011-07-27  kevin
                Added sigma, 2D params
    """
    origin, sigma = etc
    
    foo = ((params.T - origin)/sigma).T
    if foo.ndim == 1:
        foo = foo[:, np.newaxis]
    
    return np.squeeze(np.asarray(trans*foo))

