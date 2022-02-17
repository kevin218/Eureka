import numpy as np

def pcaInvtrans(params, invtrans, origin):
    """
    This function uses principal component analysis to modify parameter values.

    Parameters
    ----------
    params:     Array of parameters to be modified
    invtrans:   Inverse transformation matrix, np.matrix() type
    origin:	    Array of len(params) indicating the reference frame origin

    Returns
    -------
    This function returns the modified parameter values

    Revisions
    ---------
    201-07-22	Kevin Stevenson, UCF  
	            kevin218@knights.ucf.edu
                Original version
    """
    #print(invtrans, params, origin)
    return np.asarray(invtrans*params[:, np.newaxis]).T[0] + origin

