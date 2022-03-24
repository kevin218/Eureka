import numpy as np

__all__ = ['interpolating_row']

def interpolating_row(data, mask, reg=2, arrtype='data'):
    """
    Fills in masked pixel values with either a median value from
    surrounding pixels along the row, interpolating values,
    or filling with 0.

    Parameters
    ----------
    data : np.ndarray
       Image frame.
    mask : np.ndarray
       Mask of integers, where values greater than 0 are bad
       pixels.
    reg : int, optional
       The number of pixels along the row to interpolate over.
       Default is 2.
    arrtype : str, optional
       Array type. Options are `data` and `var`, where `data`
       results in a median interpolated value to fill in and 
       `var` sets the bad pixels equal to 0. Default is `data`.

    Returns
    -------
    interp : np.ndarray
       Image where the bad pixels are filled in with the 
       appropriate values. Should return the same shape
       as `data`.
    """
    nanx, nany = np.where(mask>0)
    interp = np.zeros(data.shape)

    for loc in np.array([nanx,nany]).T:
        if loc[0] < reg:
            y = np.arange(0,5,1,dtype=int)
        elif loc[0] > 256-reg:
            y = np.arange(loc[0]-reg, 256, 1, dtype=int)
        else:
            y = np.arange(loc[0]-reg, loc[0]+reg,1,dtype=int)

        ind = np.where(y==loc[0])[0]
        y = np.delete(y, ind)
        if arrtype=='data':
            newval = np.nanmedian(data[y,loc[1]])
        else:
            newval = 0.0
        interp[loc[0], loc[1]] = newval + 0.0
    return interp
