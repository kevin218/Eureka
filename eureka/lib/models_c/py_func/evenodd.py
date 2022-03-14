

def evenodd(params, t, etc = []):
    """
    This function applies a flux offset between even/odd observations

    Parameters
    ----------
    a       : multiplication factor
    t       : array of time/phase points

    Returns
    -------
    returns an array of flux values

    Revisions
    ---------
    Kevin Stevenson       Nov 2012
    """
    a       = params[0]

    y       = np.ones(len(t))
    y[::2] *= a

    return y

