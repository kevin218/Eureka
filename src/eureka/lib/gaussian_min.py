from scipy.optimize import minimize
import numpy as np
import sys
from photutils.centroids import (centroid_com, centroid_1dg,  # noqa: F401
                                 centroid_2dg, centroid_sources)


def evalgauss(params, x, y, x0, y0):
    """
    Calculate the values of an unrotated Gauss function 
    given positions in x and y in a mesh grid.

    Parameters
    ----------
    params : list
        List components : amplitude, x_stddev, y_stddev. 
        Acts as the inital guess from meta (params_guess).
    x : ndarray
        The x-coordinates for every pixel within the considered frame.
    y :  ndarray
        The y-coordinates for every pixel within the considered frame.
    x0 : float
        X position guess for centroid.
    y0 : float
        Y position guess for centroid.

    Returns
    -------
    ndarray 
        A 2D array of a 2D gaussian formula.

    Notes
    -----
    History:

    - Feb 22, 2023 Isaac Edelman 
        Initial implementation.
    """
    # unpack params
    amplitude, x_stddev, y_stddev = params

    # Formula for a 2D gaussian function
    return (amplitude*np.exp(-(x-x0)**2/(2*x_stddev**2) -
                             (y-y0)**2/(2*y_stddev**2)))


def minfunc(params, frame, x, y, x_mean, y_mean):
    """
    A cost function that should be minimized 
    when fitting for the Gaussian PSF widths.

    Parameters
    ----------
    params : list
        List components : amplitude, x_stddev, y_stddev. 
        Acts as the inital guess from meta (params_guess).
    frame : 2D ndarray
        Array containing the star image.
    x : ndarray
        The x-coordinates for every pixel within the considered frame.
    y : ndarray
        The y-coordinates for every pixel within the considered frame.
    x_mean : float
        X position guess for centroid.
    y_mean : float
        Y position guess for centroid.

    Returns 
    ------- 
    float 
        The mean-squared error of the Gaussian centroid model.

    Notes
    -----
    History:

    - Feb 22, 2023 Isaac Edelman 
        Initial implementation.
    """
    # Evaluates the guassian using parameters given
    model_gauss = evalgauss(params, x, y, x_mean, y_mean)
    
    # Returns the model - observations squared or "R^2" value
    return np.nanmean((model_gauss-frame)**2)


def pri_cent(img, meta, saved_ref_median_frame):
    """
    Create initial centroid guess based off of median frame of data.
    
    Parameters
    ----------
    img : 2D ndarray
        Array containing the star image.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    saved_ref_median_frame : ndarray
        The stored median frame of the first batch.

    Returns 
    ------- 
    x : float 
        First guess of x centroid position. 
    y : float 
        First guess of y centroid position.
    refrence_median_frame : ndarray
        Median frame of the first batch.

    Notes
    -----
    History:

    - Feb 22, 2023 Isaac Edelman 
        Initial implementation.
    """

    # Create median frame
    if saved_ref_median_frame is None:
        refrence_median_frame = np.ma.median(img, axis=0)
    else:
        refrence_median_frame = saved_ref_median_frame

    # Create initial centroid guess using specified method
    if meta.centroid_tech.lower() in ['com', '1dg', '2dg']:
        cent_func = getattr(sys.modules[__name__],
                            ("centroid_" + meta.centroid_tech.lower()))
        x, y = cent_func(refrence_median_frame)
    else:
        print("Invalid centroid_tech option")

    return x, y, refrence_median_frame


def mingauss(img, yxguess, meta):
    """
    Using an inital centroid guess, 
    get a more precise centroid and PSF-width measurement 
    using only pixels near the inital guess.
    
    Parameters
    ----------
    img : 2D ndarray
        Array containing the star image.
    yxguess : tuple
        A guess at the y and x centroid positions.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    sy : float
        Gaussian width in y direction.
    sx : float
        Gaussian width in x direction.
    x : float
        Refined centroid x position.
    y : float
        Refined centroid y position.

    Notes
    -----
    History:

    - Feb 22, 2023 Isaac Edelman 
        Initial implementation.
    """
    # Create centroid position x,y 
    # based off of centroid method 
    # and inital centroid guess
    if meta.centroid_tech.lower() in ['com', '1dg', '2dg']:
        cent_func = getattr(sys.modules[__name__],
                            ("centroid_" + meta.centroid_tech.lower()))
        x, y = centroid_sources(img, yxguess[1], yxguess[0],
                                centroid_func=cent_func,
                                box_size=(2*meta.ctr_cutout_size+1))
        x, y = x[0], y[0]
    else:
        print("Invalid centroid_tech option")
    
    # Cropping frame to speed up guassian fit
    minx = -int(meta.gauss_frame)+int(x)
    maxx = int(meta.gauss_frame)+int(x)

    # Set Frame size based off of frame crop
    frame = img[:, minx:maxx]

    # Create meshgrid
    x_shape = np.arange(img.shape[1])[minx:maxx]
    y_shape = np.arange(img.shape[0])
    x_mesh, y_mesh = np.meshgrid(x_shape, y_shape)

    # The initial guess for [Gaussian amplitude, xsigma, ysigma]
    if meta.inst == 'miri':
        initial_guess = [400, 5, 5]
    elif meta.inst == 'nircam':
        initial_guess = [400, 41, 41]
    else:
        print(f"Warning: Photometry has only been tested on MIRI and NIRCam"
              f"while meta.inst is set to {meta.inst}")
        initial_guess = [400, 20, 20]

    # Fit the gaussian width by minimizing minfunc with the Nelder-Mead method.
    results = minimize(minfunc, initial_guess,
                       args=(frame, x_mesh, y_mesh, x, y),
                       method='Powell',
                       bounds=[(1e-9, None), (1e-9, None), (1e-9, None)])
    sy = results.x[2]
    sx = results.x[1]

    # Returns guassian widths (sy, sx) and centroid position (x,y)
    return sy, sx, y, x
