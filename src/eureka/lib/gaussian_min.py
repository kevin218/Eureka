from scipy.optimize import minimize
import numpy as np
from photutils.centroids import (centroid_com, centroid_1dg, 
                                 centroid_2dg, centroid_sources)


# minfunc > evaluate guassian model 
# with frame centroid pos + param guess
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
        Data frame map within meta bounds.
    y :  ndarray
        Data frame map within meta bounds.
    x0 : float
        X position guess for centroid.
    y0 : float
        Y position guess for centroid.

    Returns
    -------
    A 2D array of a 2D gaussian formula.

    Notes
    -----
    History:

    - Feb 22, 2023 Isaac Edelman 
        Written and implemented by Isaac Edelman, edelman@baeri.org
    """
    # unpack params
    amplitude, x_stddev, y_stddev = params

    # Formula for a 2D gaussian function
    return (amplitude*np.exp(-(x-x0)**2/(2*x_stddev**2) -
                             (y-y0)**2/(2*y_stddev**2)))


# results > minimize the gaussian model 
# against the frame
def minfunc(params, frame, x, y, x_mean, y_mean):
    """
    Use the mesh grid to create a two dimensional gaussian distribution.

    Parameters
    ----------
    params : list
        List components : amplitude, x_stddev, y_stddev. 
        Acts as the inital guess from meta (params_guess).
    frame : 2D ndarray
        Array containing the star image.
    x : ndarray
        Data frame map within meta bounds.
    y : ndarray
        Data frame map within meta bounds.
    x_mean : float
        X position guess for centroid.
    y_mean : float
        Y position guess for centroid.

    Returns
    -------
    A 2D array of a reduced 2D gaussian array.

    Notes
    -----
    History:

    - Feb 22, 2023 Isaac Edelman 
        Written and implemented by Isaac Edelman, edelman@baeri.org
    """
    # Evaluates the guassian using parameters given
    model_gauss = evalgauss(params, 
                            x, 
                            y, 
                            x_mean, 
                            y_mean)
    
    # Returns the model - observations squared or "R^2" value
    return np.nansum((model_gauss-frame)**2)


# Make median frame 
# then create centroid inital guess with com, 1dg, 2dg
def pri_cent(img, meta):
    """
    Create initial centroid guess based off of median frame of data.
    
    Parameters
    ----------
    img : 2D ndarray
        Array containing the star image.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    x : First guess of x centroid position, float.
    y : Frirst guess of y centroid position, float.

    Notes
    -----
    History:

    - Feb 22, 2023 Isaac Edelman 
        Written and implemented by Isaac Edelman, edelman@baeri.org
    """
    # Create median frame from Integration
    median_Frame = np.ma.median(img, axis=0)

    # Create initial centroid guess using specified method
    if meta.centroid_tech.lower() == 'com': 
        x, y = centroid_com(median_Frame)

    elif meta.centroid_tech.lower() == 'gauss1d':
        x, y = centroid_1dg(median_Frame)

    elif meta.centroid_tech.lower() == 'gauss2d':
        x, y = centroid_2dg(median_Frame)

    # Returns x,y coordinates of centroid position
    return x, y


# Second centroid fit for x,y values 
# + guassian fit for xs, ys
def mingauss(img, yxguess, meta):
    """
    Use an inital centroid guess 
    and a minimized gaussian meshgrid on each frame 
    to create an enchanced centroid position (x, y) 
    and gaussian widths (sx, sy).
    
    Parameters
    ----------
    img : 2D ndarray
        Array containing the star image.
    yxguess : tuple
        A guess at the centroid. 
        Defaults to None which uses the data point 
        with the highest value.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    sy : gaussian width in y direction, float.
    sx : gaussian width in x direction, float.
    x : Refined centroid x position, float.
    y : Refined centroid y position, float.

    Notes
    -----
    History:

    - Feb 22, 2023 Isaac Edelman 
        Written and implemented by Isaac Edelman, 
        edelman@baeri.org
    """
    # Create centroid position x,y 
    # based off of centroid method 
    # and inital centroid guess
    if meta.centroid_tech.lower() == 'com':
        x, y = centroid_sources(img, yxguess[1], 
                                yxguess[0], centroid_func=centroid_com, 
                                box_size=meta.ctr_cutout_size)
    elif meta.centroid_tech.lower() == 'gauss1d':
        x, y = centroid_sources(img, yxguess[1], 
                                yxguess[0], centroid_func=centroid_1dg, 
                                box_size=meta.ctr_cutout_size)
    elif meta.centroid_tech.lower() == 'gauss2d':
        x, y = centroid_sources(img, yxguess[1], 
                                yxguess[0], centroid_func=centroid_1dg, 
                                box_size=meta.ctr_cutout_size)
    
    # Cropping frame to speed up guassian fit
    minx = -int(meta.gauss_frame)+int(x)
    maxx = int(meta.gauss_frame)+int(x)

    # Set Frame size based off of frame crop
    frame = img[:, minx:maxx]

    # Create meshgrid
    x_shape = np.arange(img.shape[1])
    y_shape = np.arange(img.shape[0])
    x_mesh, y_mesh = np.meshgrid(x_shape, y_shape)

    # Fit the gaussian width by minimizing minfunc with Nelder-Mead method. 
    # The inital guess is [400, 41, 41] 
    # or [brightness of pixles in area of interest, xsigma, ysigma].
    results = minimize(minfunc, [400, 41, 41], 
                       args=(frame, x_mesh[:, minx:maxx], 
                             y_mesh[:, minx:maxx], x, y), 
                       method='Nelder-Mead', 
                       bounds=[(1e-9, None), 
                               (1e-9, None), 
                               (1e-9, None)])
    sy = results.x[2]
    sx = results.x[1]

    # Returns guassian widths (sy, sx) and centroid position (x,y)
    return float(sy), float(sx), float(y), float(x)
