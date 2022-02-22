"""
A library of custom weighted profiles
to fit to the NIRISS orders to complete
the optimal extraction of the data.
"""
import numpy as np
import scipy.optimize as so
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import pyximport
pyximport.install()
from . import niriss_cython

__all__ = ['profile_niriss_median', 'profile_niriss_gaussian',
           'profile_niriss_moffat']

def profile_niriss_median(data, medprof, sigma=50):
    """                         
    Builds a median profile for the NIRISS images.

    Parameters          
    ----------          
    data : object  
    medprof : np.ndarray
       A median image from all NIRISS images. This
       is a first pass attempt, and optimized in  
       the optimal extraction routine. 
    sigma : float, optional 
       Sigma for which to remove outliers above.
       Default is 50.  

    Returns
    -------
    medprof : np.ndarray
       Optimized median profile for optimal extraction.
    """

    for i in range(medprof.shape[1]):

        col = data.median[:,i]+0.0
        x = np.arange(0,len(col),1)

        # fits the spatial profile with a savitsky-golay filter
        # window length needs to be quite small for the NIRISS columns
        filt = savgol_filter(col, window_length=15, polyorder=5)
        resid = np.abs(col-filt)

        # finds outliers 
        inliers = np.where(resid<=sigma*np.nanstd(resid))[0]
        outliers = np.delete(x,inliers)
        
        # removes outliers
        if len(outliers)>0:
            filt = savgol_filter(col[inliers], window_length=3, polyorder=2)

            # finds values that are above/below the interpolation range    
            # these need to be masked first, otherwise it will raise an error
            above = np.where(x[outliers]>x[inliers][-1])[0]
            below = np.where(x[outliers]<x[inliers][0])[0]

            # fills pixels that are above/below the interpolation range 
            # with 0s
            if len(above)>0:
                medprof[:,i][outliers[above]]=0
                outliers = np.delete(outliers, above)
            if len(below)>0:
                medprof[:,i][outliers[below]]=0
                outliers = np.delete(outliers, below)

            # fills outliers with interpolated values
            interp = interp1d(x[inliers], filt)
            if len(outliers)>0:
                medprof[:,i][outliers] = interp(x[outliers])

    return medprof

def profile_niriss_gaussian(data, pos1, pos2):
    """
    Creates a Gaussian spatial profile for NIRISS to complete
    the optimal extraction.

    Parameters
    ----------
    data : np.ndarray
       Image to fit a Gaussian profile to.
    pos1 : np.array
       x-values for the center of the first order.
    pos2 : np.array
       x-values for the center of the second order.

    Returns
    -------
    out_img1 : np.ndarray
       Gaussian profile mask for the first order.
    out_img2 : np.ndarray
       Gaussian profile mask for the second order.
    """
    def residuals(params, data, y1_pos, y2_pos):
        """ Calcualtes residuals for best-fit profile. """
        A, B, sig1 = params
        # Produce the model: 
        model,_ = niriss_cython.build_gaussian_images(data, [A], [B], [sig1], y1_pos, y2_pos)
        # Calculate residuals:
        res = (model[0] - data)
        return res.flatten()

    # fits the mask  
    results = so.least_squares( residuals,
                                x0=np.array([2,3,30]),
                                args=(data, pos1, pos2),
                                bounds=([0.1,0.1,0.1], [100, 100, 30]),
                                xtol=1e-11, ftol=1e-11, max_nfev=1e3
                               )
    # creates the final mask
    out_img1,out_img2,_= niriss_cython.build_gaussian_images(data,
                                                          results.x[0:1],
                                                          results.x[1:2],
                                                          results.x[2:3],
                                                          pos1,
                                                          pos2,
                                                          return_together=False)
    return out_img1[0], out_img2[0]


def profile_niriss_moffat(data, pos1, pos2):
    """
    Creates a Moffat spatial profile for NIRISS to complete
    the optimal extraction.

    Parameters
    ----------
    data : np.ndarray
       Image to fit a Moffat profile to.
    pos1 : np.array  
       x-values for the center of the first order.
    pos2 : np.array  
       x-values for the center of the second order.

    Returns  
    -------  
    out_img1 : np.ndarray
       Moffat profile mask for the first order.
    out_img2 : np.ndarray
       Moffat profile mask for the second order.
    """

    def residuals(params, data, y1_pos, y2_pos):
        """ Calcualtes residuals for best-fit profile. """
        A, alpha, gamma = params
        # Produce the model:
        model = niriss_cython.build_moffat_images(data, [A], [alpha], [gamma], y1_pos, y2_pos)
        # Calculate residuals:
        res = (model[0] - data)
        return res.flatten()

    # fits the mask  
    results = so.least_squares( residuals,
                                x0=np.array([2,3,3]),
                                args=(data, pos1, pos2),
                                xtol=1e-11, ftol=1e-11, max_nfev=1e3
                               )
    # creates the final mask
    out_img1,out_img2 = niriss_cython.build_moffat_images(data,
                                                          results.x[0:1],
                                                          results.x[1:2],
                                                          results.x[2:3],
                                                          pos1,
                                                          pos2,
                                                          return_together=False)
    return out_img1[0], out_img2[0]
