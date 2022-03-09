"""
A library of custom weighted profiles
to fit to the NIRISS orders to complete
the optimal extraction of the data.
"""
import numpy as np
from tqdm import tqdm
import scipy.optimize as so
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import pyximport
pyximport.install()
from . import niriss_cython

__all__ = ['profile_niriss_median', 'profile_niriss_gaussian',
           'profile_niriss_moffat', 'optimal_extraction']

def profile_niriss_median(medprof, sigma=50):
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

        col = medprof[:,i]+0.0
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


def optimal_extraction(data, spectrum, spectrum_var, sky_bkg,
                       pos1=None, pos2=None, sigma=20, cr_mask=None, Q=18,
                       proftype='gaussian', isplots=0, per_quad=False):
    """
    Optimal extraction routine for NIRISS. This is different from the 
    general `optspex.optimize` since there are two ways to extract the
    NIRISS data. The first is breaking up the image into quadrants. The
    second is extracting the spectra all together.

    Parameters
    ----------
    data : np.ndarray
       Set of raw NIRISS 2D images.
    spectrum : np.ndarray
       Box-extracted spectra for each image to use in the 
       optimal extraction routine.
    spectrum_var : np.ndarray
       Variance array for the box-extracted spectra.
    sky_bkg : np.ndarray
       Images of the estimated sky background.
    pos1 : np.ndarray, optional
       Initial guesses for the center of the first order. These 
       can be taken from `meta.tab1` or `meta.tab2`. 
       Default is None. This is not optional if you are using
       the `gaussian` or `moffat` profile types.
    pos2 : np.ndarray
       Initial guesses for the center of the second order. These
       can be taken from `meta.tab1` or `meta.tab2`.
       Default is None. This is not optional if you are using
       the `gaussian` or `moffat` profile types.
    sigma : float, optional
       Sigma to use when looking for cosmic rays. Default is 20.
    cr_mask : np.ndarray , optional
       A set of masks with cosmic rays already identified. This
       will be used in the very last step, when extracting the
       spectra. Default is None.
    proftype : str, optional
       Sets which type of profile to use when extracting the spectra.
       Default is `gaussian`. Other options include `median` and `moffat`.
    per_quad : bool, optional
       Allows the extraction to happen via quadrants of the image.
       Default is False.
    isplots : int, optional
       A key to decide which diagnostic plots to save. Default is 0 
       (no plots are saved).
    """
    block_extra = np.ones(data[0].shape)

    if per_quad:
        es_all = np.zeros(3, dtype=np.ndarray)
        ev_all = np.zeros(3, dtype=np.ndarray)

        for quad in range(1,4): # CHANGE BACK TO 4
            # Figures out which quadrant location to use
            if quad == 1: # Isolated first order (top right)
                x1,x2 = 1000, data.shape[2]
                y1,y2 = 0, 100
                block_extra[80:y2,x1:x1+250]=0 # there's a bit of 2nd order
                                               # that needs to be masked
                newdata = (data * block_extra)[:,y1:y2, x1:x2]

            elif quad == 2: # Isolated second order (bottom right)
                x1,x2 = 1000, 1900
                y1,y2 = 70, data.shape[1]
                newdata = np.copy(data)[:,y1:y2, x1:x2] # Takes the proper data slice 
    
            elif quad == 3: # Overlap region (left-hand side)
                x1,x2 = 0, 1000
                y1,y2 = 0, data.shape[1]
                newdata = np.copy(data)[:,y1:y2, x1:x2] # Takes the proper data slice

            new_sky_bkg = np.copy(sky_bkg[:, y1:y2, x1:x2]) + 0.0

            if cr_mask is not None:
                new_cr_mask = np.copy(cr_mask[:, y1:y2, x1:x2]) + 0
                print(newdata.shape, new_sky_bkg.shape, new_cr_mask.shape)
            else:
                new_cr_mask = None

            new_spectrum = np.copy(spectrum[:,x1:x2]) + 0.0
            new_spectrum_var = np.copy(spectrum_var[:,x1:x2]) + 0.0

            es, ev = extraction_routine(newdata, 
                                        new_spectrum, 
                                        new_spectrum_var,
                                        new_sky_bkg, 
                                        pos1=pos1[x1:x2],
                                        pos2=pos2[x1:x2],
                                        sigma=sigma,
                                        cr_mask=new_cr_mask,
                                        Q=Q,
                                        proftype=proftype,
                                        isplots=isplots)
            es_all[quad-1] = es + 0.0
            ev_all[quad-1] = ev + 0.0

        return es_all, ev_all

    else: # Full image
        es, ev = extraction_routine(data, spectrum, spectrum_var,
                                    sky_bkg,
                                    pos1=pos1,
                                    pos2=pos2,
                                    sigma=sigma,
                                    cr_mask=cr_mask,
                                    Q=Q,
                                    proftype=proftype,
                                    isplots=isplots)
    
        return es, ev


def extraction_routine(data, spectrum, spectrum_var, sky_bkg,
                       pos1=None, pos2=None, sigma=20, cr_mask=None, Q=18,
                       proftype='gaussian', isplots=0):
    """
    The actual extraction routine. `optimal extraction` is a wrapper
    for this function, since it needs to loop through *if* you want
    to extract the data via quadrants.
    """
    # initialize arrays
    extracted_spectra = np.zeros((len(data), data.shape[2]))
    extracted_error   = np.zeros((len(data), data.shape[2]))

    for i in tqdm(range(len(data))):
    
        ny, nx = data[i].shape
        x = np.arange(0,ny,1)
        
        median = np.nanmedian(data,axis=0)
        median[median<0]=0

        isnewprofile=True
        M = np.ones(data[i].shape) # cosmic ray mask

        while isnewprofile==True:

            # 5. construct spatial profile
            # Median mask creation
            if proftype.lower() == 'median':
                median = profile_niriss_median(median, sigma=5)
                P = (median-sky_bkg[i])*M

            # Gaussian mask creation
            elif proftype.lower() == 'gaussian':
                if (pos1 is not None) and (pos2 is not None):
                    P,_ = profile_niriss_gaussian((median-sky_bkg[i])*M, 
                                                  pos1,
                                                  pos2)
                else:
                    return('Please pass in `pos1` and `pos2` arguments.')

            # Moffat mask creation
            elif proftype.lower() == 'moffat':
                if (pos1 is not None) and (pos2 is not None):
                    P,_ = profile_niriss_moffat((median-sky_bkg[i])*M,
                                                pos1,
                                                pos2)
                else:
                    return('Please pass in `pos1` and `pos2` arguments.')

            else:
                return('Mask profile type not implemeted.')

            P = P/np.nansum(P,axis=0) # profile normalized along columns
            
            isnewprofile=False

            # 6. Revise variance estimates
            V = spectrum_var[i] + np.abs(sky_bkg[i]+(P*spectrum[i]))/Q

            # 7. Mask *more* cosmic ray hits
            stdevs = np.sqrt((np.abs(data[i] - sky_bkg[i] - spectrum[i])*P)**2.0/V)
            yy1,xx1 = np.where((stdevs*M)>sigma)

            if isplots>=9:
                plt.title('check')
                plt.imshow(stdevs*M, vmin=sigma-2, vmax=sigma)
                plt.show()

                plt.title('profile')
                plt.imshow(P)
                plt.show()

            # If cosmic rays found, continue looping through untl 
            # they are all masked
            if len(yy1)>0 or len(xx1)>0:
                M[yy1,xx1] = 0.0
                isnewprofile=True

            # When no more cosmic rays are idetified
            else:
                # 8. Extract optimal spectrum
                denom = np.nansum(M * P**2.0 / V, axis=0)

                if cr_mask is not None:
                    f = np.nansum(M*P*((cr_mask[i])-sky_bkg[i])/V,axis=0) / denom
                else:
                    f = np.nansum(M*P*(data[i]-sky_bkg[i])/V,axis=0) / denom

                var_f = np.nansum(M*P,axis=0) / denom # This may need a sqrt ?

                if isplots>=8:
                    plt.imshow(P)
                    plt.colorbar()
                    plt.show()

        extracted_spectra[i] = f + 0.0
        extracted_error[i] = var_f + 0.0
        
    return extracted_spectra, extracted_error
