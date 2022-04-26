"""
A library of custom weighted profiles
to fit to the NIRISS orders to complete
the optimal extraction of the data.

Written by: Adina Feinstein
Last updated: March 23, 2022
"""
import numpy as np
from tqdm import tqdm
import scipy.optimize as so
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

import pyximport
pyximport.install()
from . import niriss_cython as profiles


__all__ = ['box_extract', 'dirty_mask',
           'profile_niriss_median', 'profile_niriss_gaussian',
           'profile_niriss_moffat', 'optimal_extraction_routine',
           'extraction_routine']


def box_extract(data, var, boxmask):
    """
    Quick & dirty box extraction to use in the optimal extraction routine.

    Parameters
    ----------
    data : np.ndarray
       Array of science frames.
    var : np.ndarray
       Array of variance frames.
    boxmask : np.ndarray
       Array of masks for each individual order.

    Returns
    -------
    spec1 : np.ndarray
       Extracted spectra for the first order.
    spec2 : np.ndarray
       Extracted spectra for the second order.
    """

    spec1 = np.zeros((data.shape[0],
                      data.shape[2]))
    spec2 = np.zeros((data.shape[0],
                      data.shape[2]))
    var1 = np.zeros(spec1.shape)
    var2 = np.zeros(spec2.shape)

    m1, m2 = boxmask

    summed = (m1+0)+(m2*2)
    masked_second = summed+0.0
    masked_second[summed==2] = 0
    masked_second[masked_second>0] = 1

    masked_first = summed+0.0
    masked_first[summed==1] = 0
    masked_first[masked_first>0] = 1

    for i in range(len(data)):
        spec1[i] = np.nansum(data[i]*masked_second,
                             axis=0)
        spec2[i] = np.nansum(data[i]*masked_first,
                             axis=0)

        var1[i] = np.nansum(var[i] * m1, axis=0)
        var2[i] = np.nansum(var[i] * m2, axis=0)

    return spec1, spec2, var1, var2


def dirty_mask(img, tab=None, boxsize1=70, boxsize2=60, booltype=True,
               return_together=True, pos1=None, pos2=None):
    """
    Really dirty box mask for background purposes.

    Parameters
    ----------
    img : np.ndarray
       Science image.
    tab : astropy.table.Table
       Table containing the location of the traces.
    boxsize1 : int, optional
       Box size for the first order. Default is 70.
    boxsize2 : int, optional
       Box size for the second order. Default is 60.
    booltype : bool, optional
       Sets the dtype of the mask array. Default is
       True (returns array of boolean values).
    return_together : bool, optional
       Determines whether or not to return one combined
       profile mask or masks for both orders separately.
       Default is True.

    Return
    ------

    """
    order1 = np.zeros((boxsize1, len(img[0])))
    order2 = np.zeros((boxsize2, len(img[0])))
    mask = np.zeros(img.shape)

    if tab is not None:
        pos1 = tab['order_1'] + 0.0
        pos2 = tab['order_2'] + 0.0

    m1, m2 = 1, 2

    for i in range(img.shape[1]):
        s,e = int(pos1[i]-boxsize1/2), int(pos1[i]+boxsize1/2)
        order1[:,i] = img[s:e,i]
        mask[s:e,i] += m1

        s,e = int(pos2[i]-boxsize2/2), int(pos2[i]+boxsize2/2)
        try:
            order2[:,i] = img[s:e,i]
            mask[s:e,i] += m2
        except:
            pass

    if return_together:
        if booltype==True:
            mask = ~np.array(mask,dtype=bool)
        return mask
    else:
        m1, m2 = np.zeros(mask.shape), np.zeros(mask.shape)
        m1[(mask==1) | (mask==3)] = 1
        m2[mask>=2] = 1

        if booltype==True:
            return np.array(m1,dtype=bool), np.array(m2,dtype=bool)
        else:
            return m1, m2


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
        inliers = np.where(resid<=sigma)[0]
        outliers = np.delete(x,inliers)

        # removes outliers
        if len(outliers)>0:
            filt = savgol_filter(col[inliers], window_length=7, polyorder=2)

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

        #if i > 300 and i < 350:
        #    plt.plot(x, medprof[:,i], 'r')
        #    plt.show()

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
        model,_ = profiles.build_gaussian_images(data, [A], [B], [sig1], y1_pos, y2_pos)
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
    out_img1,out_img2,_= profiles.build_gaussian_images(data,
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
        model = profiles.build_moffat_images(data, [A], [alpha], [gamma], y1_pos, y2_pos)
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
    out_img1,out_img2 = profiles.build_moffat_images(data,
                                                     results.x[0:1],
                                                     results.x[1:2],
                                                     results.x[2:3],
                                                     pos1,
                                                     pos2,
                                                     return_together=False)
    return out_img1[0], out_img2[0]


def optimal_extraction_routine(data, var, spectrum, spectrum_var, sky_bkg, medframe=None,
                               pos1=None, pos2=None, sigma=20, cr_mask=None, Q=18,
                               proftype='gaussian', isplots=0, per_quad=False, test=False):
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
       can be taken from `tab1` or `tab2`.
       Default is None. This is not optional if you are using
       the `gaussian` or `moffat` profile types.
    pos2 : np.ndarray
       Initial guesses for the center of the second order. These
       can be taken from `tab1` or `tab2`.
       Default is None. This is not optional if you are using
       the `gaussian` or `moffat` profile types.
    sigma : float, optional
       Sigma to use when looking for cosmic rays. Default is 20.
    Q : float, optional
       Gain value.
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

    # Create a box mask to set pixels far from the order to 0
    boxmask = dirty_mask(medframe, pos1=pos1, pos2=pos2)
    boxmask = np.array(~boxmask, dtype=int)

    # Loops over each quadrant
    if per_quad:
        es_all = np.zeros(3, dtype=np.ndarray)
        ev_all = np.zeros(3, dtype=np.ndarray)
        p_all = np.zeros(3, dtype=np.ndarray)

        for quad in range(1,4): # CHANGE BACK TO 4
            # Figures out which quadrant location to use
            if quad == 1: # Isolated first order (top right)
                x1,x2 = 1000, data.shape[2]
                y1,y2 = 0, 90
                block_extra[80:y2,x1:x1+250]=0 # there's a bit of 2nd order
                                               # that needs to be masked
                newdata = (data * block_extra)[:,y1:y2, x1:x2]

                index = 0 # Index for the box extracted spectra

            elif quad == 2: # Isolated second order (bottom right)
                x1,x2 = 1000, 1900
                y1,y2 = 70, data.shape[1]
                newdata = np.copy(data)[:,y1:y2, x1:x2] # Takes the proper data slice

                index = 1 # Index for the box extracted spectra

            elif quad == 3: # Overlap region (left-hand side)
                x1,x2 = 0, 1000
                y1,y2 = 0, data.shape[1]
                print(x1,x2, y1,y2)
                newdata = np.copy(data)[:,y1:y2, x1:x2] # Takes the proper data slice

                index = 0 # Index for the box extracted spectra

            new_sky_bkg = np.copy(sky_bkg[:, y1:y2, x1:x2]) + 0.0

            if cr_mask is not None:
                new_cr_mask = np.copy(cr_mask[:, y1:y2, x1:x2]) + 0
            else:
                new_cr_mask = None

            # Clip 1D arrays to the length of the quadrant
            new_spectrum = np.copy(np.array(spectrum)[index,:,x1:x2]) + 0.0
            newvar = np.copy(var[:,y1:y2,x1:x2]) + 0.0
            new_spectrum_var = np.copy(np.array(spectrum_var)[index,:,x1:x2]) + 0.0

            # Run the optimal extraction routine on the quadrant
            es, ev, p = extraction_routine(newdata*boxmask[y1:y2, x1:x2],
                                           newvar,
                                           new_spectrum,
                                           new_spectrum_var,
                                           new_sky_bkg,
                                           medframe=medframe[y1:y2, x1:x2],
                                           pos1=pos1[x1:x2],
                                           pos2=pos2[x1:x2],
                                           sigma=sigma,
                                           cr_mask=new_cr_mask,
                                           Q=Q,
                                           proftype=proftype,
                                           isplots=isplots,
                                           test=test)
            es_all[quad-1] = es + 0.0
            ev_all[quad-1] = ev + 0.0
            p_all[quad-1] = p + 0.0

        return es_all, ev_all, p_all

    else: # Full image
        es, ev, p = extraction_routine(data, var,
                                       spectrum,
                                       spectrum_var,
                                       sky_bkg,
                                       medframe=medframe,
                                       pos1=pos1,
                                       pos2=pos2,
                                       sigma=sigma,
                                       cr_mask=cr_mask,
                                       Q=Q,
                                       proftype=proftype,
                                       isplots=isplots,
                                       test=test)

        return es, ev, p


def extraction_routine(data, var,
                       spectrum, spectrum_var, sky_bkg, medframe=None,
                       pos1=None, pos2=None, sigma=20, cr_mask=None, Q=18,
                       proftype='gaussian', isplots=0, test=False):
    """
    The actual extraction routine. `optimal extraction` is a wrapper
    for this function, since it needs to loop through *if* you want
    to extract the data via quadrants.

    Parameters
    ----------
    data : np.array
       Raw data frames.
    var : np.array
       Variance frames.
    spectrum : np.array
       Box-extracted spectrum.
    spectrum_var : np.array
       Variance on box-extracted spectrum.
    sky_bkg : np.array
       Modeled sky background frames.
    """
    # initialize arrays
    extracted_spectra = np.zeros((len(data), data.shape[2]))
    extracted_error   = np.zeros((len(data), data.shape[2]))
    best_fit_prof     = np.zeros(data.shape)

    if test:
        frames = 5
    else:
        frames = len(data)

    # Loop through each frame
    for i in tqdm(range(frames)):

        ny, nx = data[i].shape
        x = np.arange(0,ny,1)

        median = np.nanmedian(data,axis=0)
        median[median<0]=0

        isnewprofile=True
        M = np.ones(data[i].shape) # cosmic ray mask
        fill_vals = np.zeros(data[i].shape) # array with values to fill in masked pixels

        reference = spectrum[i] + 0.0

        while isnewprofile==True:

            # 5. construct spatial profile
            # Median mask creation
            if proftype.lower() == 'median':
                P = medframe*M + 0.0

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
            P = np.abs(P)

            # 6. Revise variance estimates
            V = spectrum_var[i] + np.abs(sky_bkg[i]+(P*reference))/Q
            #V = var[i] + np.abs(sky_bkg[i]+(P*reference))/Q

            # 7. Mask *more* cosmic ray hits
            stdevs = (np.abs(data[i] - sky_bkg[i] - reference)*P)/np.sqrt(V)

            # this needs to be the *worst* pixel in a *single column*
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

                isnewprofile = True

                for u in np.unique(xx1):
                    r = np.where(xx1==u)[0]
                    o = np.argmax(data[0,yy1[r], xx1[r]])
                    M[yy1[r][o], xx1[r][0]] *= 0.0 # Set bad pixels to 0

                    fv = np.nanmedian(data[i,yy1[r][o-1:o+2],
                                           xx1[r][0]] * M[yy1[r][o-1:o+2],
                                           xx1[r][0]])
                    fill_vals[yy1[r][o], xx1[r][0]] += fv

                    if isplots==14:
                        l = len(data[i][:,xx1[r][0]])
                        plt.plot(np.arange(0,l,1), P[:,xx1[r][0]], 'red')
                        plt.title('{}, {}'.format(yy1[r][o],xx1[r][0]))
                        plt.show()

                denom = np.nansum(M * P**2.0 / V, axis=0)

                # Remake spectrum[i] reference
                reference = np.nansum(M*P*( (data[i]+fill_vals) - sky_bkg[i])/V, axis=0) / denom

            # When no more cosmic rays are idetified
            else:
                isnewprofile = False
                # 8. Extract optimal spectrum
                denom = np.nansum(M * P**2.0 / V, axis=0)

                if cr_mask is not None:
                    f = np.nansum(M*P*( (cr_mask[i]+fill_vals) - sky_bkg[i])/V, axis=0) / denom
                    var_f = np.nansum( (M*P)*cr_mask[i], axis=0) / denom # This may need a sqrt ?
                else:
                    f = np.nansum(M*P*( (data[i]+fill_vals) - sky_bkg[i])/V, axis=0) / denom
                    var_f = np.nansum(M*P,axis=0) / denom # This may need a sqrt ?

                if isplots>=8:
                    plt.imshow(P)
                    plt.colorbar()
                    plt.show()

        extracted_spectra[i] = f + 0.0
        extracted_error[i] = var_f + 0.0
        best_fit_prof[i] = P + 0.0

    return extracted_spectra, extracted_error, best_fit_prof
