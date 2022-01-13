# NIRISS specific rountines go here
#
# Written by: Adina Feinstein
# Last updated by: Adina Feinstein
# Last updated date: January 13, 2022
#
####################################

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.table import Table
from skimage.morphology import disk
from skimage import filters, feature
from scipy.ndimage import gaussian_filter

#from jwst.datamodels import WaveMapModel, WaveMapSingleModel

from .background import fitbg3


__all__ = ['read', 'create_niriss_mask', 'image_filtering',
           'f277_mask', 'fit_bg', 'wave_NIRISS']


def read(filename, f277_filename, data, meta):
    """
    Reads a single FITS file from JWST's NIRISS instrument.
    This takes in the Stage 2 processed files.

    Parameters
    ----------
    filename : str
       Single filename to read. Should be a `.fits` file.
    data : object
       Data object in which the fits data will be stored.

    Returns
    -------
    data : object
       Data object now populated with all of the FITS file
       information.
    meta : astropy.table.Table
       Metadata stored in the FITS file.
    """

    assert(filename, str)

    meta.filename = filename

    hdu = fits.open(filename)
    f277= fits.open(f277_filename)

    # loads in all the header data
    data.mhdr = hdu[0].header
    data.shdr = hdu['SCI',1].header

    data.intend = hdu[0].header['NINTS'] + 0.0
    data.bjdtbd = np.linspace(data.mhdr['EXPSTART'], 
                              data.mhdr['EXPEND'], 
                              int(data.intend))

    # loads all the data into the data object
    data.data = hdu['SCI',1].data + 0.0
    data.err  = hdu['ERR',1].data + 0.0
    data.dq   = hdu['DQ' ,1].data + 0.0

    data.f277 = f277[1].data + 0.0

    data.var  = hdu['VAR_POISSON',1].data + 0.0
    data.v0   = hdu['VAR_RNOISE' ,1].data + 0.0

    meta.meta = hdu[-1].data

    # removes NaNs from the data & error arrays
    data.data[np.isnan(data.data)==True] = 0
    data.err[ np.isnan(data.err) ==True] = 0

    return data, meta

def image_filtering(img, radius=1, gf=4):
    """
    Does some simple image processing to isolate where the
    spectra are located on the detector. This routine is 
    optimized for NIRISS S2 processed data and the F277W filter.

    Parameters
    ----------
    img : np.ndarray
       2D image array. 
    radius : np.float, optional
       Default is 1.
    gf : np.float, optional
       The standard deviation by which to Gaussian
       smooth the image. Default is 4.

    Returns
    -------
    img_mask : np.ndarray
       A mask for the image that isolates where the spectral 
       orders are.
    """
    mask = filters.rank.maximum(img/np.nanmax(img),
                                disk(radius=radius))
    mask = np.array(mask, dtype=bool)

    # applies the mask to the main frame
    data = img*~mask
    g = gaussian_filter(data, gf)
    g[g>6] = 200
    edges = filters.sobel(g)
    edges[edges>0] = 1

    # turns edge array into a boolean array
    edges = (edges-np.nanmax(edges)) * -1
    z = feature.canny(edges)

    return z, g

def f277_mask(img):
    """        
    Marks the overlap region in the f277w filter image.                                                       
    
    Parameters
    ----------
    img : np.ndarray
       2D image of the f277w filter.
    
    Returns
    -------
    mask : np.ndarray
       2D mask for the f277w filter.
    mid : np.ndarray
       (x,y) anchors for where the overlap region is located.
    """
    mask, _ = image_filtering(img[:150,:500])
    mid = np.zeros((mask.shape[1], 2),dtype=int)
    new_mask = np.zeros(img.shape)
    
    for i in range(mask.shape[1]):
        inds = np.where(mask[:,i]==True)[0]
        if len(inds) > 1:
            new_mask[inds[1]:inds[-2], i] = True
            mid[i] = np.array([i, (inds[1]+inds[-2])/2])

    q = ((mid[:,0]<420) & (mid[:,1]>0) & (mid[:,0] > 0))
    return new_mask, mid[q]


def init_mask_guess(data, meta, isplots=True, save=True):
    """
    There are some hard-coded numbers in here right now. The idea
    is that once we know what the real data looks like, nobody will
    have to actually call this function and we'll provide a CSV
    of a good initial guess for each order.

    Parameters  
    ----------  
    data : object
    meta : object
    isplots : bool, optional
       An option to plot the data and intermediate steps to
       retrieve the mask per each order. Default is False.     
    save : bool, optional
       An option to save the polynomial fits to a CSV. Default
       is True. Output table is saved under `niriss_order_guesses.csv`.

    Returns
    -------
    x : np.array
       x-array for the polynomial fits to each order.
    y1 : np.array
       Polynomial fit to the first order.
    y2 : np.array
       Polynomial fit to the second order.
    """
    def rm_outliers(arr):
        # removes instantaneous outliers
        diff = np.diff(arr)
        outliers = np.where(np.abs(diff)>=np.nanmean(diff)+3*np.nanstd(diff))
        arr[outliers] = 0
        return arr
    
    def find_centers(img, cutends):
        """ Finds a running center """
        centers = np.zeros(len(img[0]), dtype=int)
        for i in range(len(img[0])):
            inds = np.where(img[:,i]>0)[0]
            if len(inds)>0:
                centers[i] = np.nanmean(inds)

        centers = rm_outliers(centers)

        if cutends is not None:
            centers[cutends:] = 0

        return centers
    
    def clean_and_fit(x1,x2,y1,y2):
        x1,y1 = x1[y1>0], y1[y1>0]
        x2,y2 = x2[y2>0], y2[y2>0]
        
        poly = np.polyfit(np.append(x1,x2),
                          np.append(y1,y2),
                          deg=4) # hard coded deg of polynomial fit
        fit = np.poly1d(poly)
        return fit


    g = create_niriss_mask(data, meta)
    f,_ = f277_mask(np.nanmax(data.f277,axis=(0,1)))

    g_centers = find_centers(g,cutends=None)
    f_centers = find_centers(f,cutends=430) # hard coded end of the F277 img

    gcenters_1 = np.zeros(len(g[0]),dtype=int)
    gcenters_2 = np.zeros(len(g[0]),dtype=int)

    for i in range(len(g[0])):
        inds = np.where(g[:,i]>100)[0]
        inds_1 = inds[inds <= 78] # hard coded y-boundary for the first order
        inds_2 = inds[inds>=80]   # hard coded y-boundary for the second order
        if len(inds_1)>=1:
            gcenters_1[i] = np.nanmean(inds_1)
        if len(inds_2)>=1:
            gcenters_2[i] = np.nanmean(inds_2)

    gcenters_1 = rm_outliers(gcenters_1)
    gcenters_2 = rm_outliers(gcenters_2)
    x = np.arange(0,len(gcenters_1),1)

    fit1 = clean_and_fit(x, x[x>800],
                         f_centers, gcenters_1[x>800])
    fit2 = clean_and_fit(x, x[(x>800) & (x<1800)],
                         f_centers, gcenters_2[(x>800) & (x<1800)])
    
    if isplots:
        plt.imshow(g+f)
        plt.plot(x, fit1(x), 'k')
        plt.plot(x, fit2(x), 'r')
        plt.show()

    if save:
        tab = Table()
        tab['x'] = x
        tab['order_1'] = fit1(x)
        tab['order_2'] = fit2(x)
        tab.write('niriss_order_guesses.csv',format='csv')

    return x, fit1(x), fit2(x)

def create_niriss_mask(data, meta, isplots=False):
    """
    This routine takes the output S2 processed images and creates
    a mask for each order. This routine creates a single image from
    all 2D images, applies a Gaussian filter to smooth the image, 
    and a Sobel edge detection method to identify the outlines of
    each order. The orders are then fit with 2nd degree polynomials.

    Parameters
    ----------
    data : object
    meta : object
    isplots : bool, optional
       An option to plot the data and intermediate steps to 
       retrieve the mask per each order. Default is False.

    Returns
    -------
    img_mask : np.ndarray
       A mask for the 2D images that marks the first and second
       orders for NIRISS observations. The first order is marked
       with value = 1; the second order is marked with value = 2.
       Overlap regions are marked with value = 3.
    bkg_mask : np.ndarray
       A mask for the 2D images that marks where the background
       is for NIRISS observations. Background regions are given
       value = 1. Regions to ignore are given value = 0.

    """
    def poly_fit(x,y,deg):
        poly = np.polyfit(x,y,deg=deg)
        return np.poly1d(poly)

    perc  = np.nanmax(data.data, axis=0)
    fperc = np.nanmax(data.f277, axis=(0,1))

    # creates data img mask
    z,g = image_filtering(perc)

    # creates mask for f277w image and anchors
    fmask, fmid = f277_mask(fperc)

    # Identify the center of the 1st and 2nd
    # spectral orders
    zmask = np.zeros(z.shape)
    start = 800
    mid1 = np.zeros((z[:,start:].shape[1],2),dtype=int)
    mid2 = np.zeros((z[:,start:].shape[1],2),dtype=int)

    for y in np.arange(start,z.shape[1]-1,1,dtype=int):
        inds = np.where(z[:,y]==True)[0]
        
        if len(inds)>=4:
            zmask[inds[0]:inds[1],y] = True
            zmask[inds[2]:inds[-1],y] = True
            
            mid1[y-start] = np.array([y, (inds[0]+inds[1])/2])
            mid2[y-start] = np.array([y, (inds[2]+inds[-1])/2])
            
        if y > 1900:
            zmask[inds[0]:inds[-1],y] = True
            mid1[y-start] = np.array([y, (inds[0]+inds[-1])/2])

    # Clean 1st order of outliers
    mid1 = mid1[np.argsort(mid1[:,0])]
    tempfit = poly_fit(mid1[:,0], mid1[:,1], 3)
    q1 = ((np.abs(tempfit(mid1[:,0])-mid1[:,1]) <2) &
          (mid1[:,0] > start))
    mid1 = mid1[q1]

    # Clean 2nd order of outliers
    mid2 = mid2[np.argsort(mid2[:,0])]
    tempfit = poly_fit(mid2[:,0], mid2[:,1], 3)
    q2 = (( np.abs(tempfit(mid2[:,0])-mid2[:,1]) <2) &
          (mid2[:,0] > start) )
    mid2 = mid2[q2]

    # Append overlap region to non-overlap regions
    x1, y1 = np.append(fmid[:,0], mid1[:,0]), np.append(fmid[:,1], mid1[:,1])
    x2, y2 = np.append(fmid[:,0], mid2[:,0]), np.append(fmid[:,1], mid2[:,1])

    fit1 = poly_fit(x1,y1,4)
    fit2 = poly_fit(x2,y2,4)

    img_mask = np.zeros(perc.shape)
    bkg_mask = np.ones(perc.shape)

    bkg_width = 30
    for i in range(perc.shape[1]):
        img_mask[int(fit1(i)-meta.order_width):
                     int(fit1(i)+meta.order_width),i] += 1
        bkg_mask[int(fit1(i)-bkg_width):
                     int(fit1(i)+bkg_width),i] = np.nan

        if i < x2[-1]:
            img_mask[int(fit2(i)-meta.order_width):
                         int(fit2(i)+meta.order_width),i] += 2
            bkg_mask[int(fit2(i)-bkg_width):
                         int(fit2(i)+bkg_width),i] = np.nan

                
    # plots some of the intermediate and final steps
    if isplots:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, 
                                                 figsize=(14,10))
        ax1.imshow(g)
        ax1.set_title('Gaussian smoothed data')
        ax2.imshow(z)
        ax2.set_title('Canny edge detector')
        ax3.imshow(img_mask, vmin=0, vmax=3)
        ax3.set_title('Final mask')
        ax4.imshow(bkg_mask, vmin=0, vmax=1)
        ax4.set_title('Background mask')
        plt.show()

    data.order_mask = img_mask
    data.bkg_mask  = bkg_mask

    return img_mask, bkg_mask


def wave_NIRISS(wavefiles, meta):
    """
    Creates the wavelength solution using code from `jwst` and `gwcs`.
    """
    wavemap = WaveMapModel()

    for fn in wavefiles:
        wavemap.map.append(WaveMapSingleModel(init=fn))

    meta.wavelength = wavemap
    return meta


def fit_bg(data, meta):
    """
    Subtracts background from non-spectral regions.

    # want to create some background mask to pass in to 
      background.fitbg2
    """
    bg = fitbg3(data.data, data.order_mask, 
                data.bkg_mask,
                deg=meta.bg_deg, threshold=meta.bg_thresh)

    return bg
