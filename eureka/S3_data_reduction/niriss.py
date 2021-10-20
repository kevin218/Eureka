# NIRISS specific rountines go here
#
# Written by: Adina Feinstein
# Last updated by: Adina Feinstein
# Last updated date: October 12, 2021
#
####################################

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage import filters, feature
from scipy.ndimage import gaussian_filter


__all__ = ['read', 'create_niriss_mask', 'image_filtering',
           'f277_mask']


def read(filename, data, meta):
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

    hdu = fits.open(filename)

    # loads in all the header data
    data.mhdr = hdu[0].header
    data.shdr = hdu['SCI'][1].header

    data.intend = hdu[0].header['NINTS'] + 0.0
    data.bjdtbd = np.linspace(data.mhdr['EXPSTART'], 
                              data.mhdr['EXPEND'], 
                              data.intend)

    # loads all the data into the data object
    data.data = hdu['SCI',1].data
    data.err  = hdu['ERR',1].data
    data.dq   = hdu['DQ' ,1].data

    data.var  = hdu['VAR_POISSON',1].data
    data.v0   = hdu['VAR_RNOISE' ,1].data

    meta = hdu['ASDF_METADATA',1].data

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


def create_niriss_mask(imgs, f277, order_width=14, plot=False):
    """
    This routine takes the output S2 processed images and creates
    a mask for each order. This routine creates a single image from
    all 2D images, applies a Gaussian filter to smooth the image, 
    and a Sobel edge detection method to identify the outlines of
    each order. The orders are then fit with 2nd degree polynomials.

    Parameters
    ----------
    imgs : np.ndarray
       The output `SCI` extension files for NIRISS observations.
    f277 : np.ndarray
       The F277W filtered image. Necessary for identifying the 
       overlap between spectral orders 1 and 2.
    plot : bool, optional
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

    perc  = np.nanmax(imgs, axis=0)
    fperc = np.nanmax(f277, axis=(0,1))

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

    for i in range(perc.shape[1]):
        img_mask[int(fit1(i)-order_width):
                     int(fit1(i)+order_width),i] += 1
        img_mask[int(fit2(i)-order_width):
                     int(fit2(i)+order_width),i] += 2

        # background mask creates orders that are twice the specified 
        # width, to ensure they are not included in the background removal
        bkg_mask[int(fit1(i)-order_width):
                     int(fit1(i)+order_width),i] = np.nan
        bkg_mask[int(fit2(i)-order_width):
                     int(fit2(i)+order_width),i] = np.nan
                
    # plots some of the intermediate and final steps
    if plot:
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

    return img_mask, bkg_mask


def bkg_sub():
    """
    Subtracts background from non-spectral regions.

    # want to create some background mask to pass in to 
      background.fitbg2
    """


