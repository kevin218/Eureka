# NIRISS specific rountines go here
#
# Written by: Adina Feinstein
# Last updated by: Adina Feinstein
# Last updated date: October 1, 2021
#
####################################

import numpy as np
from astropy.io import fits
from skimage.morphology import disk
from skimage import filters, feature
from scipy.ndimage import gaussian_filter

__all__ = ['read_niriss', 'create_niriss_mask']


def read_niriss(filename, data, meta):
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

    # loads all the data into the data object
    data.data = hdu['SCI',1].data
    data.err  = hdu['ERR',1].data
    data.dq   = hdu['DQ' ,1].data

    data.var  = hdu['VAR_POISSON',1].data
    data.v0   = hdu['VAR_RNOISE' ,1].data

    meta = hdu['ASDF_METADATA',1].data

    return data, meta


def create_niriss_mask(imgs, anchors1=None, anchors2=None, plot=False):
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
    plot : bool, optional
       An option to plot the data and intermediate steps to 
       retrieve the mask per each order. Default is False.
    anchors1 : np.array
       (x,y) anchors (in pixel space) for fitting the 1st order. 
       Default is None.
    anchors2 : np.array
       (x,y) anchors (in pixel space) for fitting the 2nd order.
       Default is None.

    Returns
    -------
    img_mask : np.ndarray
       A mask for the 2D images that marks the first and second
       orders for NIRISS observations. The first order is marked
       with value = 1; the second order is marked with value = 2.
    """

    def poly_fit(x,y,deg):
        poly = np.polyfit(x,y,deg=deg)
        return np.poly1d(poly)

    perc = np.nanpercentile(imgs, q=99, axis=0)

    # masks most (not all) cosmic ray outliers
    mask = filters.rank.maximum(perc/np.nanmax(perc),
                                disk(radius=1))
    mask = np.array(mask, dtype=bool)
    
    # applies the mask to the main frame
    data = perc*~mask
    g = gaussian_filter(data, 4)
    g[g>6] = 200
    edges = filters.sobel(g)
    edges[edges>0] = 1
    
    # turns edge array into a boolean array
    edges = (edges-np.nanmax(edges)) * -1
    z = feature.canny(edges)

    x_true = np.arange(0,z.shape[1],1)
    # fits lines to the top and bottom of each order
    y, x = np.indices(z.shape)
    valid_z = z.ravel() == True
    
    # if things go wrong, check here!!
    argsort = np.argsort(x.ravel()[valid_z])
    x_valid = x.ravel()[valid_z][argsort]
    y_valid = y.ravel()[valid_z][argsort]
    z_valid = z.ravel()[valid_z][argsort]

    if anchors1 is None:
        anchors1 = [ [0, 1500, z.shape[1]], [80, 35, 70]]
    if anchors2 is None:
        anchors2 = [ [750, 1500, 1800], [88, 150, 210]]
        
    l1 = poly_fit(anchors1[0], anchors1[1], deg=4)
    l2 = poly_fit(anchors2[0], anchors2[1], deg=4)

    # masks for each order based on the above fitted lines
    # probably want to get rid of hard coded numbers at some point... maybe?
    o1t = y_valid < l1(x_valid)
    o1b = (y_valid > l1(x_valid)) & (y_valid < l1(x_valid)+30)

    o2t = ((y_valid < l2(x_valid)) & (x_valid>750) & 
           (y_valid > l1(x_valid)+29) & (y_valid <220))
    o2b = (y_valid > l2(x_valid)) & (y_valid < l2(x_valid)+30) & (y_valid >90)

    masks = [o1t,o1b,o2t,o2b]
    fits = np.zeros((len(masks),z.shape[1]),dtype=int)
    for i,m in enumerate(masks):
        fit = poly_fit(x_valid[m], y_valid[m], deg=4)
        fits[i] = fit(x_true) + 0.0

    # fills in a mask for the 1st and 2nd NIRISS orders
    # currently preserves the width of each order
    img_mask = np.zeros(z.shape)
    diff2 = np.nanmax(fits[3]-fits[2])
    diff1 = np.nanmax(fits[1]-fits[0])
    
    for i in range(z.shape[1]):
        img_mask[fits[0][i]:fits[0][i]+diff1,i] += 1
        img_mask[fits[2][i]:fits[2][i]+diff2,i] += 2
        
        
    # plots some of the intermediate and final steps
    if plot:
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(14,8))
        ax1.imshow(g)
        ax1.set_title('Gaussian smoothed data')
        ax2.imshow(z)
        ax2.set_title('Canny edge detector')
        ax3.imshow(img_mask, vmin=0, vmax=3)
        ax3.set_title('Final mask')
        plt.show()

    return img_mask
