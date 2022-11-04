import os
import numpy as np
import ccdproc as ccdp
from astropy import units
from astropy.table import Table
from astropy.nddata import CCDData
from astropy.io import fits
from scipy.signal import find_peaks
from skimage.morphology import disk
from skimage import filters, feature
from scipy.ndimage import gaussian_filter
from astropy.modeling.models import Moffat1D
import matplotlib.pyplot as plt

__all__ = ['image_filtering', 'simplify_niriss_img',
           'mask_method_edges', 'f277_mask', 'ref_file']


def image_filtering(img, radius=1, gf=4, cutoff=90, isplots=0):
    """
    Does some simple image processing to isolate where the
    spectra are located on the detector.

    This routine is optimized for NIRISS S2 processed data and
    the F277W filter.

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
    z : np.ndarray
       The identified edges of the first two orders.
    g : np.ndarray
       Gaussian filtered image of the orders. Used just for plotting as a check.
    """
    normalized = img/np.nanmax(img)
    normalized[normalized < -1] = np.nanmedian(normalized)
    normalized[70:, 750:1300] = -1

    mask = filters.rank.maximum(normalized,
                                disk(radius=radius))
    mask = np.array(mask, dtype=bool)

    # applies the mask to the main frame
    data = img*mask
    g = gaussian_filter(data, gf)

    # g > 4 to be a good cut-off for what is part of the
    #   the profile, and what is background. 10000 is simply
    #   an easy number to identify later.
    g[g > 90] = 10000
    edges = filters.sobel(g)

    # This defines the edges. Edges will have values
    #   > 0, as set by the filter. 10 is arbitrary and
    #   simply an easy number to identify later.
    edges[edges > 0] = 10

    # turns edge array into a boolean array
    edges = (edges-np.nanmax(edges)) * -1
    edges[70:, 750:1300] = np.nanmedian(edges[70:, 1300:1500])
    edges[:40, :620] = np.nanmedian(edges[70:, 1300:1500])
    z = feature.canny(edges)

    if isplots == 8:
        fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(14,4))
        ax1.imshow(g, aspect='auto', vmin=-1, vmax=70)
        ax1.set_title('gaussian filtered')
        ax2.imshow(edges, aspect='auto', vmin=-1, vmax=10)
        ax2.set_title('edges')
        ax3.imshow(z, aspect='auto', vmin=-1, vmax=1)
        ax3.set_title('canny edges')
        plt.show()

    return z, g


def simplify_niriss_img(data, cutoff, isplots=0):
    """
    Creates an image to map out where the orders are in
    the NIRISS data.

    Parameters
    ----------
    data : np.array
       A 3D array of all frames to calculate the
       maximum frame.

    Returns
    -------
    g : np.ndarray
       A 2D array that marks where the NIRISS first
       and second orders are.
    """
    # creates data img mask
    z, g = image_filtering(data, cutoff=cutoff, isplots=isplots)
    return g


def f277_mask(f277, radius=1, gf=4):
    """
    Marks the overlap region in the f277w filter image.

    Parameters
    ----------
    f277 : np.ndarray
       Frames of the F277W filtered observations.
    radius : float, optional
       The size of the radius to use in the image filtering. Default is 1.
    gf : float, optional
       The size of the Gaussian filter to use in the image filtering. Default is
       4.

    Returns
    -------
    mask : np.ndarray
       2D mask for the f277w filter.
    mid : np.ndarray
       (x,y) anchors for where the overlap region is located.
    """
    mask, _ = image_filtering(f277[:150, :500], radius, gf)
    mid = np.zeros((mask.shape[1], 2), dtype=int)
    new_mask = np.zeros(f277.shape)

    for i in range(mask.shape[1]):
        inds = np.where(mask[:, i])[0]
        if len(inds) > 1:
            new_mask[inds[1]:inds[-2], i] = True
            mid[i] = np.array([i, (inds[1]+inds[-2])/2])

    q = ((mid[:, 0] < 420) & (mid[:, 1] > 0) & (mid[:, 0] > 0))

    return new_mask, mid[q]


def mask_method_edges(data, radius=1, gf=4, cutoff=90,
                      save=False,
                      outdir=None, isplots=0):
    """
    There are some hard-coded numbers in here right now. The idea
    is that once we know what the real data looks like, nobody will
    have to actually call this function and we'll provide a CSV
    of a good initial guess for each order. This method uses some fun
    image processing to identify the boundaries of the orders and fits
    the edges of the first and second orders with a 4th degree polynomial.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object in which the fits data will stored.
    radius : float, optional
       The size of the radius to use in the image filtering. Default is 1.
    gf : float, optional
       The size of the Gaussian filter to use in the image filtering. Default is
       4.
    save : bool, optional
       An option to save the polynomial fits to a CSV. Default
       is True. Output table is saved under `niriss_order_guesses.csv`.
    outdir : str, optional
       The path where to save the output table, if requested. Default is `none`.

    Returns
    -------
    tab : astropy.table.Table
       Table with the x, y center values for the first
       and second orders.
    """

    def rm_outliers(arr):
        # removes instantaneous outliers
        diff = np.diff(arr)
        outliers = np.where(np.abs(diff) >= np.nanmean(diff)+3*np.nanstd(diff))
        arr[outliers] = 0
        return arr

    def find_centers(img, cutends):
        """ Finds a running center """
        centers = np.zeros(len(img[0]), dtype=int)
        for i in range(len(img[0])):
            inds = np.where(img[:, i] > 0)[0]
            if len(inds) > 0:
                centers[i] = np.nanmean(inds)

        centers = rm_outliers(centers)
        if cutends is not None:
            centers[cutends:] = 0

        return centers

    def clean_and_fit(x1, y1):#x2, y1, y2):
        # Cleans up outlier points potentially included when identifying
        #   the center of each trace. Removes those bad points in the
        #   profile fitting.
        x1, y1 = x1[y1 > 0], y1[y1 > 0]
        #x2, y2 = x2[y2 > 0], y2[y2 > 0]

        poly = np.polyfit(x1,#np.append(x1, x2),
                          y1,#np.append(y1, y2),
                          deg=4)  # hard coded deg of polynomial fit
        fit = np.poly1d(poly)
        return fit

    g = simplify_niriss_img(data, cutoff, isplots=isplots)

    # g_centers = find_centers(g, cutends=None)

    gcenters_1 = np.zeros(len(g[0]), dtype=int)

    for i in range(len(g[0])):
        inds = np.where(g[:, i] > 100)[0]
        gcenters_1[i] = np.nanmedian(inds)

    gcenters_1 = rm_outliers(gcenters_1)
    x = np.arange(0, len(gcenters_1), 1)

    fit1 = clean_and_fit(x, gcenters_1)

    tab = Table()
    tab['x'] = x
    tab['order_1'] = fit1(x)

    fn = 'niriss_order_fits_edges.csv'
    if save:
        if outdir is not None:
            path = os.path.join(outdir, fn)
        else:
            path = fn
        tab.write(path, format='csv')

    return tab

def ref_file(filename):
    """Reads in the order traces from the STScI JWST reference file.

    Parameters
    ----------
    filename : str
       Name of the local trace reference file.

    Returns
    -------
    tab : astropy.table.Table
       Table with x,y positions for the first and second NIRISS
       orders.
    """
    with fits.open(filename) as hdu:
        tab = Table()
        tab['x'] = hdu[0].data['X']
        tab['order_1'] = hdu[0].data['Y']
        tab['order_2'] = hdu[1].data['Y']
        tab['order_3'] = hdu[2].data['Y']

    return tab
