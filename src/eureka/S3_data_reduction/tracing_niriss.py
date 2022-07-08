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
           'mask_method_edges', 'mask_method_ears', 'f277_mask',
           'ref_file']


def image_filtering(img, radius=1, gf=4):
    """Does some simple image processing to isolate where the
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
       Gaussian filtered image of the orders. Used for plotting as a check.
    """
    mask = filters.rank.maximum(img/np.nanmax(img),
                                disk(radius=radius))
    mask = np.array(mask, dtype=bool)

    # applies the mask to the main frame
    data = img*mask
    g = gaussian_filter(data, gf)

    # g > 4 to be a good cut-off for what is part of the
    #   the profile, and what is background. 10000 is simply
    #   an easy number to identify later.
    g[g > 4] = 10000
    edges = filters.sobel(g)

    # This defines the edges. Edges will have values
    #   > 0, as set by the filter. 10 is arbitrary and
    #   simply an easy number to identify later.
    edges[edges > 0] = 10

    # turns edge array into a boolean array
    edges = (edges-np.nanmax(edges)) * -1
    z = feature.canny(edges)

    return z, g


def simplify_niriss_img(data):
    """Creates an image to map out where the orders are in
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
    perc = np.nanmax(data, axis=0)
    # creates data img mask
    z, g = image_filtering(perc)
    return g


def f277_mask(f277, radius=1, gf=4):
    """Marks the overlap region in the f277w filter image.

    Parameters
    ----------
    f277 : np.ndarray
       Frames of the F277W filtered observations.
    radius : float, optional
       The size of the radius to use in the image filtering. Default is 1.
    gf : float, optional
       The size of the Gaussian filter to use in the image filtering. Default
       is 4.

    Returns
    -------
    mask : np.ndarray
       2D mask for the f277w filter.
    mid : np.ndarray
       (x,y) anchors for where the overlap region is located.
    """
    img = np.nanmax(f277, axis=(0, 1))
    mask, _ = image_filtering(img[:150, :500], radius, gf)
    mid = np.zeros((mask.shape[1], 2), dtype=int)
    new_mask = np.zeros(img.shape)

    for i in range(mask.shape[1]):
        inds = np.where(mask[:, i])[0]
        if len(inds) > 1:
            new_mask[inds[1]:inds[-2], i] = True
            mid[i] = np.array([i, (inds[1]+inds[-2])/2])

    q = ((mid[:, 0] < 420) & (mid[:, 1] > 0) & (mid[:, 0] > 0))

    return new_mask, mid[q]


def mask_method_edges(data, radius=1, gf=4,
                      save=False,
                      outdir=None):
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
       The size of the Gaussian filter to use in the image filtering. Default
       is 4.
    save : bool, optional
       An option to save the polynomial fits to a CSV. Default
       is True. Output table is saved under `niriss_order_guesses.csv`.
    outdir : str, optional
       The path where to save the output table, if requested. Default is
       None.

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

    def clean_and_fit(x1, x2, y1, y2):
        # Cleans up outlier points potentially included when identifying
        #   the center of each trace. Removes those bad points in the
        #   profile fitting.
        x1, y1 = x1[y1 > 0], y1[y1 > 0]
        x2, y2 = x2[y2 > 0], y2[y2 > 0]

        poly = np.polyfit(np.append(x1, x2),
                          np.append(y1, y2),
                          deg=4)  # hard coded deg of polynomial fit
        fit = np.poly1d(poly)
        return fit

    g = simplify_niriss_img(data.data)
    f, _ = f277_mask(data.f277, radius, gf)

    # g_centers = find_centers(g, cutends=None)
    f_centers = find_centers(f, cutends=430)  # hard coded end of the F277 img

    gcenters_1 = np.zeros(len(g[0]), dtype=int)
    gcenters_2 = np.zeros(len(g[0]), dtype=int)

    for i in range(len(g[0])):
        inds = np.where(g[:, i] > 100)[0]
        inds_1 = inds[inds <= 78]  # hard coded y-boundary for the first order
        inds_2 = inds[inds >= 80]  # hard coded y-boundary for the second order

        if len(inds_1) >= 1:
            gcenters_1[i] = np.nanmean(inds_1)
        if len(inds_2) >= 1:
            gcenters_2[i] = np.nanmean(inds_2)

    gcenters_1 = rm_outliers(gcenters_1)
    gcenters_2 = rm_outliers(gcenters_2)
    x = np.arange(0, len(gcenters_1), 1)

    fit1 = clean_and_fit(x, x[x > 800],
                         f_centers, gcenters_1[x > 800])
    fit2 = clean_and_fit(x, x[(x > 800) & (x < 1800)],
                         f_centers, gcenters_2[(x > 800) & (x < 1800)])

    tab = Table()
    tab['x'] = x
    tab['order_1'] = fit1(x)
    tab['order_2'] = fit2(x)

    fn = 'niriss_order_fits_edges.csv'
    if save:
        if outdir is not None:
            path = os.path.join(outdir, fn)
        else:
            path = fn
        tab.write(path, format='csv')

    return tab


def mask_method_ears(data, degree=4, save=False, outdir=None, isplots=0):
    """A second method to extract the masks for the first and
    second orders in NIRISS data.

    This method uses the vertical profile of a summed image to identify the
    borders of each order.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object in which the fits data will stored.
    degree : int, optional
       The degree of the polynomial to fit to the orders.
       Default is 4.
    save : bool, optional
       Has the option to save the initial guesses for the location
       of the NIRISS orders. This is set in the .ecf control files.
       Default is False.
    outdir : str, optional
       The path where to save the output table, if requested. Default is
       None.
    isplots : int, optional
        Sets how many diagnostic plots to create. Default is 0.

    Returns
    -------
    tab : astropy.table.Table
       Table with x,y positions for the first and second NIRISS
       orders.
    """
    def define_peak_params(column, which_std=1):
        height = np.nanmax(column)  # used to find peak in profile
        std = np.nanstd(column)  # used to find second peak
        return height - which_std*std

    def identify_peaks(column, height, distance):
        """ Identifies peaks in the spatial profile. """
        p, _ = find_peaks(column, height=height, distance=distance)
        return p

    def fit_function(x, y, deg=4):
        """ Fits a n-degree polynomial to x and y data. """
        q = (~np.isnan(x)) & (~np.isnan(y))
        poly = np.polyfit(x[q], y[q], deg=deg)
        fit = np.poly1d(poly)
        return fit

    def find_fit_outliers(x, y, m, deg=4, which_std=2):
        """ Uses difference between data and model to remove outliers. """
        diff = np.abs(y - m)
        outliers = np.where(diff >= (np.nanmedian(diff) +
                                     which_std*np.nanstd(diff)))
        tempx = np.delete(x, outliers)
        tempy = np.delete(y, outliers)
        return tempx, tempy

    def mask_profile(mu, x, y, alpha=3, gamma=13):
        """ Masks profiles that have already been fitted. """
        m1 = Moffat1D(x_0=mu, alpha=alpha, gamma=gamma)
        rmv = np.where(m1(x) < 0.01)[0]  # and points beyond the 1st orders
        newx, newcol = np.copy(x[rmv]), np.copy(y[rmv])
        return newx, newcol

    def diagnostic_plotting(x, y, model, model_final):
        """ Plots the data, the first fit, and the final best-fit. """
        nonlocal summed
        plt.figure(3330)
        plt.clf()
        plt.imshow(summed, vmin=0, vmax=np.nanpercentile(summed, 75))
        plt.plot(x, y, 'k.', label='Data')
        plt.plot(x, model(x), 'darkorange', label='First Fit Attempt')
        plt.plot(x, model_final(x), 'deepskyblue', lw=2, label='Final Fit')
        plt.legend(ncol=3)
        plt.show()

    summed = np.copy(data.median)  # np.nansum(data.median, axis=0)
    ccd = CCDData(summed*units.electron)

    new_ccd_no_premask = ccdp.cosmicray_lacosmic(ccd, readnoise=150,
                                                 sigclip=4, verbose=False)

    x = np.arange(0, new_ccd_no_premask.data.shape[1], 1)

    # Initializes astropy.table.Table to save traces to
    tab = Table()
    tab['x'] = x

    # Extraction for the first order
    center_1 = np.zeros(new_ccd_no_premask.data.shape[1])
    for i in range(len(center_1)):
        height = define_peak_params(new_ccd_no_premask.data[:, i])
        p = identify_peaks(new_ccd_no_premask.data[:, i],
                           height=height,
                           distance=10.0)
        center_1[i] = np.nanmedian(x[p])  # Takes the median between peaks
    # Iterate on fitting a profile to remove outliers from the first go
    fit1 = fit_function(x, center_1, deg=degree)
    x1, y1 = find_fit_outliers(x, center_1, fit1(x))  # Finds bad points
    fit1_final = fit_function(x1, y1, deg=degree)

    tab['order_1'] = fit1_final(x)  # Adds fit of 1st order to output table

    if new_ccd_no_premask.shape[0] == 256:
        # Checks to see if 2nd & 3rd orders available

        # Some arrays we'll be populating later on
        colx = np.arange(0, new_ccd_no_premask.data.shape[0], 1)
        center_2 = np.zeros(new_ccd_no_premask.data.shape[1])
        center_3 = np.zeros(new_ccd_no_premask.data.shape[1])

        # We almost certainly want to fit the 3rd order first,
        #    since it's physically distinct

        for i in range(5, new_ccd_no_premask.shape[1]):
            col = new_ccd_no_premask.data[:, i]
            newx, newcol = mask_profile(mu=tab['order_1'][i], x=colx, y=col)

            if i <= 750:
                # Can't get a good guesstimate for 3rd order past pixel~750
                height = define_peak_params(newcol, which_std=4)
                p = identify_peaks(newcol, height=height, distance=10.0)
                # want to make sure we get the 3rd order
                inds = np.where(newx[p] > 120)[0]

                if not np.isnan(np.nanmedian(newx[p[inds]])):
                    center_3[i] = np.nanmedian(newx[p[inds]])
                    newx, newcol = mask_profile(mu=center_3[i], x=newx,
                                                y=newcol)  # masks 3rd order
            else:
                newx, newcol = mask_profile(mu=center_1[i], x=newx,
                                            y=newcol)  # masks 1st order

            if (i >= 500) and (i <= 1850):
                # Can't get a good guesstimate for 2nd order past pixel~500
                height = define_peak_params(newcol, which_std=2)
                p = identify_peaks(newcol, height=height, distance=10.0)
                center_2[i] = np.nanmedian(newx[p])

        # Fitting polynomial to 3rd order
        q3 = ((center_3 > 0) & (~np.isnan(center_3)))
        fit3 = fit_function(x[q3], center_3[q3], deg=degree)
        # Finds bad points
        x3, y3 = find_fit_outliers(x[q3], center_3[q3], fit3(x[q3]))
        fit3_final = fit_function(x3, y3, deg=degree)
        tab['order_3'] = fit3_final(x)  # Adds fit of 3rd order to output table
        # Remove parts of the fit where no 3rd order
        tab['order_3'][1000:len(tab['order_3'])] = np.nan

        # Fitting polynomial to 2nd order
        # removes first 500 and last 268 points
        rmv_nans = ((~np.isnan(center_2)) &
                    (center_2 > 0) & (x < 1760))
        fit2 = fit_function(x[rmv_nans], center_2[rmv_nans], deg=degree)
        x2, y2 = find_fit_outliers(x[rmv_nans],
                                   center_2[rmv_nans], fit2(x[rmv_nans]),
                                   which_std=1)
        # Need some points from the first order to anchor second order
        x2 = np.append(x[:100], x2)
        y2 = np.append(tab['order_1'][:100]+15, y2)

        fit2_final = fit_function(x2, y2, deg=degree)

        if isplots >= 6:
            diagnostic_plotting(x, center_1, fit1, fit1_final)
            diagnostic_plotting(x, center_2, fit2, fit2_final)
            diagnostic_plotting(x, center_3, fit3, fit3_final)

        # Add fit of 2nd order to output table
        tab['order_2'] = fit2_final(x)

    if save:
        fn = 'niriss_order_fits_ears.csv'
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
