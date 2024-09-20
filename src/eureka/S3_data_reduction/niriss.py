import itertools
import numpy as np
import ccdproc as ccdp
from astropy import units
from astropy.io import fits
import scipy.optimize as so
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.nddata import CCDData
from scipy.signal import find_peaks
from skimage.morphology import disk
from skimage import filters, feature
from scipy.ndimage import gaussian_filter

from .background import fitbg3
from . import niriss_python


__all__ = ['read', 'simplify_niriss_img', 'image_filtering',
           'f277_mask', 'fit_bg', 'wave_NIRISS',
           'mask_method_one', 'mask_method_two', 'fit_orders',
           'fit_orders_fast']


def read(filename, f277_filename, data, meta):
    """
    Reads a single FITS file from JWST's NIRISS instrument.
    This takes in the Stage 2 processed files.

    Parameters
    ----------
    filename : str
        Single filename to read. Should be a `.fits` file.
    f277_filename : str
        A filename of a F277 FITS file.
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object now populated with all of the FITS file
        information.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    """
    meta.filename = filename

    hdu = fits.open(filename)
    f277 = fits.open(f277_filename)

    # loads in all the header data
    data.filename = filename
    data.mhdr = hdu[0].header
    data.shdr = hdu['SCI', 1].header

    data.intend = hdu[0].header['NINTS'] + 0.0
    data.time = np.linspace(data.mhdr['EXPSTART'],
                            data.mhdr['EXPEND'],
                            int(data.intend))
    meta.time_units = 'BJD_TDB'

    meta.photometry = False  # Photometry for NIRISS not implemented yet.

    # loads all the data into the data object
    data.data = hdu['SCI', 1].data
    data.err = hdu['ERR', 1].data
    data.dq = hdu['DQ', 1].data

    data.f277 = f277[1].data

    data.var = hdu['VAR_POISSON', 1].data
    data.v0 = hdu['VAR_RNOISE', 1].data

    meta.meta = hdu[-1].data

    # removes NaNs from the data & error arrays
    data.data[np.isnan(data.data)] = 0
    data.err[np.isnan(data.err)] = 0

    data.median = np.nanmedian(data.data, axis=0)

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
    radius : np.float; optional
       Default is 1.
    gf : np.float; optional
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
    g[g > 6] = 200
    edges = filters.sobel(g)
    edges[edges > 0] = 1

    # turns edge array into a boolean array
    edges = (edges-np.nanmax(edges)) * -1
    z = feature.canny(edges)

    return z, g


def f277_mask(data, isplots=0):
    """
    Marks the overlap region in the f277w filter image.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    isplots : int; optional
       Level of plots that should be created in the S3 stage.
       This is set in the .ecf control files. Default is 0.
       This stage will plot if isplots >= 5.

    Returns
    -------
    mask : np.ndarray
       2D mask for the f277w filter.
    mid : np.ndarray
       (x,y) anchors for where the overlap region is located.
    """
    img = np.nanmax(data.f277, axis=(0, 1))
    mask, _ = image_filtering(img[:150, :500])
    mid = np.zeros((mask.shape[1], 2), dtype=int)
    new_mask = np.zeros(img.shape, dtype=bool)

    for i in range(mask.shape[1]):
        inds = np.where(mask[:, i])[0]
        if len(inds) > 1:
            new_mask[inds[1]:inds[-2], i] = True
            mid[i] = np.array([i, (inds[1]+inds[-2])/2])

    q = ((mid[:, 0] < 420) & (mid[:, 1] > 0) & (mid[:, 0] > 0))

    data.f277_img = new_mask

    if isplots >= 5:
        plt.imshow(new_mask)
        plt.title('F277 Mask')
        plt.show()

    return new_mask, mid[q]


def mask_method_one(data, meta, isplots=0, save=True):
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
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    isplots : int; optional
       Level of plots that should be created in the S3 stage.
       This is set in the .ecf control files. Default is 0.
       This stage will plot if isplots >= 5.
    save : bool; optional
       An option to save the polynomial fits to a CSV. Default
       is True. Output table is saved under `niriss_order_guesses.csv`.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
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
        x1, y1 = x1[y1 > 0], y1[y1 > 0]
        x2, y2 = x2[y2 > 0], y2[y2 > 0]

        poly = np.polyfit(np.append(x1, x2),
                          np.append(y1, y2),
                          deg=4)  # hard coded deg of polynomial fit
        fit = np.poly1d(poly)
        return fit

    g = simplify_niriss_img(data, meta, isplots)

    f, _ = f277_mask(data)

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

    if isplots >= 5:
        plt.figure(figsize=(14, 4))
        plt.title('Order Approximation')
        plt.imshow(g+f)
        plt.plot(x, fit1(x), 'k', label='First Order')
        plt.plot(x, fit2(x), 'r', label='Second Order')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    tab = Table()
    tab['x'] = x
    tab['order_1'] = fit1(x)
    tab['order_2'] = fit2(x)

    if save:
        tab.write('niriss_order_fits_method1.csv', format='csv')

    meta.tab1 = tab

    return meta


def mask_method_two(data, meta, isplots=0, save=False):
    """
    A second method to extract the masks for the first and
    second orders in NIRISS data. This method uses the vertical
    profile of a summed image to identify the borders of each
    order.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    isplots : int; optional
       Level of plots that should be created in the S3 stage.
       This is set in the .ecf control files. Default is 0.
       This stage will plot if isplots >= 5.
    save : bool; optional
       Has the option to save the initial guesses for the location
       of the NIRISS orders. This is set in the .ecf control files.
       Default is False.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    """
    def identify_peaks(column, height, distance):
        p, _ = find_peaks(column, height=height, distance=distance)
        return p

    temp_data = np.ma.masked_where(data.mask, data.data)
    summed = np.ma.sum(temp_data, axis=0)
    ccd = CCDData(summed*units.electron)

    new_ccd_no_premask = ccdp.cosmicray_lacosmic(ccd, readnoise=150,
                                                 sigclip=5, verbose=False)

    summed_f277 = np.nansum(data.f277, axis=(0, 1))

    f277_peaks = np.zeros((summed_f277.shape[1], 2))
    peaks = np.zeros((new_ccd_no_premask.shape[1], 6))
    # hard coded numbers to help set height bounds
    double_peaked = [500, 700, 1850]

    for i in range(summed.shape[1]):
        # Identifies peaks in the F277W filtered image
        fp = identify_peaks(summed_f277[:, i], height=100000, distance=10)
        if len(fp) == 2:
            f277_peaks[i] = fp

        if i < double_peaked[0]:
            height = 2000
        elif i >= double_peaked[0] and i < double_peaked[1]:
            height = 100
        elif i >= double_peaked[1]:
            height = 5000

        p = identify_peaks(new_ccd_no_premask[:, i].data, height=height,
                           distance=10)
        if i < 900:
            p = p[p > 40]  # sometimes catches an upper edge that doesn't exist

        peaks[i][:len(p)] = p

    # Removes 0s from the F277W boundaries
    xf = np.arange(0, summed_f277.shape[1], 1)
    good = f277_peaks[:, 0] != 0
    xf = xf[good]
    f277_peaks = f277_peaks[good]

    # Fitting a polynomial to the boundary of each order
    x = np.arange(0, new_ccd_no_premask.shape[1], 1)
    avg = np.zeros((new_ccd_no_premask.shape[1], 6))

    for ind in range(4):  # CHANGE THIS TO 6 TO ADD THE THIRD ORDER
        q = peaks[:, ind] > 0

        # removes outliers
        diff = np.diff(peaks[:, ind][q])
        good = np.where(np.abs(diff) <= np.nanmedian(diff)+2*np.nanstd(diff))
        good = good[5:-5]
        y = peaks[:, ind][q][good]
        y = y[x[q][good] > xf[-1]]

        # removes some of the F277W points to better fit the 2nd order
        if ind < 2:
            cutoff = -1
        else:
            cutoff = 250

        xtot = np.append(xf[:cutoff], x[q][good][x[q][good] > xf[-1]])
        if ind in [0, 2]:
            ytot = np.append(f277_peaks[:, 0][:cutoff], y)
        else:
            ytot = np.append(f277_peaks[:, 1][:cutoff], y)

        # Fits a 4th degree polynomiall
        poly = np.polyfit(xtot, ytot, deg=4)
        fit = np.poly1d(poly)

        avg[:, ind] = fit(x)

    if isplots >= 5:
        plt.figure(figsize=(14, 4))
        plt.title('Order Approximation')
        plt.imshow(summed, vmin=0, vmax=2e3)
        plt.plot(x, np.nanmedian(avg[:, :2], axis=1), 'k', lw=2,
                 label='First Order')
        plt.plot(x, np.nanmedian(avg[:, 2:4], axis=1), 'r', lw=2,
                 label='Second Order')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()

    tab = Table()
    tab['x'] = x
    tab['order_1'] = np.nanmedian(avg[:, :2], axis=1)
    tab['order_2'] = np.nanmedian(avg[:, 2:4], axis=1)

    if save:
        tab.write('niriss_order_fits_method2.csv', format='csv')

    meta.tab2 = tab

    return meta


def simplify_niriss_img(data, meta, isplots=0):
    """
    Creates an image to map out where the orders are in
    the NIRISS data.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    isplots : int; optional
       Level of plots that should be created in the S3 stage.
       This is set in the .ecf control files. Default is 0.

    Returns
    -------
    g : np.ndarray
       A 2D array that marks where the NIRISS first
       and second orders are.
    """
    perc = np.nanmax(data.data, axis=0)

    # creates data img mask
    z, g = image_filtering(perc)

    if isplots >= 6:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(14, 4),
                                       sharex=True, sharey=True)
        ax1.imshow(z)
        ax1.set_title('Canny Edge')
        ax2.imshow(g)
        ax2.set_title('Gaussian Blurred')
        ax2.set_ylabel('y')
        ax1.set_ylabel('y')
        ax2.set_xlabel('x')
        plt.show()

    data.simple_img = g
    return g


def wave_NIRISS(wavefile, meta):
    """
    Adds the 2D wavelength solutions to the meta object.

    Parameters
    ----------
    wavefile : str
       The name of the .FITS file with the wavelength
       solution.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    """
    with fits.open(wavefile) as hdu:
        meta.wavelength_order1 = hdu[1].data
        meta.wavelength_order2 = hdu[2].data
        meta.wavelength_order3 = hdu[3].data

    return meta


def flag_bg(data, meta):
    '''A placeholder function until a flag_bg function is implemented.

    Outlier rejection of sky background along time axis.

    Parameters
    ----------
    data:   DataClass
        The data object in which the fits data will stored
    meta:   MetaClass
        The metadata object

    Returns
    -------
    data : DataClass
        The updated data object with outlier background pixels flagged.
    '''

    print('WARNING, niriss.flag_bg is not yet implemented!')

    return


def dirty_mask(img, meta, boxsize1=70, boxsize2=60):
    """Really dirty box mask for background purposes."""
    mask = np.zeros(img.shape, dtype=bool)

    for i in range(img.shape[1]):
        s = int(meta.tab2['order_1'][i]-boxsize1/2)
        e = int(meta.tab2['order_1'][i]+boxsize1/2)
        mask[s:e, i] = True

        s = int(meta.tab2['order_2'][i]-boxsize2/2)
        e = int(meta.tab2['order_2'][i]+boxsize2/2)
        try:
            mask[s:e, i] = True
        except:
            # FINDME: Need to change this except to only catch the
            # specific type of exception we expect
            pass

    return mask


def fit_bg(data, meta, readnoise=11, sigclip=[4, 4, 4]):
    """
    Subtracts background from non-spectral regions.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    readnoise : float; optional
       An estimation of the readnoise of the detector.
       Default is 11.
    sigclip : interable; optional
       A list or array corresponding to the
       sigma-level which should be clipped in the cosmic
       ray removal routine. Default is [4, 4, 4].

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object now contains new attribute `bkg_removed`.
    """
    box_mask = dirty_mask(data.median, meta)
    data = fitbg3(data, box_mask, readnoise, sigclip)
    return data


def set_which_table(i, meta):
    """
    A little routine to return which table to
    use for the positions of the orders.

    Parameters
    ----------
    i : int
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns
    -------
    pos1 : np.array
       Array of locations for first order.
    pos2 : np.array
       Array of locations for second order.
    """
    if i == 2:
        pos1, pos2 = meta.tab2['order_1'], meta.tab2['order_2']
    elif i == 1:
        pos1, pos2 = meta.tab1['order_1'], meta.tab1['order_2']
    return pos1, pos2


def fit_orders(data, meta, which_table=2):
    """
    Creates a 2D image optimized to fit the data. Currently
    runs with a Gaussian profile, but will look into other
    more realistic profiles at some point. This routine
    is a bit slow, but fortunately, you only need to run it
    once per observations.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    which_table : int; optional
       Sets with table of initial y-positions for the
       orders to use. Default is 2.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object with two new attributes:
        `order1_mask` and `order2_mask`.
    """
    print("Go grab some food. This routing could take up to 30 minutes.")

    def construct_guesses(A, B, sig, length=10):
        # amplitude of gaussian for first order
        As = np.linspace(A[0], A[1], length)
        # amplitude of gaussian for second order
        Bs = np.linspace(B[0], B[1], length)
        # std of gaussian profile
        sigs = np.linspace(sig[0], sig[1], length)
        # generates all possible combos
        combos = np.array(list(itertools.product(*[As, Bs, sigs])))
        return combos

    pos1, pos2 = set_which_table(which_table, meta)

    # Good initial guesses
    combos = construct_guesses([0.1, 30], [0.1, 30], [1, 40])

    # generates length x length x length number of images and fits to the data
    img1, sigout1 = niriss_python.build_image_models(data.median,
                                                     combos[:, 0],
                                                     combos[:, 1],
                                                     combos[:, 2],
                                                     pos1, pos2)

    # Iterates on a smaller region around the best guess
    best_guess = combos[np.argmin(sigout1)]
    combos = construct_guesses([best_guess[0]-0.5, best_guess[0]+0.5],
                               [best_guess[1]-0.5, best_guess[1]+0.5],
                               [best_guess[2]-0.5, best_guess[2]+0.5])

    # generates length x length x length number of images centered around the
    # previous guess to optimize the image fit
    img2, sigout2 = niriss_python.build_image_models(data.median,
                                                     combos[:, 0],
                                                     combos[:, 1],
                                                     combos[:, 2],
                                                     pos1, pos2)

    # creates a 2D image for the first and second orders with the best-fit
    # gaussian profiles
    final_guess = combos[np.argmin(sigout2)]
    ord1, ord2, _ = niriss_python.build_image_models(data.median,
                                                     [final_guess[0]],
                                                     [final_guess[1]],
                                                     [final_guess[2]],
                                                     pos1, pos2,
                                                     return_together=False)
    meta.order1_mask = ord1[0]
    meta.order2_mask = ord2[0]

    return meta


def fit_orders_fast(data, meta, which_table=2):
    """
    A faster method to fit a 2D mask to the NIRISS data.
    Very similar to `fit_orders`, but works with
    `scipy.optimize.leastsq`.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    which_table : int; optional
       Sets with table of initial y-positions for the
       orders to use. Default is 2.

    Returns
    -------
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    """
    def residuals(params, data, y1_pos, y2_pos):
        """ Calcualtes residuals for best-fit profile. """
        A, B, sig1 = params
        # Produce the model:
        model, _ = niriss_python.build_image_models(data, [A], [B], [sig1],
                                                    y1_pos, y2_pos)
        # Calculate residuals:
        res = (model[0] - data)
        return res.flatten()

    pos1, pos2 = set_which_table(which_table, meta)

    # fits the mask
    results = so.least_squares(residuals,
                               x0=np.array([2, 3, 30]),
                               args=(data.median, pos1, pos2),
                               xtol=1e-11, ftol=1e-11, max_nfev=1e3)

    # creates the final mask
    out_img1, out_img2, _ = niriss_python.build_image_models(
        data.median, results.x[0:1], results.x[1:2], results.x[2:3],
        pos1, pos2, return_together=False)
    meta.order1_mask_fast = out_img1[0]
    meta.order2_mask_fast = out_img2[0]

    return meta
