import numpy as np
import ccdproc as ccdp
from astropy import units
from astropy.table import Table
from astropy.nddata import CCDData
from scipy.signal import find_peaks
from skimage.morphology import disk
from skimage import filters, feature
from scipy.ndimage import gaussian_filter

__all__ = ['image_filtering', 'simplify_niriss_img',
           'mask_method_one', 'mask_method_two', 'f277_mask',
           'ref_file']


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
    data = img*mask
    g = gaussian_filter(data, gf)
    g[g>4] = 10000
    edges = filters.sobel(g)
    edges[edges>0] = 10

    # turns edge array into a boolean array
    edges = (edges-np.nanmax(edges)) * -1
    z = feature.canny(edges)

    return z, g


def simplify_niriss_img(data):
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
    perc  = np.nanmax(data, axis=0)
    # creates data img mask 
    z,g = image_filtering(perc)
    return g


def f277_mask(f277, radius=1, gf=4):
    """
    Marks the overlap region in the f277w filter image.

    Parameters
    ----------
    data : object

    Returns
    -------
    mask : np.ndarray
       2D mask for the f277w filter.
    mid : np.ndarray 
       (x,y) anchors for where the overlap region is located.
    """
    img = np.nanmax(f277, axis=(0,1))
    mask, _ = image_filtering(img[:150,:500], radius, gf)
    mid = np.zeros((mask.shape[1], 2),dtype=int)
    new_mask = np.zeros(img.shape)

    for i in range(mask.shape[1]):
        inds = np.where(mask[:,i]==True)[0]
        if len(inds) > 1:
            new_mask[inds[1]:inds[-2], i] = True
            mid[i] = np.array([i, (inds[1]+inds[-2])/2])

    q = ((mid[:,0]<420) & (mid[:,1]>0) & (mid[:,0] > 0))

    f277_img = new_mask

    return new_mask, mid[q]


def mask_method_one(data, radius=1, gf=4,
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
    data : object
    save : bool, optional
       An option to save the polynomial fits to a CSV. Default
       is True. Output table is saved under `niriss_order_guesses.csv`.

    Returns
    ------- 
    tab : astropy.table.Table
       Table with the x, y center values for the first
       and second orders.
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

    g = simplify_niriss_img(data)
    f,_ = f277_mask(data, radius, gf)

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

    tab = Table()
    tab['x'] = x
    tab['order_1'] = fit1(x)
    tab['order_2'] = fit2(x)

    fn = 'niriss_order_fits_method1.csv'
    if save:
        if outdir is not None:
            path = os.path.join(outdir, fn)
        else:
            path = fn
        tab.write(path, format='csv')

    return tab


def mask_method_two(data, save=False, outdir=None):
    """
    A second method to extract the masks for the first and
    second orders in NIRISS data. This method uses the vertical
    profile of a summed image to identify the borders of each 
    order.

    Parameters
    ----------
    data : object
    save : bool, optional
       Has the option to save the initial guesses for the location
       of the NIRISS orders. This is set in the .ecf control files.
       Default is False.

    Returns
    -------
    tab : astropy.table.Table
       Table with x,y positions for the first and second NIRISS
       orders.
    """

    def identify_peaks(column, height, distance):
        p,_ = find_peaks(column, height=height, distance=distance)
        return p

    
    summed = np.nansum(data.data, axis=0)
    ccd = CCDData(summed*units.electron)

    new_ccd_no_premask = ccdp.cosmicray_lacosmic(ccd, readnoise=150,
                                                 sigclip=5, verbose=False)

    summed_f277 = np.nansum(data.f277, axis=(0,1))

    f277_peaks = np.zeros((summed_f277.shape[1],2))
    peaks = np.zeros((new_ccd_no_premask.shape[1], 6))
    double_peaked = [500, 700, 1850] # hard coded numbers to help set height bounds

    for i in range(summed.shape[1]):

        # Identifies peaks in the F277W filtered image
        fp = identify_peaks(summed_f277[:,i], height=100000, distance=10)
        if len(fp)==2:
            f277_peaks[i] = fp

        if i < double_peaked[0]:
            height=200
        elif i >= double_peaked[0] and i < double_peaked[1]:
            height = 200
        elif i >= double_peaked[1]:
            height = 500

        p = identify_peaks(new_ccd_no_premask[:,i].data, height=height, distance=10)

        if i < 900:
            p = p[p>40] # sometimes catches an upper edge that doesn't exist 

        peaks[i][:len(p)] = p


    # Removes 0s from the F277W boundaries
    xf = np.arange(0,summed_f277.shape[1],1)
    good = f277_peaks[:,0]!=0
    xf=xf[good]
    f277_peaks=f277_peaks[good]

    # Fitting a polynomial to the boundary of each order
    x = np.arange(0,new_ccd_no_premask.shape[1],1)
    avg = np.zeros((new_ccd_no_premask.shape[1], 6))

    for ind in range(4): # CHANGE THIS TO 6 TO ADD THE THIRD ORDER
        q = peaks[:,ind] > 0

        # removes outliers 
        diff = np.diff(peaks[:,ind][q])
        good = np.where(np.abs(diff)<=np.nanmedian(diff)+2*np.nanstd(diff))
        good = good[5:-5]
        y = peaks[:,ind][q][good] + 0
        y = y[x[q][good]>xf[-1]]

        # removes some of the F277W points to better fit the 2nd order
        if ind < 2:
            cutoff=-1
        else:
            cutoff=250

        xtot = np.append(xf[:cutoff], x[q][good][x[q][good]>xf[-1]])
        if ind == 0 or ind == 2:
            ytot = np.append(f277_peaks[:,0][:cutoff], y)
        else:
            ytot = np.append(f277_peaks[:,1][:cutoff], y)

        # Fits a 4th degree polynomial
        poly= np.polyfit(xtot, ytot, deg=4)
        fit = np.poly1d(poly)

        avg[:,ind] = fit(x)

    tab = Table()
    tab['x'] = x
    tab['order_1'] = np.nanmedian(avg[:,:2],axis=1)
    tab['order_2'] = np.nanmedian(avg[:,2:4],axis=1)

    if save:
        fn = 'niriss_order_fits_method2.csv'
        print(outdir)
        if outdir is not None:
            path = os.path.join(outdir, fn)
        else:
            path = fn
        tab.write(path, format='csv')

    return tab


def ref_file(filename):
    """
    Reads in the order traces from the STScI JWST reference
    file.

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
    hdu = fits.open(filename)

    tab = Table()
    tab['x'] = hdu[0].data['X']
    tab['order_1'] = hdu[0].data['Y']
    tab['order_2'] = hdu[1].data['Y']
    tab['order_3'] = hdu[2].data['Y']

    hdu.close()

    return tab
