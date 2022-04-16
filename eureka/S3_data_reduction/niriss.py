# NIRISS specific rountines go here
#
# Written by: Adina Feinstein
# Last updated by: Adina Feinstein
# Last updated date: February 7, 2022
#
####################################
import os
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

from jwst import datamodels
from jwst.pipeline import calwebb_spec2
from jwst.pipeline import calwebb_detector1

from .background import fitbg3
from .niriss_extraction import *

# some cute cython code
import pyximport
pyximport.install()
from . import niriss_cython


__all__ = ['read', 'simplify_niriss_img', 'image_filtering',
           'f277_mask', 'flag_bg', 'fit_bg', 'wave_NIRISS',
           'mask_method_one', 'mask_method_two', 'fit_orders',
           'fit_orders_fast', 'box_extract', 'dirty_mask']


def read(filename, data, meta, f277_filename=None):
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
    if f277_filename is not None:
        f277= fits.open(f277_filename)
        data.f277 = f277[1].data + 0.0
        f277.close()

    # loads in all the header data
    data.filename = filename
    data.mhdr = hdu[0].header
    data.shdr = hdu['SCI',1].header

    data.intend = hdu[0].header['NINTS'] + 0.0
    data.time = np.linspace(data.mhdr['EXPSTART'], 
                              data.mhdr['EXPEND'], 
                              int(data.intend))
    meta.time_units = 'BJD_TDB'

    # loads all the data into the data object
    data.data = hdu['SCI',1].data * hdu[0].header['EFFINTTM']
    data.err  = hdu['ERR',1].data + 0.0
    data.dq   = hdu['DQ' ,1].data + 0.0

    data.var  = hdu['VAR_POISSON',1].data * hdu[0].header['EFFINTTM']**2.0
    data.v0   = hdu['VAR_RNOISE' ,1].data * hdu[0].header['EFFINTTM']**2.0

    meta.meta = hdu[-1].data

    # removes NaNs from the data & error arrays
    data.data[np.isnan(data.data)==True] = 0
    data.err[ np.isnan(data.err) ==True] = 0

    data.median = np.nanmedian(data.data, axis=0)
    hdu.close()

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
    data = img*mask
    g = gaussian_filter(data, gf)
    g[g>4] = 10000
    edges = filters.sobel(g)
    edges[edges>0] = 10

    # turns edge array into a boolean array
    edges = (edges-np.nanmax(edges)) * -1
    z = feature.canny(edges)

    return z, g

def f277_mask(data, radius=1, gf=4, isplots=0):
    """        
    Marks the overlap region in the f277w filter image.
    
    Parameters
    ----------
    data : object
    isplots : int, optional
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
    img = np.nanmax(data.f277, axis=(0,1))
    mask, _ = image_filtering(img[:150,:500], radius, gf)
    mid = np.zeros((mask.shape[1], 2),dtype=int)
    new_mask = np.zeros(img.shape)
    
    for i in range(mask.shape[1]):
        inds = np.where(mask[:,i]==True)[0]
        if len(inds) > 1:
            new_mask[inds[1]:inds[-2], i] = True
            mid[i] = np.array([i, (inds[1]+inds[-2])/2])

    q = ((mid[:,0]<420) & (mid[:,1]>0) & (mid[:,0] > 0))

    data.f277_img = new_mask

    if isplots >= 5:
        plt.imshow(new_mask)
        plt.title('F277 Mask')
        plt.show()

    return new_mask, mid[q]


def mask_method_one(data, meta=None, radius=1, gf=4,
                    isplots=0, save=False, inclass=False,
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
    meta : object
    isplots : int, optional
       Level of plots that should be created in the S3 stage.
       This is set in the .ecf control files. Default is 0.
       This stage will plot if isplots >= 5.
    save : bool, optional
       An option to save the polynomial fits to a CSV. Default
       is True. Output table is saved under `niriss_order_guesses.csv`.

    Returns
    -------
    meta : object
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

    g = simplify_niriss_img(data, meta, isplots)
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
    
    if isplots >= 5:
        plt.figure(figsize=(14,4))
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

    fn = 'niriss_order_fits_method1.csv'
    if save:
        if outdir is not None:
            path = os.path.join(outdir, fn)
        else:
            path = fn
        tab.write(path, format='csv')

    if inclass==False:
        meta.tab1 = tab
        return meta
    else:
        return tab


def mask_method_two(data, meta=None, isplots=0, save=False, inclass=False,
                    outdir=None):
    """
    A second method to extract the masks for the first and
    second orders in NIRISS data. This method uses the vertical
    profile of a summed image to identify the borders of each
    order.
    
    ""
    Parameters
    -----------
    data : object
    meta : object
    isplots : int, optional
       Level of plots that should be created in the S3 stage.
       This is set in the .ecf control files. Default is 0.
       This stage will plot if isplots >= 5.
    save : bool, optional
       Has the option to save the initial guesses for the location
       of the NIRISS orders. This is set in the .ecf control files.
       Default is False.

    Returns
    -------
    meta : object
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
    
        if isplots>5:
            plt.plot(np.arange(0,len(summed_f277[:,i]),1), summed_f277[:,i],'k')
            plt.plot(np.arange(0,len(summed_f277[:,i]),1)[fp], summed_f277[:,i][fp],'ro')
            plt.show()

        if i < double_peaked[0]:
            height=200
        elif i >= double_peaked[0] and i < double_peaked[1]:
            height = 200
        elif i >= double_peaked[1]:
            height = 500
            
        p = identify_peaks(new_ccd_no_premask[:,i].data, height=height, distance=10)

        if isplots>5:
            plt.plot(np.arange(0,len(new_ccd_no_premask[:,i].data),1), new_ccd_no_premask[:,i].data,'k')
            plt.plot(np.arange(0,len(new_ccd_no_premask[:,i].data),1)[p], new_ccd_no_premask[:,i].data[p],'go')
            plt.show()

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
        
        # Fits a 4th degree polynomiall
        poly= np.polyfit(xtot, ytot, deg=4)
        fit = np.poly1d(poly)
            
        avg[:,ind] = fit(x)

    if isplots >= 5:
        plt.figure(figsize=(14,4))
        plt.title('Order Approximation')
        plt.imshow(summed, vmin=0, vmax=2e3)
        plt.plot(x, np.nanmedian(avg[:,:2],axis=1), 'k', lw=2,
                 label='First Order')
        plt.plot(x, np.nanmedian(avg[:,2:4],axis=1), 'r', lw=2,
                 label='Second Order')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
    

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

    if inclass == False:
        meta.tab2 = tab
        return meta
    else:
        return tab


def simplify_niriss_img(data, meta, isplots=False):
    """
    Creates an image to map out where the orders are in
    the NIRISS data.

    Parameters     
    ----------     
    data : object  
    meta : object 
    isplots : int, optional
       Level of plots that should be created in the S3 stage.
       This is set in the .ecf control files. Default is 0.  

    Returns
    -------
    g : np.ndarray
       A 2D array that marks where the NIRISS first
       and second orders are.
    """
    perc  = np.nanmax(data.data, axis=0)

    # creates data img mask
    z,g = image_filtering(perc)
    
    if isplots >= 6:
        fig, (ax1,ax2) = plt.subplots(nrows=2,figsize=(14,4),
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


def wave_NIRISS(wavefile, meta=None, inclass=False, filename=None):
    """
    Adds the 2D wavelength solutions to the meta object.
    
    Parameters
    ----------
    wavefile : str
       The name of the .FITS file with the wavelength
       solution.
    meta : object
    filename : str, optional
       The flux filename. Default is None. Needs a filename if
       the `meta` class is not provided.

    Returns
    -------
    meta : object
    """
    if meta is not None:
        rampfitting_results = datamodels.open(meta.filename)
    else:
        rampfitting_results = datamodels.open(filename)

    # Run assignwcs step on Stage 1 outputs:
    assign_wcs_results = calwebb_spec2.assign_wcs_step.AssignWcsStep.call(rampfitting_results)

    # Extract 2D wavelenght map for order 1:
    rows, columns = assign_wcs_results.data[0,:,:].shape
    wavelength_map = np.zeros([3, rows, columns])
    
    # Loops through the three orders to retrieve the wavelength maps
    for order in [1,2,3]:
        for row in tqdm(range(rows)):
            for column in range(columns):
                wavelength_map[order-1, row, column] = assign_wcs_results.meta.wcs(column, 
                                                                                   row, 
                                                                                   order)[-1]
    if inclass == False:
        meta.wavelength_order1 = wavelength_map[0] + 0.0
        meta.wavelength_order2 = wavelength_map[1] + 0.0
        meta.wavelength_order3 = wavelength_map[2] + 0.0
        return meta
    else:
        return wavelength_map[0], wavelength_map[1], wavelength_map[2]


def flag_bg(data, meta, readnoise=11, sigclip=[4,4,4], 
            box=(5,2), filter_size=(2,2), bkg_estimator=['median'], isplots=0):
    """ 
    I think this is just a wrapper for fit_bg, because I perform outlier
    flagging at the same time as the background fitting.
    """
    data, bkg, bkg_var = fit_bg(data, meta, readnoise, sigclip, 
                                bkg_estimator=bkg_estimator, box=box, 
                                filter_size=filter_size, isplots=isplots)
    data.bkg = bkg
    data.bkg_var = bkg_var
    return data


def dirty_mask(img, meta=None, boxsize1=70, boxsize2=60, booltype=True,
               return_together=True, pos1=None, pos2=None):
    """Really dirty box mask for background purposes."""
    order1 = np.zeros((boxsize1, len(img[0])))
    order2 = np.zeros((boxsize2, len(img[0])))
    mask = np.zeros(img.shape)

    if meta is not None:
        pos1 = meta.tab2['order_1'] + 0.0
        pos2 = meta.tab2['order_2'] + 0.0
    if meta is None and pos1 is None:
        return('Cannot create box mask without trace.')

    if booltype==True:
        m1, m2 = -1, -1
    else:
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
        
    if booltype==True:
        mask = ~np.array(mask, dtype=bool)

    if return_together:
        return mask
    else:
        m1, m2 = np.zeros(mask.shape), np.zeros(mask.shape)
        m1[(mask==1) | (mask==3)] = 1
        m2[mask>=2] = 1
        return m1, m2


def box_extract(data, meta, boxsize1=60, boxsize2=50, bkgsub=False):
    """
    Quick & dirty box extraction to use in the optimal extraction routine.
    
    Parameters
    ----------
    data : object
    boxsize1 : int, optional
       Size of the box for the first order. Default is 60 pixels.
    boxsize2 : int, optional
       Size of the box for the second order. Default is 50 pixels.

    Returns
    -------
    spec1 : np.ndarray
       Extracted spectra for the first order.
    spec2 : np.ndarray
       Extracted spectra for the second order.
    """
    spec1 = np.zeros((data.data.shape[0], 
                      data.data.shape[2]))
    spec2 = np.zeros((data.data.shape[0],
                      data.data.shape[2]))
    var1 = np.zeros(spec1.shape)
    var2 = np.zeros(spec2.shape)

    m1,m2 = dirty_mask(data.median, meta,
                       boxsize1, boxsize2,
                       booltype=False, return_together=False)
    
    if bkgsub:
        d=data.bkg_removed+0.0
    else:
        d=data.data+0.0

    for i in range(len(d)):
        spec1[i] = np.nansum(d[i],# * m1, 
                             axis=0)
        spec2[i] = np.nansum(d[i],# * m2
                             axis=0)

        var1[i] = np.nansum(data.var[i] * m1, axis=0)
        var2[i] = np.nansum(data.var[i] * m2, axis=0)

    return spec1, spec2, var1, var2


def fit_bg(data, meta, readnoise=11, sigclip=[4,4,4], box=(5,2), filter_size=(2,2), 
           bkg_estimator=['median'], isplots=0):
    """
    Subtracts background from non-spectral regions.

    Parameters
    ----------
    data : object
    meta : object
    readnoise : float, optional
       An estimation of the readnoise of the detector.
       Default is 5.
    sigclip : list, array, optional
       A list or array of len(n_iiters) corresponding to the
       sigma-level which should be clipped in the cosmic
       ray removal routine. Default is [4,2,3].
    isplots : int, optional
       The level of output plots to display. Default is 0 
       (no plots).

    Returns
    -------
    data : object
    bkg : np.ndarray
    """
    box_mask = dirty_mask(data.median, meta, booltype=True,
                          return_together=True)
    data, bkg, bkg_var = fitbg3(data, np.array(box_mask-1, dtype=bool), 
                                readnoise, sigclip, bkg_estimator=bkg_estimator,
                                box=box, filter_size=filter_size, isplots=isplots)
    return data, bkg, bkg_var


def set_which_table(i, meta):
    """ 
    A little routine to return which table to
    use for the positions of the orders.

    Parameters
    ----------
    i : int
    meta : object

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
    data : object
    meta : object
    which_table : int, optional
       Sets with table of initial y-positions for the
       orders to use. Default is 2.

    Returns
    -------
    meta : object
       Adds two new attributes: `order1_mask` and `order2_mask`.
    """
    print("Go grab some food. This routing could take up to 30 minutes.")

    def construct_guesses(A, B, sig, length=10):
        As   = np.linspace(A[0],   A[1],   length)  # amplitude of gaussian for first order
        Bs   = np.linspace(B[0],   B[1],   length)  # amplitude of gaussian for second order
        sigs = np.linspace(sig[0], sig[1], length)  # std of gaussian profile
        combos = np.array(list(itertools.product(*[As,Bs,sigs]))) # generates all possible combos
        return combos

    pos1, pos2 = set_which_table(which_table, meta)
    
    # Good initial guesses
    combos = construct_guesses([0.1,30], [0.1,30], [1,40])
    
    # generates length x length x length number of images and fits to the data
    img1, sigout1 = niriss_cython.build_gaussian_images(data.median,
                                                        combos[:,0], combos[:,1], 
                                                        combos[:,2], 
                                                        pos1, pos2)

    # Iterates on a smaller region around the best guess
    best_guess = combos[np.argmin(sigout1)]
    combos = construct_guesses( [best_guess[0]-0.5, best_guess[0]+0.5],
                                [best_guess[1]-0.5, best_guess[1]+0.5],
                                [best_guess[2]-0.5, best_guess[2]+0.5] )

    # generates length x length x length number of images centered around the previous
    #   guess to optimize the image fit
    img2, sigout2 = niriss_cython.build_gaussian_images(data.median, 
                                                        combos[:,0], combos[:,1],
                                                        combos[:,2],
                                                        pos1, pos2)

    # creates a 2D image for the first and second orders with the best-fit gaussian
    #    profiles
    final_guess = combos[np.argmin(sigout2)]
    ord1, ord2, _ = niriss_cython.build_gaussian_images(data.median,
                                                        [final_guess[0]],
                                                        [final_guess[1]],
                                                        [final_guess[2]],
                                                        pos1, pos2,
                                                        return_together=False)
    meta.order1_mask = ord1[0]
    meta.order2_mask = ord2[0]

    return meta
    

def fit_orders_fast(data, meta, which_table=2, profile='gaussian'):
    """
    A faster method to fit a 2D mask to the NIRISS data.
    Very similar to `fit_orders`, but works with 
    `scipy.optimize.leastsq`.

    Parameters
    ----------
    data : object
    meta : object
    which_table : int, optional
       Sets with table of initial y-positions for the
       orders to use. Default is 2.

    Returns
    -------
    meta : object
    """
    def residuals(params, data, y1_pos, y2_pos):
        """ Calcualtes residuals for best-fit profile. """
        nonlocal profile

        A, B, sig1 = params
        # Produce the model:   
        if profile.lower() == 'gaussian':
            model,_ = niriss_cython.build_gaussian_images(data, [A], [B], [sig1], y1_pos, y2_pos)
        elif profile.lower() == 'moffat':
            model,_ = niriss_cython.build_moffat_images(data, [A], [B], [sig1], y1_pos, y2_pos)
        # Calculate residuals:     
        res = (model[0] - data)
        return res.flatten()

    pos1, pos2 = set_which_table(which_table, meta)

    # fits the mask
    if profile.lower()=='gaussian':
        x0=[2,3,30]
    elif profile.lower()=='moffat':
        x0=[]
    else:
        print('profile shape not implemented. using gaussian')
        profile='gaussian'
        x0=[2,3,30]

    results = so.least_squares( residuals, 
                                x0=np.array(x0), 
                                args=(data.median, pos1, pos2),
                                xtol=1e-11, ftol=1e-11, max_nfev=1e3
                               )

    # creates the final mask
    if profile.lower() == 'gaussian':
        out_img1,out_img2,_= niriss_cython.build_gaussian_images(data.median, 
                                                                 results.x[0:1], 
                                                                 results.x[1:2], 
                                                                 results.x[2:3], 
                                                                 pos1, 
                                                                 pos2,
                                                                 return_together=False)
    meta.order1_mask_fast = out_img1[0]
    meta.order2_mask_fast = out_img2[0]

    return meta
