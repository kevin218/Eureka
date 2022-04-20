# NIRISS specific rountines go here
#
# Written by: Adina Feinstein
# Last updated by: Adina Feinstein
# Last updated date: April 16, 2022
#
####################################
import os
from tqdm import tqdm
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

from ..lib import simultaneous_order_fitting as sof
from ..lib import tracing_niriss as tn

# some cute cython code
import pyximport
pyximport.install()
from . import niriss_cython


__all__ = ['read',
           'flag_bg', 'fit_bg', 'wave_NIRISS',
           'mask_method_edges', 'mask_method_profile']


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
    data.data = hdu['SCI',1].data + 0.0
    data.err  = hdu['ERR',1].data + 0.0
    data.dq   = hdu['DQ' ,1].data + 0.0

    data.var  = hdu['VAR_POISSON',1].data
    data.v0   = hdu['VAR_RNOISE' ,1].data

    meta.meta = hdu[-1].data

    # removes NaNs from the data & error arrays
    data.data[np.isnan(data.data)==True] = 0
    data.err[ np.isnan(data.err) ==True] = 0

    data.median = np.nanmedian(data.data, axis=0)
    hdu.close()

    return data, meta


def mask_method_edges(data, meta=None, radius=1, gf=4,
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

    tab = tn.mask_method_one(data, radius=radius, gf=gf,
                             save=save, outdir=outdir)

    if inclass==False:
        meta.tab1 = tab
        return meta
    else:
        return tab


def mask_method_profile(data, meta=None, isplots=0, save=False, inclass=False,
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
    tab = tn.mask_method_two(data, save=save, outdir=outdir)

    if inclass == False:
        meta.tab2 = tab
        return meta
    else:
        return tab


def wave_NIRISS(filename, orders=None, meta=None, inclass=False):
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
    if orders is None:
        orders = [1,2,3]

    for order in orders:
        for row in tqdm(range(rows)):
            for column in range(columns):
                wavelength_map[order-1, row, column] = assign_wcs_results.meta.wcs(column, 
                                                                                   row, 
                                                                                   order)[-1]
    if inclass == False:
        meta.wavelength_order = wavelength_map
        return meta
    else:
        return wavelength_map


def flag_bg(data, meta, readnoise=11, sigclip=[4,4,4], 
            box=(5,2), filter_size=(2,2), bkg_estimator=['median'], isplots=0):
    """ 
    I think this is just a wrapper for fit_bg, because I perform outlier
    flagging at the same time as the background fitting.
    """
    data = fit_bg(data, meta, readnoise, sigclip, 
                  bkg_estimator=bkg_estimator, box=box, 
                  filter_size=filter_size, isplots=isplots)
    return data



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
    bkg, bkg_var, cr_mask = fitbg3(data, np.array(box_mask-1, dtype=bool), 
                                   readnoise, sigclip, bkg_estimator=bkg_estimator,
                                   box=box, filter_size=filter_size, isplots=isplots)
    data.bkg = bkg
    data.bkg_var = bkg_var
    data.bkg_removed = data.data - data.bkg
    return data


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
