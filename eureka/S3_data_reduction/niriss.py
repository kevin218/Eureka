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
import astraeus.xarrayIO as xrio
from astropy import units
from astropy.io import fits
from astropy.table import Table
from astropy.nddata import CCDData
from scipy.signal import find_peaks
from skimage.morphology import disk
from skimage import filters, feature
from scipy.ndimage import gaussian_filter

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
    hdulist = fits.open(filename)

    # Load master and science headers
    data.attrs['filename'] = filename
    data.attrs['mhdr'] = hdulist[0].header
    data.attrs['shdr'] = hdulist['SCI', 1].header
    data.attrs['NINTS'] = data.attrs['mhdr']['NINTS']

    # need some placeholder right now for testing
    data.attrs['intstart'] = 0 #data.attrs['mhdr']['INTSTART']
    data.attrs['intend'] = 3 #data.attrs['mhdr']['INTEND']

    if f277_filename is not None:
        f277= fits.open(f277_filename)
        data.attrs['f277'] = f277[1].data + 0.0
        f277.close()

    # Load data
    sci = hdulist['SCI', 1].data
    err = hdulist['ERR', 1].data
    dq = hdulist['DQ', 1].data
    v0 = hdulist['VAR_RNOISE', 1].data
    # var  = hdulist['VAR_POISSON',1].data
    wave_2d = hdulist['WAVELENGTH', 1].data
    # int_times = hdulist['INT_TIMES', 1].data[data.attrs['intstart']-1:
    #                                          data.attrs['intend']]

    # Record integration mid-times in BJD_TDB
    try:
        int_time = hdulist['INT_TIMES', 1].data
        time = int_times['int_mid_BJD_TDB']
        time_units = 'BJD_TDB'
    except:
        # This exception is (hopefully) only for simulated data
        print("WARNING: INT_TIMES not found. Using EXPSTART and EXPEND in UTC.")
        time = np.linspace(data.attrs['mhdr']['EXPSTART'],
                                  data.attrs['mhdr']['EXPEND'],
                                  int(data.attrs['NINTS']))
        time_units = 'UTC'
        # Check that number of SCI integrations matches NINTS from header
        if data.attrs['NINTS'] != sci.shape[0]:
            print("WARNING: Number of SCI integrations doesn't match NINTS from header. Updating NINTS.")
            data.attrs['NINTS'] = sci.shape[0]
            time = time[:data.attrs['NINTS']]

    # Record units
    flux_units = data.attrs['shdr']['BUNIT']
    # wave_units = 'microns'

    # Not sure if these are saved somewhere already?
    meta.time = time
    meta.flux_units = flux_units
    meta.time_units = time_units

    # removes NaNs from the data & error arrays
    sci[np.isnan(sci) == True] = 0
    err[np.isnan(sci) == True] = 0
    # median = np.nanmedian(sci, axis=0)

    data['flux'] = xrio.makeFluxLikeDA(sci, time, flux_units, time_units,
                                       name='flux')
    data['err'] = xrio.makeFluxLikeDA(err, time, flux_units, time_units,
                                      name='err')
    data['dq'] = xrio.makeFluxLikeDA(dq, time, "None", time_units,
                                     name='dq')
    data['v0'] = xrio.makeFluxLikeDA(v0, time, flux_units, time_units,
                                     name='v0')

    hdulist.close()

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
    isplots : int; optional
       Level of plots that should be created in the S3 stage.
       This is set in the .ecf control files. Default is 0.
       This stage will plot if isplots >= 5.
    save : bool; optional
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

    Parameters
    ----------
    data : object
    meta : object
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
    meta : object
    """
    tab = tn.mask_method_two(data, save=save, outdir=outdir)

    if inclass == False:
        meta.tab2 = tab
        return meta
    else:
        return tab


def fit_bg(data, meta, log,
           readnoise=11, sigclip=[4,4,4],
           box=(5,2), filter_size=(2,2),
           bkg_estimator=['median'],
           testing=False, isplots=0):
    """
    Subtracts background from non-spectral regions.

    Parameters
    ----------
    data : object
    meta : object
    readnoise : float, optional
       An estimation of the readnoise of the detector.
       Default is 5.
    sigclip : list, array; optional
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
    box_mask = dirty_mask(data.medflux.values, meta.tab1, booltype=True,
                          return_together=True)
    bkg, bkg_var, cr_mask = fitbg3(data, np.array(box_mask-1, dtype=bool),
                                   readnoise, sigclip, bkg_estimator=bkg_estimator,
                                   box=box, filter_size=filter_size,
                                   testing=testing, isplots=isplots)

    data['bg'] = xrio.makeFluxLikeDA(bkg, meta.time,
                                     meta.flux_units, meta.time_units,
                                     name='bg')
    data['bg_var'] = xrio.makeFluxLikeDA(bkg_var,
                                          meta.time, meta.flux_units,
                                          meta.time_units,
                                          name='bg_var')
    data['bg_removed'] = xrio.makeFluxLikeDA(data.flux - data.bg,
                                              meta.time,
                                              meta.flux_units, meta.time_units,
                                              name='bg_removed')

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
