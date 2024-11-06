# NIRISS specific rountines go here
import numpy as np
from astropy.io import fits
import astraeus.xarrayIO as xrio
from . import nircam, sigrej
from ..lib.util import read_time, supersample

# import itertools
# import ccdproc as ccdp
# from astropy import units
# import scipy.optimize as so
# import matplotlib.pyplot as plt
# from astropy.table import Table
# from astropy.nddata import CCDData
# from scipy.signal import find_peaks
# from skimage.morphology import disk
# from skimage import filters, feature
# from scipy.ndimage import gaussian_filter

from .background import fitbg3
from . import niriss_python

# FINDME: update list
__all__ = ['read', 'simplify_niriss_img', 'image_filtering',
           'f277_mask', 'fit_bg', 'wave_NIRISS',
           'mask_method_one', 'mask_method_two', 'fit_orders',
           'fit_orders_fast']

'''
Thoughts as I work through S3_reduce:
    Don't need to run source_pos.source_pos_wrapper(), should call PASTASOSS instead
    How to handle 2D wavelength solution?
    Will need to implement inst.calibrated_spectra()
    inst.flag_ff() should work the same as NIRCam
    Do we want to correct the curvature using straighten.straighten_trace()?

'''

def read(filename, data, meta, log):
    '''Reads single FITS file from JWST's NIRISS instrument.

    Parameters
    ----------
    filename : str
        Single filename to read.
    data : Xarray Dataset
        The Dataset object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with the fits data stored inside.
    meta : eureka.lib.readECF.MetaClass
        The metadata object
    log : logedit.Logedit
        The current log.

    '''
    hdulist = fits.open(filename)

    # Load master and science headers
    data.attrs['filename'] = filename
    data.attrs['mhdr'] = hdulist[0].header
    data.attrs['shdr'] = hdulist['SCI', 1].header
    # try:
    data.attrs['intstart'] = data.attrs['mhdr']['INTSTART']-1
    data.attrs['intend'] = data.attrs['mhdr']['INTEND']
    # except:
    #     # FINDME: Need to only catch the particular exception we expect
    #     print('  WARNING: Manually setting INTSTART to 1 and INTEND to NINTS')
    #     data.attrs['intstart'] = 0
    #     data.attrs['intend'] = data.attrs['mhdr']['NINTS']
    meta.filter = data.attrs['mhdr']['GRATING']

    sci = hdulist['SCI', 1].data
    err = hdulist['ERR', 1].data
    dq = hdulist['DQ', 1].data
    v0 = hdulist['VAR_RNOISE', 1].data
    wave_2d = hdulist['WAVELENGTH', 1].data
    int_times = hdulist['INT_TIMES', 1].data

    # meta.photometry = False  # Photometry for NIRSpec not implemented yet.

    # Increase pixel resolution along cross-dispersion direction
    if meta.expand > 1:
        log.writelog(f'    Super-sampling y axis from {sci.shape[1]} ' +
                     f'to {sci.shape[1]*meta.expand} pixels...',
                     mute=(not meta.verbose))
        sci = supersample(sci, meta.expand, 'flux', axis=1)
        err = supersample(err, meta.expand, 'err', axis=1)
        dq = supersample(dq, meta.expand, 'cal', axis=1)
        v0 = supersample(v0, meta.expand, 'flux', axis=1)
        wave_2d = supersample(wave_2d, meta.expand, 'wave', axis=0)

    # Record integration mid-times in BMJD_TDB
    if meta.time_file is not None:
        time = read_time(meta, data, log)
    elif len(int_times['int_mid_BJD_TDB']) == 0:
        # There is no time information in the simulated NIRSpec data
        print('  WARNING: The timestamps for the simulated NIRSpec data are '
              'currently\n'
              '           hardcoded because they are not in the .fits files '
              'themselves')
        time = np.linspace(data.mhdr['EXPSTART'], data.mhdr['EXPEND'],
                           data.intend)
    else:
        time = int_times['int_mid_BJD_TDB']

    # Record units
    flux_units = data.attrs['shdr']['BUNIT']
    time_units = 'BMJD_TDB'
    wave_units = 'microns'

    if (meta.firstFile and meta.spec_hw == meta.spec_hw_range[0] and
            meta.bg_hw == meta.bg_hw_range[0]):
        # Only apply super-sampling expansion once
        meta.ywindow[0] *= meta.expand
        meta.ywindow[1] *= meta.expand

    data['flux'] = xrio.makeFluxLikeDA(sci, time, flux_units, time_units,
                                       name='flux')
    data['err'] = xrio.makeFluxLikeDA(err, time, flux_units, time_units,
                                      name='err')
    data['dq'] = xrio.makeFluxLikeDA(dq, time, "None", time_units,
                                     name='dq')
    data['v0'] = xrio.makeFluxLikeDA(v0, time, flux_units, time_units,
                                     name='v0')
    data['wave_2d'] = (['y', 'x'], wave_2d)
    data['wave_2d'].attrs['wave_units'] = wave_units

    return data, meta, log


def flag_ff(data, meta, log):
    '''Outlier rejection of full frame along time axis.
    For data with deep transits, there is a risk of masking good transit data.
    Proceed with caution.

    Parameters
    ----------
    data : Xarray Dataset
        The Dataset object.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with outlier pixels flagged.
    '''
    return nircam.flag_ff(data, meta, log)




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
