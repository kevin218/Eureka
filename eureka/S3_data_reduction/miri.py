# MIRI specific rountines go here
import numpy as np
from astropy.io import fits
from eureka.S3_data_reduction import sigrej, optspex
from . import bright2flux as b2f
from . import nircam
import numpy as np
from jwst import datamodels
from gwcs.wcstools import grid_from_bounding_box

# Read FITS file from JWST's NIRCam instrument
def read(filename, data):
    '''
    Reads single FITS file from JWST's MIRI instrument.

    Parameters
    ----------
    filename          : Single filename to read
    data              : data object in which the fits data will stored

    Returns
    -------
    data              : updated data object with the fits data stored inside

    History
    -------
    Written by Kevin Stevenson          November 2012
    Updated for NIRCam (KBS)            May 2021
    Updated docs for MIRI (TJB)         Jun 2021

    '''
    assert isinstance(filename, str)

    hdulist = fits.open(filename)

    # Load main and science headers
    data.mhdr    = hdulist[0].header
    data.shdr    = hdulist['SCI',1].header

    data.intstart    = data.mhdr['INTSTART']
    data.intend      = data.mhdr['INTEND']

    data.data = hdulist['SCI', 1].data
    data.err = hdulist['ERR', 1].data
    data.dq = hdulist['DQ', 1].data
    data.wave = np.tile(wave_MIRI(filename),(72,1))    # hdulist['WAVELENGTH', 1].data
    data.v0 = hdulist['VAR_RNOISE', 1].data
    data.int_times = hdulist['INT_TIMES', 1].data[data.intstart - 1:data.intend]

    # Record integration mid-times in BJD_TDB
    # There is no time information in the simulated MIRI data
    # As a placeholder, I am creating timestamps indentical to the ones in STSci-SimDataJWST/MIRI/Ancillary_files/times.dat.txt
    data.bjdtdb = np.linspace(0,1.73562874e+04, 1680)[data.intstart - 1:data.intend] # data.int_times['int_mid_BJD_TDB']

    # MIRI appears to be rotated by 90° compared to NIRCam, so rotating arrays to allow the re-use of NIRCam code
    if data.shdr['DISPAXIS']==2:
        data.data    = np.swapaxes(data.data, 1, 2)
        data.err     = np.swapaxes(data.err , 1, 2)
        data.dq      = np.swapaxes(data.dq  , 1, 2)
        #data.wave    = np.swapaxes(data.wave, 0, 1)
        data.v0      = np.swapaxes(data.v0  , 1, 2)

    return data


def wave_MIRI(filename):
    tso = datamodels.open(filename)
    x, y = grid_from_bounding_box(tso.meta.wcs.bounding_box)
    ra, dec, lam = tso.meta.wcs(x, y)

    # This array only contains wavelength information for the BB
    lam_x = [np.mean(lam[i]) for i in range(len(lam))]

    # Including nans for out of BB area (eg for reference pixels) so that length agrees with detector/subarray size
    lam_x_full = [np.float64(np.nan)] * int(y[0, 0]) + lam_x + [np.float64(np.nan)] * int(416 - y[-1, 0] - 1)

    return lam_x_full


def unit_convert(data, meta, log):
    '''
    Temporary function template that will later convert from MJy/sr to e-
    '''

    # currently: s3_reduce.py -> inst (like nircam.py) -> bright2flux.py -> bright2dn. But: data.mhdr['PUPIL'] doesn't
    # exist in MIRI. Todo: Make that instrument-specific and make it work for miri. (Not really doable right now because
    # the MIRI stage 2 calints files don't include wavelength information and you need that in order to correct for the
    # wavelength-dependent response function).
    if data.shdr['BUNIT'] == 'MJy/sr':
        # Convert from brightness units (MJy/sr) to flux units (uJy/pix)
        # log.writelog('Converting from brightness to flux units')
        # subdata, suberr, subv0 = b2f.bright2flux(subdata, suberr, subv0, shdr['PIXAR_A2'])
        # Convert from brightness units (MJy/sr) to DNs
        log.writelog('  Converting from brightness units (MJy/sr) to electrons')
        meta.photfile = meta.topdir + meta.ancildir + '/' + data.mhdr['R_PHOTOM'][7:]
        data = b2f.bright2dn(data, meta)
        meta.gainfile = meta.topdir + meta.ancildir + '/' + data.mhdr['R_GAIN'][7:]
        data = b2f.dn2electrons(data, meta)
    return data, meta


def flag_bg(data, meta):
    '''
    Temporary function template that will later flag outliers in sky background along time axis
    '''

    # Code written for NIRCam and untested for MIRI, but likely to still work (as long as MIRI data gets rotated)

    return nircam.flag_bg(data, meta)


def fit_bg(data, mask, y1, y2, bg_deg, p3thresh, n, isplots=False):
    '''
    Temporary function template that will later fit for non-uniform background
    '''

    # Code written for NIRCam and untested for MIRI, but likely to still work (as long as MIRI data gets rotated)

    return nircam.fit_bg(data, mask, y1, y2, bg_deg, p3thresh, n, isplots)
