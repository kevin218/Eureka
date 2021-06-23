# MIRI specific rountines go here

import numpy as np
from importlib import reload
from astropy.io import fits
from . import bright2flux as b2f
reload(b2f)


# Read FITS file from JWST's NIRCam instrument
def read(filename, data):
    '''
    Reads single FITS file from JWST's NIRCam instrument.

    Parameters
    ----------
    filename          : Single filename to read
    data              : data object in which the fits data will stored
    returnHdr         : Set True to return header files

    Returns
    -------
    data            : Array of data frames
    err             : Array of uncertainty frames
    hdr             : List of header files
    master_hdr      : List of master header files

    History
    -------
    Written by Kevin Stevenson          November 2012
    Updated for NIRCam (KBS)            May 2021

    '''
    assert isinstance(filename, str)

    hdulist = fits.open(filename)

    # Load master and science headers
    data.mhdr    = hdulist[0].header
    data.shdr    = hdulist['SCI',1].header

    data.intstart    = data.mhdr['INTSTART']
    data.intend      = data.mhdr['INTEND']

    data.data    = hdulist['SCI',1].data
    data.err     = hdulist['ERR',1].data
    data.dq      = hdulist['DQ',1].data
    data.wave    = hdulist['WAVELENGTH',1].data
    data.v0      = hdulist['VAR_RNOISE',1].data
    data.int_times = hdulist['INT_TIMES',1].data[data.intstart-1:data.intend]

    return data


def unit_convert(data, meta, log):
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




