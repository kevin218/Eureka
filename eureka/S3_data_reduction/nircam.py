
# NIRCam specific rountines go here
import numpy as np
from importlib import reload
from astropy.io import fits
from eureka.S3_data_reduction import sigrej
from eureka.S3_data_reduction import background as bg
from eureka.S3_data_reduction import bright2flux as b2f
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


def flag_bg(data, meta):
    '''
    Outlier rejection of sky background along time axis
    '''

    y1, y2, bg_thresh = meta.bg_y1, meta.bg_y2, meta.bg_thresh

    bgdata1 = data.subdata[:,  :y1]
    bgmask1 = data.submask[:,  :y1]
    bgdata2 = data.subdata[:,y2:  ]
    bgmask2 = data.submask[:,y2:  ]
    bgerr1  = np.median(data.suberr[:,  :y1])
    bgerr2  = np.median(data.suberr[:,y2:  ])
    estsig1 = [bgerr1 for j in range(len(bg_thresh))]
    estsig2 = [bgerr2 for j in range(len(bg_thresh))]

    data.submask[:,  :y1] = sigrej.sigrej(bgdata1, bg_thresh, bgmask1, estsig1)
    data.submask[:,y2:  ] = sigrej.sigrej(bgdata2, bg_thresh, bgmask2, estsig2)

    return data


def fit_bg(data, mask, y1, y2, bg_deg, p3thresh, n, isplots=False):
    '''

    '''
    bg, mask = bg.fitbg(data, mask, y1, y2, deg=bg_deg,
                             threshold=p3thresh, isrotate=2, isplots=isplots)
    return (bg, mask, n)
