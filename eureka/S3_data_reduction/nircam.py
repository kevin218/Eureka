
# NIRCam specific rountines go here
import numpy as np
from astropy.io import fits
from eureka.S3_data_reduction import sigrej, optspex

# Read FITS file from JWST's NIRCam instrument
def read(filename, returnHdr=True):
    '''
    Reads single FITS file from JWST's NIRCam instrument.

    Parameters
    ----------
    filename          : Single filename to read
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
    mhdr    = hdulist[0].header
    shdr    = hdulist['SCI',1].header

    intstart    = mhdr['INTSTART']
    intend      = mhdr['INTEND']

    data    = hdulist['SCI',1].data
    err     = hdulist['ERR',1].data
    dq      = hdulist['DQ',1].data
    wave    = hdulist['WAVELENGTH',1].data
    v0      = hdulist['VAR_RNOISE',1].data
    int_times = hdulist['INT_TIMES',1].data[intstart-1:intend]

    if returnHdr:
        return data, err, dq, wave, v0, int_times, mhdr, shdr
    else:
        return data, err, dq, wave, v0, int_times

def flag_bg(data, err, mask, y1, y2, bg_thresh):
    '''
    Outlier rejection of sky background along time axis
    '''
    bgdata1 = data[:,  :y1]
    bgmask1 = mask[:,  :y1]
    bgdata2 = data[:,y2:  ]
    bgmask2 = mask[:,y2:  ]
    bgerr1  = np.median(err[:,  :y1])
    bgerr2  = np.median(err[:,y2:  ])
    estsig1 = [bgerr1 for j in range(len(bg_thresh))]
    estsig2 = [bgerr2 for j in range(len(bg_thresh))]
    mask[:,  :y1] = sigrej.sigrej(bgdata1, bg_thresh, bgmask1, estsig1)
    mask[:,y2:  ] = sigrej.sigrej(bgdata2, bg_thresh, bgmask2, estsig2)

    return mask

def fit_bg(data, mask, y1, y2, bg_deg, p3thresh, n, isplots=False):
    '''

    '''
    bg, mask = optspex.fitbg(data, mask, y1, y2, deg=bg_deg,
                             threshold=p3thresh, isrotate=2, isplots=isplots)
    return (bg, mask, n)
