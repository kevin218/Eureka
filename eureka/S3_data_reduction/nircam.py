
# NIRCam specific rountines go here
import numpy as np
from astropy.io import fits
from eureka.S3_data_reduction import sigrej, optspex

# Read FITS file from JWST's NIRCam instrument
def read(filename, dat, returnHdr=True):
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
    dat.mhdr    = hdulist[0].header
    dat.shdr    = hdulist['SCI',1].header

    dat.intstart    = dat.mhdr['INTSTART']
    dat.intend      = dat.mhdr['INTEND']

    dat.data    = hdulist['SCI',1].data
    dat.err     = hdulist['ERR',1].data
    dat.dq      = hdulist['DQ',1].data
    dat.wave    = hdulist['WAVELENGTH',1].data
    dat.v0      = hdulist['VAR_RNOISE',1].data
    dat.int_times = hdulist['INT_TIMES',1].data[dat.intstart-1:dat.intend]


    #if returnHdr:
    #    return data, err, dq, wave, v0, int_times, mhdr, shdr
    #else:
    #    return data, err, dq, wave, v0, int_times
    return dat

def flag_bg(dat, md):
    '''
    Outlier rejection of sky background along time axis
    '''

    data, err, mask, y1, y2, bg_thresh = dat.subdata, dat.suberr, dat.submask, md.bg_y1, md.bg_y2, md.bg_thresh

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
