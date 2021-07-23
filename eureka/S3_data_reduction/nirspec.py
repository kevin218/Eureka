# NIRSpec specific rountines go here
import os
import numpy as np
from astropy.io import fits
from eureka.S3_data_reduction import sigrej
from eureka.S3_data_reduction import background
from eureka.S3_data_reduction import bright2flux as b2f

def read(filename, data, meta):
    '''
    Reads single FITS file from JWST's NIRSpec instrument.

    Parameters
    ----------
    filename          : Single filename to read, should be the Stage 1, *_rateints.fits file.
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
    Updated for NIRSpec by Aarynn Carter/Eva-Maria Ahrer  June 2021
    '''

    assert isinstance(filename, str)

    # Decide whether to perform the Stage 2 processing ourselves.
    # if stage2_processing:
    # 	# Run pipeline on a *_rateints.fits Stage 1 data product, but avoiding significant subarray trimming.
    # 	stage2_filename = process_to_stage2(filename, do_assignwcs=do_assignwcs, do_extract2d=do_extract2d, do_srctype=do_srctype, do_flatfield=do_flatfield, do_photom=do_photom, delete_files=delete_files)
    # else:
    # 	# Use the input file as is.
    # 	stage2_filename = filename


    # Now we can start working with the data.
    hdulist 		= fits.open(filename)
    data.mhdr 		= hdulist[0].header
    data.shdr 		= hdulist['SCI',1].header

    data.intstart 	= 1
    print('WARNING: Manually setting INTSTART to 1 for NIRSpec CV3 data.')
    #data.intstart    = data.mhdr['INTSTART']
    data.intend 	= data.mhdr['NINTS']

    data.data 		= hdulist['SCI',1].data
    data.err 		= hdulist['ERR',1].data
    data.dq 		= hdulist['DQ',1].data
    data.wave 		= hdulist['WAVELENGTH',1].data
    data.v0 		= hdulist['VAR_RNOISE',1].data
    data.int_times	= hdulist['INT_TIMES',1].data[data.intstart-1:data.intend]

    # Record integration mid-times in BJD_TDB
    # data.bjdtdb = data.int_times['int_mid_BJD_TDB']
    # There is no time information in the simulated NIRSpec data
    print('WARNING: The timestamps for the simulated NIRSpec data are currently hardcoded '
          'because they are not in the .fits files themselves')
    data.bjdtdb = np.linspace(data.mhdr['EXPSTART'], data.mhdr['EXPEND'], data.intend)

    # NIRSpec CV3 data has a lot of NaNs in the data and err arrays, which is making life difficult.
    print('WARNING: Manually changing NaNs from DATA and ERR arrays to 0 for the CV3 data')
    data.err[np.where(np.isnan(data.err))] = 0
    data.data[np.where(np.isnan(data.data))] = 0

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
    bg, mask = background.fitbg(data, mask, y1, y2, deg=bg_deg,
                             threshold=p3thresh, isrotate=2, isplots=isplots)
    return (bg, mask, n)
