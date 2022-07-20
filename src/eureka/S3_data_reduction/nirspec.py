# NIRSpec specific rountines go here
import numpy as np
from astropy.io import fits
import astraeus.xarrayIO as xrio
from . import nircam, sigrej
from ..lib.util import read_time


def read(filename, data, meta, log):
    '''Reads single FITS file from JWST's NIRCam instrument.

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

    Notes
    -----
    History:

    - November 2012 Kevin Stevenson
        Initial version
    - June 2021 Aarynn Carter/Eva-Maria Ahrer
        Updated for NIRSpec
    - Apr 22, 2022 Kevin Stevenson
        Convert to using Xarray Dataset
    '''
    hdulist = fits.open(filename)

    # Load master and science headers
    data.attrs['filename'] = filename
    data.attrs['mhdr'] = hdulist[0].header
    data.attrs['shdr'] = hdulist['SCI', 1].header
    try:
        data.attrs['intstart'] = data.attrs['mhdr']['INTSTART']-1
        data.attrs['intend'] = data.attrs['mhdr']['INTEND']
    except:
        # FINDME: Need to only catch the particular exception we expect
        print('  WARNING: Manually setting INTSTART to 1 and INTEND to NINTS')
        data.attrs['intstart'] = 0
        data.attrs['intend'] = data.attrs['mhdr']['NINTS']

    sci = hdulist['SCI', 1].data
    err = hdulist['ERR', 1].data
    dq = hdulist['DQ', 1].data
    v0 = hdulist['VAR_RNOISE', 1].data
    wave_2d = hdulist['WAVELENGTH', 1].data
    int_times = hdulist['INT_TIMES', 1].data

    # Record integration mid-times in BJD_TDB
    if (hasattr(meta, 'time_file') and meta.time_file is not None):
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
    time_units = 'BJD_TDB'
    wave_units = 'microns'

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


def flag_bg(data, meta, log):
    '''Outlier rejection of sky background along time axis.

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
        The updated Dataset object with outlier background pixels flagged.
    '''
    log.writelog('  Performing background outlier rejection...',
                 mute=(not meta.verbose))

    meta.bg_y2 = meta.src_ypos + meta.bg_hw
    meta.bg_y1 = meta.src_ypos - meta.bg_hw

    bgdata1 = data.flux[:, :meta.bg_y1]
    bgmask1 = data.mask[:, :meta.bg_y1]
    bgdata2 = data.flux[:, meta.bg_y2:]
    bgmask2 = data.mask[:, meta.bg_y2:]
    # FINDME: KBS removed estsig from inputs to speed up outlier detection.
    # Need to test performance with and without estsig on real data.
    if hasattr(meta, 'use_estsig') and meta.use_estsig:
        # This might not be necessary for real data
        bgerr1 = np.ma.median(np.ma.masked_equal(data.err[:, :meta.bg_y1], 0))
        bgerr2 = np.ma.median(np.ma.masked_equal(data.err[:, meta.bg_y2:], 0))
        estsig1 = [bgerr1 for j in range(len(meta.bg_thresh))]
        estsig2 = [bgerr2 for j in range(len(meta.bg_thresh))]
    else:
        estsig1 = None
        estsig2 = None
    data['mask'][:, :meta.bg_y1] = sigrej.sigrej(bgdata1, meta.bg_thresh,
                                                 bgmask1, estsig1)
    data['mask'][:, meta.bg_y2:] = sigrej.sigrej(bgdata2, meta.bg_thresh,
                                                 bgmask2, estsig2)

    return data


def fit_bg(dataim, datamask, n, meta, isplots=0):
    """Fit for a non-uniform background.

    Uses the code written for NIRCam which works for NIRSpec.

    Parameters
    ----------
    dataim : ndarray (2D)
        The 2D image array.
    datamask : ndarray (2D)
        An array of which data should be masked.
    n : int
        The current integration.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    isplots : int; optional
        The plotting verbosity, by default 0.

    Returns
    -------
    bg : ndarray (2D)
        The fitted background level.
    mask : ndarray (2D)
        The updated mask after background subtraction.
    n : int
        The current integration number.
    """
    return nircam.fit_bg(dataim, datamask, n, meta, isplots=isplots)


def find_column_median_shifts(data):
    '''Takes the median frame (in time) and finds the 
    center of mass (COM) in pixels for each column. It then returns the needed
    shift to apply to each column to bring the COM to the center

    Parameters
    ----------
    data : ndarray (2D)
        The median of all data frames

    Returns
    -------
    shifts : ndarray (2D)
        The shifts to apply to each column to straighten the trace.
    new_center : int
        The central row of the detector where the trace is moved to.
    '''
    # get the dimensions of the data
    nb_rows = data.shape[0]

    # define an array of pixel positions (in their centers)
    pix_centers = np.arange(nb_rows) + 0.5

    # Compute the center of mass of each column and convert to integer (pixels)
    column_coms = (np.sum(pix_centers[:, None]*data, axis=0) /
                   np.sum(data, axis=0))
    column_coms = np.around(column_coms).astype(int)

    # define the new center (where we will align the trace) in the 
    # middle of the detector
    new_center = int(nb_rows/2)

    # define an array containing the needed shift to bring the COMs to the
    # new center for each column of each integration
    shifts = new_center - column_coms

    # ensure columns of zeros or nans are not moved and dont induce bugs
    shifts[column_coms < 0] = 0
    shifts[column_coms > nb_rows] = 0

    return shifts, new_center


def roll_columns(data, shifts):
    '''For each column of each integration, rolls the columns by the 
    values specified in shifts 

    Parameters
    ----------
    data : ndarray (3D)
        Image data.
    shifts : ndarray (2D)
        The shifts to apply to each column to straighten the trace.

    Returns
    -------
    rolled_data : ndarray (3D)
        Image data with the shifts applied.
    '''
    # init straight_data
    rolled_data = np.zeros_like(data)
    # loop over all images (integrations)
    for i in range(len(data)):
        # do the equivalent of 'np.roll' but with a different shift
        # in each column
        
        arr = np.swapaxes(data[i], 0, -1)
        all_idcs = np.ogrid[[slice(0, n) for n in arr.shape]]

        # make the shifts positive
        shifts_i = shifts[i]
        shifts_i[shifts_i < 0] += arr.shape[-1]

        # apply the shifts
        all_idcs[-1] = all_idcs[-1] - shifts_i[:, np.newaxis]
        result = arr[tuple(all_idcs)]
        arr = np.swapaxes(result, -1, 0)

        # store in straight_data
        rolled_data[i] = arr

    return rolled_data
    

def straighten_trace(data, meta, log):
    '''Takes a set of integrations with a curved trace and shifts the 
    columns to bring the center of mass to the middle of the detector
    (and straighten the trace)

    The correction is made by whole pixels (i.e. no fractional pixel shifts)
    The shifts to be applied are computed once from the median frame and then
    applied to each integration in the timeseries

    Parameters
    ----------
    data : Xarray Dataset
            The Dataset object in which the fits data will stored.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with the fits data stored inside.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    '''
    log.writelog('  Correcting curvature and bringing trace in the center '
                 'of the detector...', mute=(not meta.verbose))
    # This method only works with the median profile for the extraction
    log.writelog('  !!! Ensure that you are using meddata for the optimal '
                 'extraction profile !!!', mute=(not meta.verbose))

    # Find the median shift needed to bring the trace centered on the detector
    # obtain the median frame
    median_frame = np.copy(data.medflux.values)
    # compute the correction needed from this median frame
    shifts, new_center = find_column_median_shifts(median_frame)

    # Correct wavelength (only one frame) 
    log.writelog('  Correcting the wavelength solution...',
                 mute=(not meta.verbose))
    # broadcast to (1, detector.shape) which is the expected shape of
    # the function
    single_shift = np.expand_dims(shifts, axis=0)
    wave_data = np.expand_dims(data.wave_2d.values, axis=0)
    # apply the correction and update wave_1d accordingly
    data.wave_2d.values = roll_columns(wave_data, single_shift)[0]
    data.wave_1d.values = data.wave_2d[new_center].values

    log.writelog('  Correcting the curvature over all integrations...',
                 mute=(not meta.verbose))
    # broadcast the shifts to the number of integrations
    shifts = np.reshape(np.repeat(shifts, data.flux.shape[0]),
                        (data.flux.shape[0], data.flux.shape[2]), order='F')

    # apply the shifts to the data
    data.flux.values = roll_columns(data.flux.values, shifts)
    data.err.values = roll_columns(data.err.values, shifts)
    data.dq.values = roll_columns(data.dq.values, shifts)
    data.v0.values = roll_columns(data.v0.values, shifts)
    
    # update the new src_ypos
    log.writelog(f'  Updating src_ypos to new center, row {new_center}...',
                 mute=(not meta.verbose))
    meta.src_ypos = new_center

    # update the median frame
    log.writelog('  Updating median frame now that the trace is corrected...',
                 mute=(not meta.verbose))
    data.medflux.values = np.median(data.flux.values, axis=0)

    return data, meta


def cut_aperture(data, meta, log):
    """Select the aperture region out of each trimmed image.

    Uses the code written for NIRCam which works for NIRSpec.

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
    apdata : ndarray
        The flux values over the aperture region.
    aperr : ndarray
        The noise values over the aperture region.
    apmask : ndarray
        The mask values over the aperture region.
    apbg : ndarray
        The background flux values over the aperture region.
    apv0 : ndarray
        The v0 values over the aperture region.

    Notes
    -----
    History:

    - 2022-06-17, Taylor J Bell
        Initial version based on the code in s3_reduce.py
    """
    return nircam.cut_aperture(data, meta, log)
