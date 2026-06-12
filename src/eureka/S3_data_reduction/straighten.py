import numpy as np
from ..lib import smooth
from . import plots_s3
from .source_pos import gauss
from scipy.optimize import curve_fit


def find_column_median_shifts(data, meta, m):
    '''Takes the median frame (in time) and finds the
    center of mass (COM) in pixels for each column. It then returns the needed
    shift to apply to each column to bring the COM to the center

    Parameters
    ----------
    data : ndarray (2D)
        The median of all data frames
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    m : int
        The file number.

    Returns
    -------
    shifts : ndarray (2D)
        The shifts to apply to each column to straighten the trace.
    new_center : int
        The central row of the detector where the trace is moved to.
    '''
    # make a copy of the array
    data = np.ma.masked_invalid(data)

    # get the dimensions of the data
    nb_rows = data.shape[0]

    # define an array of pixel positions (in their centers)
    pix_centers = np.arange(nb_rows)

    # Do a super quick/simple background subtraction to reduce biases
    data = np.copy(data) - np.ma.min(data, axis=0)

    # Compute the center of mass of each column
    column_coms = (np.ma.sum(pix_centers[:, None]*data, axis=0) /
                   np.ma.sum(data, axis=0))

    # Center of mass doesn't always work well with calibrated spectra
    # Therefore, use CoM as starting point for Gaussian fit
    if meta.calibrated_spectra:
        # Rough initial guess (heigh, center, width, BG)
        width = np.ma.sqrt(np.ma.sum(data[:, 0] *
                           (pix_centers-column_coms[0])**2) /
                           np.ma.sum(data[:, 0]))
        guess = [np.ma.max(data[:, 0]), column_coms[0], width, 0]
        for i in np.arange(data.shape[1]):
            # Gaussian fit for each column
            params, _ = curve_fit(gauss, pix_centers, data[:, i],
                                  guess, maxfev=10000)
            column_coms[i] = params[1]
            # Update guess for next iteration
            guess = params

    # Smooth CoM values to get rid of outliers
    smooth_coms = smooth.medfilt(column_coms, 11)
    # if a value in smooth coms is nan, set it to the last non-nan value
    smooth_coms[np.isnan(smooth_coms)] = \
        smooth_coms[~np.isnan(smooth_coms)][-1]

    # Convert to integer pixels
    int_coms = np.round(smooth_coms).astype(int)

    if meta.isplots_S3 >= 1:
        plots_s3.curvature(meta, column_coms, smooth_coms, int_coms, m)

    # define the new center (where we will align the trace) in the
    # middle of the detector
    new_center = int(nb_rows/2) - 1

    # define an array containing the needed shift to bring the COMs to the
    # new center for each column of each integration
    shifts = new_center - int_coms

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
        all_idcs = list(np.ogrid[[slice(0, n) for n in arr.shape]])

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


def straighten_trace(data, meta, log, m):
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
    m : int
        The file number.

    Returns
    -------
    data : Xarray Dataset
        The updated Dataset object with the fits data stored inside.
    meta : eureka.lib.readECF.MetaClass
        The updated metadata object.
    '''
    if meta.fittype != 'meddata':
        # This method only works with the median profile for the extraction
        log.writelog('  !!! Strongly recommend using meddata as the optimal '
                     'extraction profile !!!', mute=(not meta.verbose))

    log.writelog('  Correcting curvature and bringing the trace to the '
                 'center of the detector...', mute=(not meta.verbose))
    # compute the correction needed from this median frame
    shifts, new_center = find_column_median_shifts(data.medflux, meta, m)

    # Correct wavelength (only one frame)
    log.writelog('    Correcting the wavelength solution...',
                 mute=(not meta.verbose))
    # broadcast to (1, detector.shape) which is the expected shape of
    # the function
    single_shift = np.expand_dims(shifts, axis=0)
    wave_data = np.expand_dims(data.wave_2d.values, axis=0)
    # apply the correction and update wave_1d accordingly
    data.wave_2d.values = roll_columns(wave_data, single_shift)[0]
    data.wave_1d.values = data.wave_2d[new_center].values
    # broadcast the shifts to the number of integrations
    shifts = np.reshape(np.repeat(shifts, data.flux.shape[0]),
                        (data.flux.shape[0], data.flux.shape[2]), order='F')

    log.writelog('    Correcting the curvature over all integrations...',
                 mute=(not meta.verbose))

    # apply the shifts to the data
    data.flux.values = roll_columns(data.flux.values, shifts)
    data.mask.values = roll_columns(data.mask.values, shifts)
    data.err.values = roll_columns(data.err.values, shifts)
    data.dq.values = roll_columns(data.dq.values, shifts)
    data.v0.values = roll_columns(data.v0.values, shifts)
    data.medflux.values = roll_columns(np.expand_dims(data.medflux.values,
                                       axis=0), shifts).squeeze()

    # update the new src_ypos
    log.writelog(f'    Updating src_ypos to new center, row {new_center}...',
                 mute=(not meta.verbose))
    meta.src_ypos = new_center

    return data, meta
