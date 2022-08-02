import numpy as np


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

    # Compute the center of mass of each column
    column_coms = (np.sum(pix_centers[:, None]*data, axis=0) /
                   np.sum(data, axis=0))
    
    #check if column_coms is smooth
    coms=column_coms.flatten()
    grad=np.gradient(coms)
    badpix_f=np.where(np.abs(grad)>np.std(grad))
    badpix_f=badpix_f[0]
    for bp_l in badpix_f:
        if (bp_l+2) in badpix_f:           
            bp=bp_l+1
            if (bp_l+3) not in badpix_f and (bp_l-1) not in badpix_f:
                column_coms[bp,]=(coms[bp_l]+coms[bp_l+2])/2.
            else:
                column_coms[bp,]=(coms[bp_l-1]+coms[bp_l+3])/2.    

    
    #convert com to integers (pixels)
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
    log.writelog('  Correcting curvature and bringing the trace to the '
                 'center of the detector...', mute=(not meta.verbose))
    # This method only works with the median profile for the extraction
    log.writelog('  !!! Ensure that you are using meddata for the optimal '
                 'extraction profile !!!', mute=(not meta.verbose))

    # Find the median shift needed to bring the trace centered on the detector
    # obtain the median frame
    data_ma = np.ma.masked_where(data.mask.values == 0, data.flux.values)
    median_frame = np.ma.median(data_ma, axis=0).data
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

    return data, meta
