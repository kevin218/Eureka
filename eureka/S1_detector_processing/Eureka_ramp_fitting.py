import logging
from multiprocessing.pool import Pool as Pool
import numpy as np
import time

import warnings

from stcal.ramp_fitting import ramp_fit_class
from stcal.ramp_fitting import utils
from stcal.ramp_fitting import ramp_fit
from stcal.ramp_fitting import ols_fit

from jwst.stpipe import Step
from jwst import datamodels

from jwst.datamodels import dqflags

from jwst.lib import reffile_utils  
from jwst.lib import pipe_utils


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

BUFSIZE = 1024 * 300000  # 300Mb cache size for data section

def mean_ramp_fit_single(model, buffsize, save_opt, readnoise_2d, gain_2d,algorithm, weighting, max_cores, dqflags):
    """
    Fit a ramp using average. Calculate the count rate for each
    pixel in all data cube sections and all integrations, equal to the weighted
    mean for all sections (intervals between cosmic rays) of the pixel's ramp.
    Parameters
    ----------
    ramp_data : RampData
        Input data necessary for computing ramp fitting.
    int_times : None
        Not used
    buffsize : int
        The working buffer size
    save_opt : bool
        Whether to return the optional output model
    readnoise_2d : ndarray
        The read noise of each pixel
    gain_2d : ndarray
        The gain of each pixel
    weighting : str
        'optimal' is the only valid value
    Return
    ------
    image_info : tuple
        The tuple of computed ramp fitting arrays.
    integ_info : tuple
        The tuple of computed integration fitting arrays.
    opt_info : tuple
        The tuple of computed optional results arrays for fitting.
    """
    tstart = time.time()
    
    ramp_data = ramp_fit.create_ramp_fit_class(model, dqflags)
    # Get readnoise array for calculation of variance of noiseless ramps, and
    #   gain array in case optimal weighting is to be done
    nframes = ramp_data.nframes
    readnoise_2d *= gain_2d / np.sqrt(2. * nframes)
    int_times = ramp_data.int_times
    
    # For MIRI datasets having >1 group, if all pixels in the final group are
    #   flagged as DO_NOT_USE, resize the input model arrays to exclude the
    #   final group.  Similarly, if leading groups 1 though N have all pixels
    #   flagged as DO_NOT_USE, those groups will be ignored by ramp fitting, and
    #   the input model arrays will be resized appropriately. If all pixels in
    #   all groups are flagged, return None for the models.
    if ramp_data.instrument_name == 'MIRI' and ramp_data.data.shape[1] > 1:
        miri_ans = ols_fit.discard_miri_groups(ramp_data)
        # The function returns False if the removed groups leaves no data to be
        # processed.  If this is the case, return None for all expected variables
        # returned by ramp_fit
        if miri_ans is not True:
            return [None] * 3

    # Save original shapes for writing to log file, as these may change for MIRI
    n_int, ngroups, nrows, ncols = ramp_data.data.shape
    orig_ngroups = ngroups
    orig_cubeshape = (ngroups, nrows, ncols)

    if ngroups == 1:
        log.warning('Dataset has NGROUPS=1, so count rates for each integration ')
        log.warning('will be calculated as the value of that 1 group divided by ')
        log.warning('the group exposure time.')


    image_info, integ_info, opt_info = ramp_fit_mean(ramp_data, gain_2d, readnoise_2d, save_opt, weighting)

    return image_info, integ_info, opt_info




def ramp_fit_mean(ramp_data, gain_2d, readnoise_2d, save_opt, weighting):
    """
    Calculate effective integration time (once EFFINTIM has been populated accessible, will
    use that instead), and other keywords that will needed if the pedestal calculation is
    requested. Note 'nframes' is the number of given by the NFRAMES keyword, and is the
    number of frames averaged on-board for a group, i.e., it does not include the groupgap.
    Parameters
    ----------
    ramp_data : RampData
        Input data necessary for computing ramp fitting.
    gain_2d : ndarrays
        gain for all pixels
    readnoise_2d : ndarrays
        readnoise for all pixels
    save_opt : bool
       calculate optional fitting results
    weighting : str
    Return
    ------
    image_info : tuple
        The tuple of computed ramp fitting arrays.
    integ_info : tuple
        The tuple of computed integration fitting arrays.
    opt_info : tuple
        The tuple of computed optional results arrays for fitting.
    """
    # Get image data information
    data = ramp_data.data
    err = ramp_data.err
    groupdq = ramp_data.groupdq
    inpixeldq = ramp_data.pixeldq

    # Get instrument and exposure data
    frame_time = ramp_data.frame_time
    group_time = ramp_data.group_time
    groupgap = ramp_data.groupgap
    nframes = ramp_data.nframes

    # Get needed sizes and shapes
    n_int, ngroups, nrows, ncols = data.shape
    imshape = (nrows, ncols)
    cubeshape = (ngroups,) + imshape

    # If all the pixels have their initial groups flagged as saturated, the DQ
    #   in the primary and integration-specific output products are updated,
    #   the other arrays in all output products are populated with zeros, and
    #   the output products are returned to ramp_fit(). If the initial group of
    #   a ramp is saturated, it is assumed that all groups are saturated.
    first_gdq = groupdq[:, 0, :, :]
    if np.all(np.bitwise_and(first_gdq, ramp_data.flags_saturated)):
        image_info, integ_info, opt_info = utils.do_all_sat(
            ramp_data, inpixeldq, groupdq, imshape, n_int, save_opt)

        return image_info, integ_info, opt_info

    # Calculate effective integration time (once EFFINTIM has been populated
    #   and accessible, will use that instead), and other keywords that will
    #   needed if the pedestal calculation is requested. Note 'nframes'
    #   is the number of given by the NFRAMES keyword, and is the number of
    #   frames averaged on-board for a group, i.e., it does not include the
    #   groupgap.
    effintim = (nframes + groupgap) * frame_time
    print(effintim)
    integ_err = np.sqrt(np.sum(err**2,axis=1))
    int_times = np.array([])
    # Compute the final 2D array of differences; create rate array
    print(data.shape)
    sum_int = np.sum(data, axis=1)
    mean_2d = np.average(sum_int, axis=0)
    print(mean_2d.shape)
    var_2d_poisson = 1/mean_2d
    var_poisson = 1/sum_int
    var_rnoise = np.std(readnoise_2d, axis=0)**2*np.ones(sum_int.shape)
    var_2d_rnoise = np.std(readnoise_2d, axis=0)**2*np.ones(mean_2d.shape)
    c_rates = mean_2d / effintim
    image_err = np.sqrt(var_2d_poisson + var_2d_rnoise)
    #del median_diffs_2d
    #del first_diffs_sect
    integ_dq = np.sum(groupdq,axis=1)
    groupdq = np.average(integ_dq,axis=0)
    ramp_data.data = data
    ramp_data.err = err
    ramp_data.groupdq = groupdq
    ramp_data.pixeldq = inpixeldq
    print('hello')
    image_info = (c_rates, groupdq, var_2d_poisson, var_2d_rnoise, image_err)
    integ_info = (sum_int, integ_dq, var_poisson, var_rnoise, int_times, integ_err)

    return image_info, integ_info, None