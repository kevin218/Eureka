#! /usr/bin/env python

# This is based on the RampFitStep from the JWST pipeline, accessed Oct 2021
# adapted by Eva-Maria Ahrer & Aarynn Carter, Oct 2021

import numpy as np
from functools import partial
import warnings

from stcal.ramp_fitting import ramp_fit, utils
import stcal.ramp_fitting.ols_fit

from jwst.stpipe import Step
from jwst import datamodels

from jwst.datamodels import dqflags

from jwst.lib import reffile_utils
from jwst.lib import pipe_utils

from jwst.firstframe.firstframe_step import FirstFrameStep
from jwst.lastframe.lastframe_step import LastFrameStep

from . import update_saturation
from . import group_level
from . import remove390

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

__all__ = ["Eureka_RampFitStep"]


class Eureka_RampFitStep(Step):
    """This step is an alternative to the pipeline rampfitstep to determine
    the count rate for each pixel.
    """

    spec = """
        int_name = string(default='')
        save_opt = boolean(default=False) # Save optional output
        opt_name = string(default='')
        maximum_cores = option('none', 'quarter', 'half', 'all', \
default='none') # max number of processes to create
    """

    algorithm = 'differenced'  # default
    weighting = 'optimal'  # Only weighting allowed for Build 7.1
    maximum_cores = 1  # default

    reference_file_types = ['readnoise', 'gain']

    def process(self, input):
        r'''Process a Stage 0 \*_uncal.fits file to Stage 1 \*_rate.fits and
        \*_rateints.fits files.

        Steps taken to perform this processing can follow the default JWST
        pipeline, or alternative methods.

        Parameters
        ----------
        input : str, tuple, `~astropy.io.fits.HDUList`, ndarray, dict, None

            - None: Create a default data model with no shape.

            - tuple: Shape of the data array.
              Initialize with empty data array with shape specified by the.

            - file path: Initialize from the given file (FITS or ASDF)

            - readable file object: Initialize from the given file
              object

            - `~astropy.io.fits.HDUList`: Initialize from the given
              `~astropy.io.fits.HDUList`.

            - A numpy array: Used to initialize the data array

        Returns
        -------
        out_model : jwst.datamodels.ImageModel
            The output ImageModel to be returned from the ramp fit step.
        int_model : jwst.datamodels.CubeModel
            The output CubeModel to be returned from the ramp fit step.

        Notes
        -----
        History:

        - October 2021 Aarynn Carter and Eva-Maria Ahrer
            Initial version
        - February 2022 Aarynn Carter and Eva-Maria Ahrer
            Updated for JWST version 1.3.3, code restructure
        '''
        with datamodels.RampModel(input) as input_model:

            if self.s1_meta.remove_390hz:
                input_model = remove390.run(input_model, self.s1_log,
                                            self.s1_meta)

                # Need to apply these steps afterward to remove 390 Hz
                if not self.s1_meta.skip_firstframe:
                    self.firstframe = FirstFrameStep()
                    self.firstframe.skip = self.s1_meta.skip_firstframe
                    input_model = self.firstframe(input_model)
                if not self.s1_meta.skip_lastframe:
                    self.lastframe = LastFrameStep()
                    self.lastframe.skip = self.s1_meta.skip_lastframe
                    input_model = self.lastframe(input_model)

            if self.s1_meta.mask_groups:
                self.s1_log.writelog('Manually marking groups '
                                     f'{self.s1_meta.mask_groups} as '
                                     'DO_NOT_USE.')
                for index in self.s1_meta.mask_groups:
                    input_model.groupdq[:, index, :, :] = \
                        np.bitwise_or(input_model.groupdq[:, index, :, :],
                                      dqflags.group['DO_NOT_USE'])

            if self.s1_meta.update_sat_flags:
                input_model = update_saturation.update_sat(input_model,
                                                           self.s1_log,
                                                           self.s1_meta)

            if self.s1_meta.masktrace:
                input_model = group_level.mask_trace(input_model,
                                                     self.s1_log,
                                                     self.s1_meta)

            if self.s1_meta.refpix_corr:
                input_model = group_level.custom_ref_pixel(input_model,
                                                           self.s1_log,
                                                           self.s1_meta)

            if self.s1_meta.grouplevel_bg and not self.s1_meta.remove_390hz:
                input_model = group_level.GLBS(input_model,
                                               self.s1_log,
                                               self.s1_meta)

            readnoise_filename = self.get_reference_file(input_model,
                                                         'readnoise')
            gain_filename = self.get_reference_file(input_model, 'gain')

            log.info('Using READNOISE reference file: %s', readnoise_filename)
            log.info('Using GAIN reference file: %s', gain_filename)

            with datamodels.ReadnoiseModel(readnoise_filename) as \
                 readnoise_model, \
                 datamodels.GainModel(gain_filename) as gain_model:

                # Try to retrieve the gain factor from the gain reference file.
                # If found, store it in the science model meta data, so that
                # it's available later in the gain_scale step, which avoids
                # having to load the gain ref file again in that step.
                if gain_model.meta.exposure.gain_factor is not None:
                    input_model.meta.exposure.gain_factor = \
                        gain_model.meta.exposure.gain_factor

                # Get gain arrays, subarrays if desired.
                frames_per_group = input_model.meta.exposure.nframes
                readnoise_2d, gain_2d = get_reference_file_subarrays(
                    input_model, readnoise_model, gain_model, frames_per_group)

            log.info('Using algorithm = %s' % self.algorithm)
            log.info('Using weighting = %s' % self.weighting)

            buffsize = ramp_fit.BUFSIZE
            if pipe_utils.is_tso(input_model) and hasattr(input_model,
                                                          'int_times'):
                input_model.int_times = input_model.int_times
            else:
                input_model.int_times = None

            # DEFAULT RAMP FITTING ALGORITHM
            if self.algorithm == 'default':
                # In our case, default just means Optimal Least Squares
                self.algorithm = 'OLS'
                if self.weighting == 'default':
                    # Want to use the default optimal weighting
                    pass
                elif self.weighting == 'fixed':
                    # Want to use default weighting, but don't want to
                    # change exponent between pixels.
                    if not isinstance(self.fixed_exponent, (int, float)):
                        raise ValueError('Weighting exponent must be of type' +
                                         ' "int" or "float" for ' +
                                         '"default_fixed" weighting')

                    # Overwrite the exponent calculation function from ols_fit
                    # Pipeline version 1.3.3
                    stcal.ramp_fitting.ols_fit.calc_power = \
                        partial(fixed_power,
                                weighting_exponent=self.fixed_exponent)
                elif self.weighting == 'interpolated':
                    # Want to use an interpolated version of default weighting.

                    # Overwrite the exponent calculation function from ols_fit
                    # Pipeline version 1.3.3
                    stcal.ramp_fitting.ols_fit.calc_power = interpolate_power
                elif self.weighting == 'uniform':
                    # Want each frame and pixel weighted equally

                    # Overwrite the entire optimal calculation function
                    # Pipeline version 1.13.4
                    stcal.ramp_fitting.ols_fit.calc_opt_sums = \
                        calc_opt_sums_uniform_weight
                elif self.weighting == 'custom':
                    # Want to manually assign snr bounds for exponent changes

                    # Overwrite the exponent calculation function from ols_fit
                    # Pipeline version 1.3.3
                    stcal.ramp_fitting.ols_fit.calc_power = \
                        partial(custom_power,
                                snr_bounds=self.custom_snr_bounds,
                                exponents=self.custom_exponents)
                else:
                    raise ValueError('Could not interpret weighting ' +
                                     f'"{self.weighting}".')

                # Important! Must set the weighting to 'optimal' for the actual
                # ramp_fit() function, previous if statements will have changed
                # it's underlying functionality.
                self.weighting = 'optimal'

                image_info, integ_info, _, _ = \
                    ramp_fit.ramp_fit(input_model, buffsize, self.save_opt,
                                      readnoise_2d, gain_2d, self.algorithm,
                                      self.weighting, self.maximum_cores,
                                      dqflags.pixel)
            # FUTURE IMPROVEMENT, WFC3-like differenced frames.
            elif self.algorithm == 'differenced':
                raise ValueError("I can't handle differenced frames yet.")
            # PRIMARILY FOR TESTING, MEAN OF RAMP
            elif self.algorithm == 'mean':
                image_info, integ_info, _ = \
                    mean_ramp_fit_single(input_model, buffsize, self.save_opt,
                                         readnoise_2d, gain_2d, self.algorithm,
                                         self.weighting, self.maximum_cores,
                                         dqflags.pixel)
            else:
                raise ValueError(f'Ramp fitting algorithm "{self.algorithm}"' +
                                 ' not implemented.')

        if image_info is not None:
            out_model = create_image_model(input_model, image_info)
            out_model.meta.bunit_data = 'DN/s'
            out_model.meta.bunit_err = 'DN/s'
            out_model.meta.cal_step.ramp_fit = 'COMPLETE'

        if integ_info is not None:
            int_model = create_integration_model(input_model, integ_info)
            int_model.meta.bunit_data = 'DN/s'
            int_model.meta.bunit_err = 'DN/s'
            int_model.meta.cal_step.ramp_fit = 'COMPLETE'

        return out_model, int_model

#######################################
#         CUSTOM FUNCTIONS            #
#######################################


def fixed_power(snr, weighting_exponent):
    """Fixed version of the weighting exponent.

    This is from `Fixsen, D.J., Offenberg, J.D., Hanisch, R.J., Mather, J.C,
    Nieto, Santisteban, M.A., Sengupta, R., & Stockman, H.S., 2000, PASP,
    112, 1350`.

    Parameters
    ----------
    snr : float32, 1D array
        Signal-to-noise for the ramp segments
    weighting_exponent : int/float
        Exponent to use for all frames/pixels

    Returns
    -------
    pow_wt.ravel() : float32, 1D array
        weighting exponent
    """
    pow_wt = snr.copy()
    pow_wt[:] = weighting_exponent

    return pow_wt.ravel()


def interpolate_power(snr):
    """Interpolated version of the weighting exponent.

    This is from `Fixsen, D.J., Offenberg, J.D., Hanisch, R.J., Mather, J.C,
    Nieto, Santisteban, M.A., Sengupta, R., & Stockman, H.S., 2000, PASP, 112,
    1350`.

    Parameters
    ----------
    snr : float32, 1D array
        signal-to-noise for the ramp segments

    Returns
    -------
    pow_wt.ravel() : float32, 1D array
        weighting exponent
    """
    pow_wt = snr.copy() * 0.0
    pow_wt[np.where(snr > 5)] = ((snr[snr > 5]-5)/(10-5))*0.6+0.4
    pow_wt[np.where(snr > 10)] = ((snr[snr > 10]-10)/(20-10))*2.0+1.0
    pow_wt[np.where(snr > 20)] = ((snr[snr > 20]-20))/(50-20)*3.0+3.0
    pow_wt[np.where(snr > 50)] = ((snr[snr > 50]-50))/(100-50)*4.0+6.0
    pow_wt[np.where(snr > 100)] = 10.0

    return pow_wt.ravel()


def custom_power(snr, snr_bounds, exponents):
    """Customised version to calculate the weighting exponent.

    This is from `Fixsen, D.J., Offenberg, J.D., Hanisch, R.J., Mather, J.C,
    Nieto, Santisteban, M.A., Sengupta, R., & Stockman, H.S., 2000, PASP, 112,
    1350`.

    Exponent array will be overwritten in the order that snr_bounds are
    defined, and therefore in most cases snr_bounds should be provided in
    ascending order.

    Parameters
    ----------
    snr : float32, 1D array
        signal-to-noise for the ramp segments
    snr_bounds : 1D array
        snr bound at which the exponent should change
    exponents : 1D array
        exponents corresponding to each SNR bound
    Returns
    -------
    pow_wt.ravel() : float32, 1D array
        weighting exponent
    """
    pow_wt = snr.copy() * 0.0

    for snr_b, exp_b in zip(snr_bounds, exponents):
        pow_wt[np.where(snr > snr_b)] = exp_b

    return pow_wt.ravel()


def calc_opt_sums_uniform_weight(ramp_data, rn_sect, gain_sect, data_masked,
                                 mask_2d, xvalues, good_pix):
    """Adjusted version of calc_opt_sums() function from stcal ramp fitting.

    Now weights are all equal to 1, except for those that correspond to NaN or
    inf's in the inverse read noise^2 arrays.

    Calculate the sums needed to determine the slope and intercept (and sigma
    of each) using the optimal weights.  For each good pixel's segment, from
    the initial and final indices and the corresponding number of counts,
    calculate the SNR. From the SNR, calculate the weighting exponent using the
    formulation by Fixsen (Fixsen et al, PASP, 112, 1350). Using this exponent
    and the gain and the readnoise, the weights are calculated from which the
    sums are calculated.

    Parameters
    ----------
    ramp_data : RampData
        Input data necessary for computing ramp fitting. (Unused)
    rn_sect : float, 2D array
        read noise values for all pixels in data section
    gain_sect : float, 2D array
        gain values for all pixels in data section
    data_masked : float, 2D array
        masked values for all pixels in data section
    mask_2d : bool, 2D array
        delineates which channels to fit for each pixel
    xvalues : int, 2D array
        indices of valid pixel values for all groups
    good_pix : int, 1D array
        indices of pixels having valid data for all groups

    Returns
    -------
    sumx : float
        sum of xvalues
    sumxx : float
        sum of squares of xvalues
    sumxy : float
        sum of product of xvalues and data
    sumy : float
        sum of data
    nreads_wtd : float, 1D array
        sum of optimal weights
    xvalues : int, 2D array
        rolled up indices of valid pixel values for all groups
    """
    c_mask_2d = mask_2d.copy()  # copy the mask to prevent propagation
    rn_sect = np.float32(rn_sect)

    # Return 'empty' sums if there is no more data to fit
    if data_masked.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), \
            np.array([]), np.array([])

    # get initial group for each good pixel for this semiramp
    fnz = np.argmax(c_mask_2d, axis=0)

    # For those pixels that are all False, set to sentinel value of -1
    fnz[c_mask_2d.sum(axis=0) == 0] = -1

    mask_2d_sum = c_mask_2d.sum(axis=0)   # number of valid groups/pixel

    # get final valid group for each pixel for this semiramp
    ind_lastnz = fnz + mask_2d_sum - 1

    # get SCI value of initial good group for semiramp
    data_zero = data_masked[fnz, range(data_masked.shape[1])]

    # get SCI value of final good group for semiramp
    data_final = data_masked[(ind_lastnz), range(data_masked.shape[1])]
    data_diff = data_final - data_zero  # correctly does *NOT* have nans

    ind_lastnz = 0

    # Use the readnoise and gain for good pixels only
    rn_sect_rav = rn_sect.flatten()[good_pix]
    rn_2_r = rn_sect_rav * rn_sect_rav

    gain_sect_r = gain_sect.flatten()[good_pix]

    # Calculate the sigma for nonzero gain values
    sigma_ir = data_final.copy() * 0.0
    numer_ir = data_final.copy() * 0.0

    # Calculate the SNR for pixels from the readnoise, the gain, and the
    # difference between the last and first reads for pixels where this results
    # in a positive SNR. Otherwise set the SNR to 0.
    sqrt_arg = rn_2_r + data_diff * gain_sect_r
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "invalid value.*", RuntimeWarning)
        wh_pos = np.where((sqrt_arg >= 0.) & (gain_sect_r != 0.))
    numer_ir[wh_pos] = \
        np.sqrt(rn_2_r[wh_pos] + data_diff[wh_pos] * gain_sect_r[wh_pos])
    sigma_ir[wh_pos] = numer_ir[wh_pos] / gain_sect_r[wh_pos]
    snr = data_diff * 0.
    snr[wh_pos] = data_diff[wh_pos] / sigma_ir[wh_pos]
    snr[np.isnan(snr)] = 0.0
    snr[snr < 0.] = 0.0

    del wh_pos

    gain_sect_r = 0
    numer_ir = 0
    data_diff = 0
    sigma_ir = 0

    # Calculate inverse read noise^2 for use in weights
    # Suppress, then re-enable, harmless arithmetic warning
    warnings.filterwarnings("ignore", ".*divide by zero.*", RuntimeWarning)
    invrdns2_r = 1. / rn_2_r
    warnings.resetwarnings()

    rn_sect = 0
    fnz = 0

    # Set weights to all be equal to 1
    wt_h = np.ones(data_masked.shape, dtype=np.float32)

    # Loop through and unweight any nan's inf from the inverse read noise^2
    for jj_rd in range(data_masked.shape[0]):
        wt_h[jj_rd, np.isnan(invrdns2_r)] = 0.
        wt_h[jj_rd, np.isinf(invrdns2_r)] = 0.

    # #### DEFAULT METHOD, COMMENTED FOR REFERENCE #####
    # # Make array of number of good groups, and exponents for each pixel
    # num_nz = (data_masked != 0.).sum(0)  # number of nonzero groups per pixel
    # nrd_data_a = num_nz.copy()
    # num_nz = 0

    # nrd_prime = (nrd_data_a - 1) / 2.
    # nrd_data_a = 0
    # power_wt_r = calc_power(snr)  # Get the interpolated power for this SNR
    # for jj_rd in range(data_masked.shape[0]):
    #     wt_h[jj_rd, :] = \
    #         abs((abs(jj_rd-nrd_prime)/nrd_prime)**power_wt_r)*invrdns2_r
    # wt_h[np.isnan(wt_h)] = 0.
    # wt_h[np.isinf(wt_h)] = 0.
    ###############################################

    # For all pixels, 'roll' up the leading zeros such that the 0th group of
    # each pixel is the lowest nonzero group for that pixel
    wh_m2d_f = np.logical_not(c_mask_2d[0, :])  # ramps with initial grp False
    while wh_m2d_f.sum() > 0:
        data_masked[:, wh_m2d_f] = np.roll(data_masked[:, wh_m2d_f], -1,
                                           axis=0)
        c_mask_2d[:, wh_m2d_f] = np.roll(c_mask_2d[:, wh_m2d_f], -1, axis=0)
        xvalues[:, wh_m2d_f] = np.roll(xvalues[:, wh_m2d_f], -1, axis=0)
        wh_m2d_f = np.logical_not(c_mask_2d[0, :])

    # Create weighted sums for Poisson noise and read noise
    nreads_wtd = (wt_h * c_mask_2d).sum(axis=0)  # using optimal weights

    sumx = (xvalues * wt_h).sum(axis=0)
    sumxx = (xvalues**2 * wt_h).sum(axis=0)

    c_data_masked = data_masked.copy()
    c_data_masked[np.isnan(c_data_masked)] = 0.
    sumy = (np.reshape((c_data_masked * wt_h).sum(axis=0), sumx.shape))
    sumxy = (xvalues*wt_h*np.reshape(c_data_masked, xvalues.shape)).sum(axis=0)

    return sumx, sumxx, sumxy, sumy, nreads_wtd, xvalues


def mean_ramp_fit_single(model, buffsize, save_opt, readnoise_2d, gain_2d,
                         algorithm, weighting, max_cores, dqflags):
    """Fit a ramp using average.

    Calculate the count rate for each pixel in all data cube sections and all
    integrations, equal to the weighted mean for all sections (intervals
    between cosmic rays) of the pixel's ramp.

    Parameters
    ----------
    model : data model
        Input data model.
    buffsize : int
        Unused. The working buffer size.
    save_opt : bool
        Whether to return the optional output model.
    readnoise_2d : ndarray
        The read noise of each pixel.
    gain_2d : ndarray
        The gain of each pixel.
    algorithm : type
        Unused.
    weighting : str
        'optimal' is the only valid value.
    max_cores : str
        The number of CPU cores to used.
    dqflags : dict
        The data quality flags needed for ramp fitting.

    Returns
    -------
    image_info : tuple
        The tuple of computed ramp fitting arrays.
    integ_info : tuple
        The tuple of computed integration fitting arrays.
    opt_info : tuple
        The tuple of computed optional results arrays for fitting.
    """
    ramp_data = ramp_fit.create_ramp_fit_class(model, dqflags)
    # Get readnoise array for calculation of variance of noiseless ramps, and
    #   gain array in case optimal weighting is to be done
    nframes = ramp_data.nframes
    readnoise_2d *= gain_2d / np.sqrt(2. * nframes)

    # For MIRI datasets having >1 group, if all pixels in the final group are
    #   flagged as DO_NOT_USE, resize the input model arrays to exclude the
    #   final group.  Similarly, if leading groups 1 though N have all pixels
    #   flagged as DO_NOT_USE, those groups will be ignored by ramp fitting,
    #   and the input model arrays will be resized appropriately. If all
    #   pixels in all groups are flagged, return None for the models.
    if ramp_data.instrument_name == 'MIRI' and ramp_data.data.shape[1] > 1:
        miri_ans = stcal.ramp_fitting.ols_fit.discard_miri_groups(ramp_data)
        # The function returns False if the removed groups leaves no data to be
        # processed.  If this is the case, return None for all expected
        # variables returned by ramp_fit
        if miri_ans is not True:
            return [None] * 3

    # Save original shapes for writing to log file, as the may change for MIRI
    n_int, ngroups, nrows, ncols = ramp_data.data.shape

    if ngroups == 1:
        log.warning('Dataset has NGROUPS=1, so count rates for each\n' +
                    'integration will be calculated as the value of that\n' +
                    '1 group divided by the group exposure time.')

    image_info, integ_info, opt_info = \
        ramp_fit_mean(ramp_data, gain_2d, readnoise_2d, save_opt, weighting)

    return image_info, integ_info, opt_info


def ramp_fit_mean(ramp_data, gain_2d, readnoise_2d, save_opt, weighting):
    """Calculate effective integration time (once EFFINTIM has been populated
    accessible, will use that instead), and other keywords that will needed if
    the pedestal calculation is requested. Note 'nframes' is the number of
    given by the NFRAMES keyword, and is the number of frames averaged
    on-board for a group, i.e., it does not include the groupgap.

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
        Unused.

    Returns
    -------
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
    groupgap = ramp_data.groupgap
    nframes = ramp_data.nframes

    # Get needed sizes and shapes
    n_int, ngroups, nrows, ncols = data.shape
    imshape = (nrows, ncols)

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
    integ_err = np.sqrt(np.sum(err**2, axis=1))
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
    # del median_diffs_2d
    # del first_diffs_sect
    integ_dq = np.sum(groupdq, axis=1)
    groupdq = np.average(integ_dq, axis=0)
    ramp_data.data = data
    ramp_data.err = err
    ramp_data.groupdq = groupdq
    ramp_data.pixeldq = inpixeldq
    image_info = (c_rates, groupdq, var_2d_poisson, var_2d_rnoise, image_err)
    integ_info = (sum_int, integ_dq, var_poisson, var_rnoise, integ_err)

    return image_info, integ_info, None


#####################################
#    NOTE FOR FUTURE (11/05/2021)   #
#####################################
'''
Space telescope is currently changing the ramp fitting structure on Github
# The following functions:

get_reference_file_subarrays()
create_integration_model()
create_image_model()

Will *not* need to be directly included in this file, we can instead simply
import them from ramp_fitting.ramp_fit_step

i.e.

from ramp_fitting.ramp_fit_step import xxx

However, this has yet incorporated in the pypi repository, so can't do things
straight away.
'''


def get_reference_file_subarrays(model, readnoise_model, gain_model, nframes):
    """Get readnoise array for calculation of variance of noiseless ramps, and
    the gain array in case optimal weighting is to be done.

    The returned readnoise has been multiplied by the gain.

    Parameters
    ----------
    model : data model
        Input data model, assumed to be of type RampModel
    readnoise_model : instance of data Model
        Readnoise for all pixels
    gain_model : instance of gain Model
        Gain for all pixels
    nframes : int
        Unused. Number of frames averaged per group; from the NFRAMES keyword.
        Does not contain the groupgap.

    Returns
    -------
    readnoise_2d : float, 2D array
        Readnoise subarray
    gain_2d : float, 2D array
        Gain subarray
    """
    if reffile_utils.ref_matches_sci(model, gain_model):
        gain_2d = gain_model.data
    else:
        log.info('Extracting gain subarray to match science data')
        gain_2d = reffile_utils.get_subarray_data(model, gain_model)

    if reffile_utils.ref_matches_sci(model, readnoise_model):
        readnoise_2d = readnoise_model.data.copy()
    else:
        log.info('Extracting readnoise subarray to match science data')
        readnoise_2d = reffile_utils.get_subarray_data(model, readnoise_model)

    return readnoise_2d, gain_2d


def create_image_model(input_model, image_info):
    """Creates an ImageModel from the computed arrays from ramp_fit.

    Parameters
    ----------
    input_model : RampModel
        Input RampModel for which the output ImageModel is created.
    image_info : tuple
        The ramp fitting arrays needed for the ImageModel.

    Returns
    -------
    out_model : jwst.datamodels.ImageModel
        The output ImageModel to be returned from the ramp fit step.
    """
    data, dq, var_poisson, var_rnoise, err = image_info

    # Create output datamodel
    out_model = datamodels.ImageModel(data.shape)

    # ... and add all keys from input
    out_model.update(input_model)

    # Populate with output arrays
    out_model.data = data
    out_model.dq = dq
    out_model.var_poisson = var_poisson
    out_model.var_rnoise = var_rnoise
    out_model.err = err
    out_model.int_times = input_model.int_times

    return out_model


def create_integration_model(input_model, integ_info):
    """Creates an CubeModel from the computed arrays from ramp_fit.

    Parameters
    ----------
    input_model : RampModel
        Input RampModel for which the output CubeModel is created.
    integ_info : tuple
        The ramp fitting arrays needed for the CubeModel for each integration.

    Returns
    -------
    int_model : CubeModel
        The output CubeModel to be returned from the ramp fit step.
    """
    data, dq, var_poisson, var_rnoise, err = integ_info
    int_model = datamodels.CubeModel(
        data=np.zeros(data.shape, dtype=np.float32),
        dq=np.zeros(data.shape, dtype=np.uint32),
        var_poisson=np.zeros(data.shape, dtype=np.float32),
        var_rnoise=np.zeros(data.shape, dtype=np.float32),
        err=np.zeros(data.shape, dtype=np.float32))

    int_model.update(input_model)  # ... and add all keys from input

    int_model.data = data
    int_model.dq = dq
    int_model.var_poisson = var_poisson
    int_model.var_rnoise = var_rnoise
    int_model.err = err
    int_model.int_times = input_model.int_times

    return int_model

############################################################
#        SEE NOTE ON LINE ~110 FOR ABOVE FUNCTIONS         #
############################################################
