#! /usr/bin/env python

# This is based on the RampFitStep from the JWST pipeline, accessed Oct 2021
# adapted by Eva-Maria Ahrer & Aarynn Carter, Oct 2021

import numpy as np
from functools import partial
import warnings

from stcal.ramp_fitting import ramp_fit, utils
import stcal.ramp_fitting.ols_fit
from stcal.ramp_fitting.ramp_fit import suppress_one_good_group_ramps
from stcal.ramp_fitting.ols_fit import discard_miri_groups, \
    find_0th_one_good_group

from stcal.ramp_fitting.likely_fit import LIKELY_MIN_NGROUPS

from jwst.stpipe import Step
from jwst import datamodels

from jwst.datamodels import dqflags

from jwst.ramp_fitting.ramp_fit_step import get_reference_file_subarrays, \
    create_image_model, create_integration_model, set_groupdq

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
        algorithm = option('OLS', 'OLS_C', 'LIKELY', 'mean', 'differenced', default='OLS_C') # 'OLS' and 'OLS_C' use the same underlying algorithm, but OLS_C is implemented in C
        int_name = string(default='')
        save_opt = boolean(default=False) # Save optional output
        opt_name = string(default='')
        suppress_one_group = boolean(default=True)  # Suppress saturated ramps with good 0th group
        firstgroup = integer(default=None)   # Ignore groups before this one (zero indexed)
        lastgroup = integer(default=None)   # Ignore groups after this one (zero indexed)
        maximum_cores = string(default='1') # cores for multiprocessing. Can be an integer, 'half', 'quarter', or 'all'
    """  # noqa: E501

    algorithm = 'OLS_C'  # default
    weighting = 'optimal'  # default
    maximum_cores = 1  # default

    reference_file_types = ['readnoise', 'gain']

    def process(self, input):
        r'''Process a Stage 0 \*_uncal.fits file to Stage 1 \*_rate.fits and
        \*_rateints.fits files.

        Steps taken to perform this processing can follow the default JWST
        pipeline, or alternative methods.

        Parameters
        ----------
        input : RampModel
            The input ramp model to fit the ramps.

        Returns
        -------
        out_model : ImageModel
            The output 2-D image model with the fit ramps.

        int_model : CubeModel
            The output 3-D image model with the fit ramps for each integration.
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

            max_cores = self.maximum_cores
            readnoise_filename = self.get_reference_file(input_model,
                                                         'readnoise')
            gain_filename = self.get_reference_file(input_model, 'gain')

            ngroups = input_model.data.shape[1]
            if (self.algorithm.upper() == "LIKELY" and
                    ngroups < LIKELY_MIN_NGROUPS):
                log.info(
                    f"When selecting the LIKELY ramp fitting algorithm the"
                    f" ngroups needs to be a minimum of {LIKELY_MIN_NGROUPS},"
                    f" but ngroups = {ngroups}.  Due to this, the ramp fitting"
                    f" algorithm is being changed to OLS_C"
                )
                self.algorithm = "OLS_C"

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
                readnoise_2d, gain_2d = get_reference_file_subarrays(
                    input_model, readnoise_model, gain_model)

            log.info(f"Using algorithm = {self.algorithm}")
            log.info(f"Using weighting = {self.weighting}")

            buffsize = ramp_fit.BUFSIZE
            if self.algorithm == "GLS":
                buffsize //= 10

            int_times = input_model.int_times

            # Set the DO_NOT_USE bit in the groupdq values for groups before
            # firstgroup and groups after lastgroup
            firstgroup = self.firstgroup
            lastgroup = self.lastgroup
            groupdqflags = dqflags.group

            if firstgroup is not None or lastgroup is not None:
                set_groupdq(firstgroup, lastgroup, ngroups,
                            input_model.groupdq, groupdqflags)

            # DEFAULT RAMP FITTING ALGORITHM
            if self.algorithm in ['OLS', 'OLS_C']:
                if self.weighting in ['default', 'optimal']:
                    # Want to use the default optimal weighting
                    self.weighting = 'optimal'
                elif self.weighting == 'unweighted':
                    # Want to use the officially-supported unweighted weighting
                    # FINDME: I think this may be the same as our 'uniform'...
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

                image_info, integ_info, _, _ = ramp_fit.ramp_fit(
                    input_model, buffsize, self.save_opt, readnoise_2d,
                    gain_2d, self.algorithm, self.weighting, max_cores,
                    dqflags.pixel, self.suppress_one_group)
            elif self.algorithm.lower() == 'likely':
                # Want to use the newer likelihood-based ramp fitting algorithm
                self.algorithm = 'LIKELY'
                image_info, integ_info, _, _ = ramp_fit.ramp_fit(
                    input_model, buffsize, self.save_opt, readnoise_2d,
                    gain_2d, self.algorithm, self.weighting, max_cores,
                    dqflags.pixel, self.suppress_one_group)
            # FUTURE IMPROVEMENT, WFC3-like differenced frames.
            elif self.algorithm == 'differenced':
                raise ValueError("I can't handle differenced frames yet.")
            # PRIMARILY FOR TESTING, MEAN OF RAMP
            elif self.algorithm == 'mean':
                image_info, integ_info, _ = \
                    mean_ramp_fit(input_model, buffsize, self.save_opt,
                                  readnoise_2d, gain_2d, self.algorithm,
                                  self.weighting, max_cores, dqflags.pixel,
                                  suppress_one_group=self.suppress_one_group)
            else:
                raise ValueError(f'Ramp fitting algorithm "{self.algorithm}"' +
                                 ' not implemented.')

        out_model, int_model = None, None
        # Create models from possibly updated info
        if image_info is not None and integ_info is not None:
            out_model = create_image_model(input_model, image_info)
            out_model.meta.bunit_data = "DN/s"
            out_model.meta.bunit_err = "DN/s"
            out_model.meta.cal_step.ramp_fit = "COMPLETE"
            if (input_model.meta.exposure.type in ["NRS_IFU", "MIR_MRS"]) or (
                input_model.meta.exposure.type in ["NRS_AUTOWAVE", "NRS_LAMP"]
                and input_model.meta.instrument.lamp_mode == "IFU"
            ):
                out_model = datamodels.IFUImageModel(out_model)

            int_model = create_integration_model(input_model, integ_info,
                                                 int_times)
            int_model.meta.bunit_data = "DN/s"
            int_model.meta.bunit_err = "DN/s"
            int_model.meta.cal_step.ramp_fit = "COMPLETE"

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
    pow_wt[snr > 5] = ((snr[snr > 5]-5)/(10-5))*0.6+0.4
    pow_wt[snr > 10] = ((snr[snr > 10]-10)/(20-10))*2.0+1.0
    pow_wt[snr > 20] = ((snr[snr > 20]-20))/(50-20)*3.0+3.0
    pow_wt[snr > 50] = ((snr[snr > 50]-50))/(100-50)*4.0+6.0
    pow_wt[snr > 100] = 10.0

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
        pow_wt[snr > snr_b] = exp_b

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
        wh_pos = np.flatnonzero((sqrt_arg >= 0.) & (gain_sect_r != 0.))
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


def mean_ramp_fit(model, buffsize, save_opt, readnoise_2d, gain_2d,
                  algorithm, weighting, max_cores, dqflags,
                  suppress_one_group):
    """Fit a ramp using average.

    Calculate the count rate for each pixel in all data cube sections and all
    integrations, equal to the weighted mean for all sections (intervals
    between cosmic rays) of the pixel's ramp.

    Parameters
    ----------
    model : data model
        Input data model, assumed to be of type RampModel
    buffsize : int
        Unused. Size of data section (buffer) in bytes
    save_opt : bool
       Calculate optional fitting results
    readnoise_2d : ndarray
        2-D array readnoise for all pixels
    gain_2d : ndarray
        2-D array gain for all pixels
    algorithm : str
        Unused, since algorithm is always 'mean' in this function.
    weighting : str
        Unused.
    max_cores : str
        Unused.
    dqflags : dict
        A dictionary with at least the following keywords:
        DO_NOT_USE, SATURATED, JUMP_DET, NO_GAIN_VALUE, UNRELIABLE_SLOPE
    suppress_one_group : bool
        Find ramps with only one good group and treat it like it has zero good
        groups.

    Returns
    -------
    image_info : tuple
        The tuple of computed ramp fitting arrays.
    integ_info : tuple
        The tuple of computed integration fitting arrays.
    opt_info : tuple
        The tuple of computed optional results arrays for fitting.
    """
    ramp_data = ramp_fit.create_ramp_fit_class(model, 'mean', dqflags,
                                               suppress_one_group)
    # Get readnoise array for calculation of variance of noiseless ramps, and
    #   gain array in case optimal weighting is to be done
    nframes = ramp_data.nframes
    readnoise_2d *= gain_2d / np.sqrt(2. * nframes)

    # Suppress one group ramps, if desired.
    if ramp_data.suppress_one_group_ramps:
        suppress_one_good_group_ramps(ramp_data)

    # For MIRI datasets having >1 group, if all pixels in the final group are
    #   flagged as DO_NOT_USE, resize the input model arrays to exclude the
    #   final group.  Similarly, if leading groups 1 though N have all pixels
    #   flagged as DO_NOT_USE, those groups will be ignored by ramp fitting,
    #   and the input model arrays will be resized appropriately. If all
    #   pixels in all groups are flagged, return None for the models.
    if ramp_data.instrument_name == 'MIRI' and ramp_data.data.shape[1] > 1:
        miri_ans = discard_miri_groups(ramp_data)
        # The function returns False if the removed groups leaves no data to be
        # processed.  If this is the case, return None for all expected
        # variables returned by ramp_fit
        if miri_ans is not True:
            return [None] * 3

    ngroups = ramp_data.data.shape[1]
    if ngroups == 1:
        log.warning('Dataset has NGROUPS=1, so count rates for each\n' +
                    'integration will be calculated as the value of that\n' +
                    '1 group divided by the group exposure time.')

    if not ramp_data.suppress_one_group_ramps:
        # This must be done before the ZEROFRAME replacements to prevent
        # ZEROFRAME replacement being confused for one good group ramps
        # in the 0th group.
        if ramp_data.nframes > 1:
            find_0th_one_good_group(ramp_data)

        if ramp_data.zeroframe is not None:
            zframe_mat, zframe_locs, cnt = \
                utils.use_zeroframe_for_saturated_ramps(ramp_data)
            ramp_data.zframe_mat = zframe_mat
            ramp_data.zframe_locs = zframe_locs
            ramp_data.cnt = cnt

    # Get image data information
    data = ramp_data.data
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

    # Compute differences between groups for each pixel in the data cube
    differences = np.diff(data, axis=1)

    # Calculate effective integration time (once EFFINTIM has been populated
    #   and accessible, will use that instead), and other keywords that will
    #   needed if the pedestal calculation is requested. Note 'nframes'
    #   is the number of given by the NFRAMES keyword, and is the number of
    #   frames averaged on-board for a group, i.e., it does not include the
    #   groupgap.
    effintim = (nframes + groupgap) * frame_time
    # Compute the final 2D array of differences; create rate array
    sum_int = np.sum(differences, axis=1)
    integ_err = 1/np.sqrt(sum_int)
    mean_2d = np.average(sum_int, axis=0)
    var_2d_poisson = 1/mean_2d
    var_poisson = 1/sum_int
    var_rnoise = np.std(readnoise_2d, axis=0)**2*np.ones(sum_int.shape)
    var_2d_rnoise = np.std(readnoise_2d, axis=0)**2*np.ones(mean_2d.shape)
    c_rates = mean_2d / effintim
    image_err = np.sqrt(var_2d_poisson + var_2d_rnoise)
    integ_dq = np.sum(groupdq, axis=1)
    image_info = (c_rates, inpixeldq, var_2d_poisson, var_2d_rnoise, image_err)
    integ_info = (sum_int, integ_dq, var_poisson, var_rnoise, integ_err)

    return image_info, integ_info, None
