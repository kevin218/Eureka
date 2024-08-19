# Module for subtracting a super-bias image from science data sets
# This is based on the superbias_step from the JWST pipeline v1.8.0
# adapted by Kevin Stevenson, Feb 2023

import numpy as np
import logging
from jwst.lib import reffile_utils
from .group_level import mask_trace
from ..lib.astropytable import savetable_S1
from ..lib.smooth import smooth

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def do_correction(input_model, bias_model, meta, log):
    """
    Execute all tasks for Super-Bias Subtraction

    Parameters
    ----------
    input_model: data model object
        science data to be corrected

    bias_model: super-bias model object
        bias data

    Returns
    -------
    output_model: data model object
        bias-subtracted science data
    """
    if meta.bias_correction is not None:
        log.writelog('Doing super-bias subtraction with correction.')

    # Check for subarray mode and extract subarray from the
    # bias reference data if necessary
    if not reffile_utils.ref_matches_sci(input_model, bias_model):
        bias_model = reffile_utils.get_subarray_model(input_model, bias_model)

    if meta.bias_correction is not None and meta.masktrace:
        # Compute trace mask, where ones are good background regions
        input_model = mask_trace(input_model, log, meta)
        trace_mask = input_model.trace_mask

        # Compute median of bias frame outside of trace
        bias_median = np.nanmedian(bias_model.data[np.where(trace_mask)])

        log.writelog('  Computing scale factor for superbias correction.')
        # Compute median of each integration/group outside of trace
        # Using double-for loop to avoid heavy memory usage
        nint, ngroup, nrow, ncol = input_model.data.shape
        scale_factor = np.zeros((nint, ngroup))
        for ii in range(nint):
            data = input_model.data[ii]
            for jj in range(ngroup):
                data_median = np.nanmedian(data[jj][np.where(trace_mask)])
                scale_factor[ii, jj] = data_median/bias_median
        mean_scale_factor = np.nanmean(scale_factor, axis=0)
        log.writelog('  Mean bias scale factors (by group): ' +
                     f'{mean_scale_factor}')

        # Get segment number
        segment = str(input_model.meta.exposure.segment_number).zfill(3)

        # Save scale factor to ECSV file
        fname = meta.outputdir+"S1_seg"+segment+"_BiasScaleFactor.ecsv"
        savetable_S1(fname, scale_factor)

        # Smooth scale factor
        if meta.bias_correction == "smooth":
            for jj in range(ngroup):
                scale_factor[:, jj] = smooth(scale_factor[:, jj],
                                             meta.bias_smooth_length)
            # Save smoothed scale factor to ECSV file
            fname = meta.outputdir + "S1_seg" + segment + \
                                     "_BiasSmoothedScaleFactor.ecsv"
            savetable_S1(fname, scale_factor)

    # Replace NaN's in the superbias with zeros
    bias_model.data[np.isnan(bias_model.data)] = 0.0

    try:
        if meta.bias_correction == "mean":
            # Create 4D bias array for each integration/group and
            # apply scale factor
            bias_data = np.ones((nint, ngroup, nrow, ncol)) * \
                bias_model.data[np.newaxis, np.newaxis, :, :]
            if isinstance(meta.bias_group, int):
                # Scale superbias frame using single value for all groups
                jj = meta.bias_group
                assert jj > 0, f"Group number should be >0, got: {jj}"
                assert jj <= ngroup, f"Group number should be <={ngroup}" + \
                    f", got: {jj}"
                log.writelog('  Applying mean bias correction using ' +
                             f'group {jj}.')
                bias_model.data *= mean_scale_factor[jj-1]
                bias_data = None
            elif meta.bias_group == "each":
                # Scale superbias frame using means values for each group
                log.writelog('  Applying mean bias correction to each group.')
                bias_model.data *= mean_scale_factor[0]
                bias_data *= mean_scale_factor[np.newaxis, :, np.newaxis,
                                               np.newaxis]
            else:
                raise ValueError('Incorrect meta.bias_group value: ' +
                                 f'{meta.bias_group}. Should be ' +
                                 '[1, 2, ..., each].')
        elif (meta.bias_correction == "group_level") or \
             (meta.bias_correction == "smooth"):
            # Create 4D bias array for each integration/group and
            # apply scale factor
            bias_data = np.ones((nint, ngroup, nrow, ncol)) * \
                bias_model.data[np.newaxis, np.newaxis, :, :]
            # Apply correction for zeroframe
            bias_model.data *= mean_scale_factor[0]
            if isinstance(meta.bias_group, int):
                # Scale superbias frame using values from one group
                jj = meta.bias_group
                assert jj > 0, f"Group number should be >0, got: {jj}"
                assert jj <= ngroup, f"Group number should be <={ngroup}" + \
                    f", got: {jj}"
                log.writelog('  Applying group-level bias correction using ' +
                             f'group {jj}.')
                bias_data *= scale_factor[:, jj-1, np.newaxis,
                                          np.newaxis, np.newaxis]
            elif meta.bias_group == "each":
                # Scale superbias frame using values from each group
                log.writelog('  Applying group-level bias correction to ' +
                             'each group.')
                bias_data *= scale_factor[:, :, np.newaxis, np.newaxis]
            else:
                raise ValueError('Incorrect meta.bias_group value: ' +
                                 f'{meta.bias_group}. Should be ' +
                                 '[1, 2, ..., each].')
        else:
            if meta.bias_correction is not None:
                log.writelog('  WARNING: Unrecognized meta.bias_correction '
                             f'value of {meta.bias_correction}... '
                             'No bias correction was applied.')
            bias_data = None
    except Exception as error:
        log.writelog(repr(error))
        raise error

    # Subtract the bias data from the science data
    output_model = subtract_bias(input_model, bias_model, bias_data)

    output_model.meta.cal_step.superbias = 'COMPLETE'

    return output_model


def subtract_bias(input, bias, bias_data=None):
    """
    Subtracts a superbias image from a science data set, subtracting the
    superbias from each group of each integration in the science data.
    The DQ flags in the bias reference image are propagated into the science
    data pixeldq array. The error array is unchanged.

    Parameters
    ----------
    input: data model object
        the input science data

    bias: superbias model object
        the superbias image data

    bias_data: array
        4D array of superbias images

    Returns
    -------
    output: data model object
        bias-subtracted science data
    """
    # Create output as a copy of the input science data model
    output = input.copy()

    # combine the science and superbias DQ arrays
    output.pixeldq = np.bitwise_or(input.pixeldq, bias.dq)

    # Subtract the superbias image from all groups and integrations
    # of the science data
    if bias_data is None:
        # Same bias frame for all groups/integrations
        output.data -= bias.data
    else:
        # Different bias frames for each group/integration
        output.data -= bias_data

    # If ZEROFRAME is present, subtract the super bias.  Zero values
    # indicate bad data, so should be kept zero.
    if input.meta.exposure.zero_frame:
        wh_zero = np.where(output.zeroframe == 0.)
        output.zeroframe -= bias.data
        output.zeroframe[wh_zero] = 0.  # Zero values indicate unusable data

    return output
