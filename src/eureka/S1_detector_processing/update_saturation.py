#! /usr/bin/env python

# File with functions to modify the saturation
# flagging routines
# PA Roy, July 27th 2022
import numpy as np
from jwst.datamodels import dqflags


def update_sat(input_model, log, meta):
    '''Function that flags saturated pixels more aggressively.
    The flags are added to the group_dq array.

    Parameters:
    -----------
    input_model : jwst.datamodels.QuadModel
        The input group-level data product before ramp fitting.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.

    Returns:
    --------
    input_model : jwst.datamodels.QuadModel
        The input group-level data product with updated saturation flags.
    '''
    log.writelog('Applying a more severe saturation flagging.')

    # Compute the median of the groupdq map
    if meta.dq_sat_mode == "percentile":
        log.writelog('  Creating a saturation map based on the ' +
                     'percentile set in the ecf file.')
        median_sat = np.percentile(input_model.groupdq,
                                   meta.dq_sat_percentile, axis=0).astype(int)
    elif meta.dq_sat_mode == "min":
        log.writelog('  Creating a saturation map based on the ' +
                     'minimum saturated group for each pixel.')
        median_sat = np.max(input_model.groupdq, axis=0).astype(int)
    elif meta.dq_sat_mode == "defined":
        log.writelog('  Creating a saturation map based on the ' +
                     'user defined columns.')
        median_sat = np.zeros_like(np.median(input_model.groupdq, axis=0))

    # Store the saturation flag value
    sat_flag = dqflags.pixel['SATURATED']
    median_sat_mask = (median_sat == sat_flag)

    # Pull do not use flag value from JWST
    do_not_use_flag = dqflags.pixel['DO_NOT_USE']

    # Expand saturated pixels to full columns
    log.writelog('  Expand flags along columns.')
    new_sat_mask = 1 * median_sat_mask
    ngrp = new_sat_mask.shape[0]
    ncols = new_sat_mask.shape[1]
    nrows = new_sat_mask.shape[2]
    if meta.dq_sat_mode == "percentile" or meta.dq_sat_mode == "min":
        for i in range(ngrp):
            is_col_sat = np.sum(median_sat_mask[i, :, :], axis=0).astype(bool)
            new_sat_mask[i, :, :] = np.broadcast_to(is_col_sat, (ncols, nrows))
    elif meta.dq_sat_mode == "defined":
        for i in range(ngrp):
            c1 = meta.dq_sat_columns[i][0]
            c2 = meta.dq_sat_columns[i][1]
            new_sat_mask[i, :, c1:c2] = sat_flag

    # Expand saturation flags to one group before
    if meta.expand_prev_group:
        log.writelog('  Expand flags to previous group.')
        for i in range(ngrp-1):
            new_sat_mask[i, :, :] += new_sat_mask[i+1, :, :]

    # If flags in first group, raise warning that columns will not be used
    if np.count_nonzero(new_sat_mask[0]) > 0:
        log.writelog('  WARNING:')
        log.writelog('    Saturation found in the first group.')
        log.writelog('    Marking saturated columns as DO_NOT_USE')

    # If flags in the second group, raise a warning
    if np.count_nonzero(new_sat_mask[1]) > 0:
        log.writelog('  WARNING:')
        log.writelog('    Saturation flags found in the 2nd group ')
        log.writelog('    This means some columns only have nGroup=1... ')
        log.writelog('    Use caution for this part of the detector.')

    # Ensure we still have a boolean mask
    new_sat_mask = new_sat_mask.astype(bool)

    # Now broadcast that mask back to the number of ints
    new_sat_mask = np.broadcast_to(new_sat_mask, input_model.groupdq.shape)

    # Saturation flagging conditions
    # Where our saturation mask is True and dq is not already flagged
    condition = np.nonzero((new_sat_mask) & (input_model.groupdq == 0))
    full_saturation = np.nonzero(new_sat_mask[0, :, :] &
                                 (input_model.groupdq == 0))
    # Now update the groupdq map
    input_model.groupdq[condition] = sat_flag
    input_model.groupdq[full_saturation] = do_not_use_flag

    return input_model
