#! /usr/bin/env python

# File with functions to perform GLBS
# trace masking, 
# custom ref pixel correction for PRISM
# EMMay, Nov 2022
import numpy as np

import scipy.signal as sgn
import scipy.ndimage as spn

from ..S3_data_reduction import background as bkg
import astraeus.xarrayIO as xrio


def GLBS(input_model, log, meta):
    '''Function that performs group level.
    background subtraction. Calls the S3 background code.

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
        The input group-level data product with background removed.
    '''

    log.writelog('  Running Group Level Background Subtraction.')

    all_data = input_model.data
    dq = input_model.groupdq

    meta.inst = input_model.meta.instrument.name.lower()
    meta.n_int = all_data.shape[0]
    meta.int_start = 0
    if not hasattr(meta, 'nplots') or meta.nplots is None:
        meta.int_end = meta.n_int
    elif meta.int_start+meta.nplots > meta.n_int:
        # Too many figures requested, so reduce it
        meta.int_end = meta.n_int
    else:
        meta.int_end = meta.int_start+meta.nplots
    if meta.inst == 'miri':
        meta.isrotate = False

    for ngrp in range(all_data.shape[1]):
        grp_data = all_data[:, ngrp, :, :]
        grp_mask = np.ones(grp_data.shape, dtype=bool)
        dqmask = np.where((dq[:, ngrp, :, :] % 2 == 1) |
                          (dq[:, ngrp, :, :] == 2))
        grp_mask[dqmask] = 0

        if hasattr(meta, 'masktrace') and meta.masktrace:
            trace_mask = np.broadcast_to(input_model.trace_mask,
                                         (grp_data.shape[0],
                                          grp_data.shape[1],
                                          grp_data.shape[2]))
            trace_mask = np.nan_to_num(trace_mask).astype(bool)
            grp_mask *= trace_mask

        xrdata = (['time', 'y', 'x'], grp_data)
        xrmask = (['time', 'y', 'x'], grp_mask)                
        xrdict = dict(flux=xrdata, mask=xrmask)
        data = xrio.makeDataset(dictionary=xrdict)
        data['flux'].attrs['flux_units'] = 'n/a'
        data.attrs['intstart'] = meta.intstart
        if ngrp == all_data.shape[1]-1:
            data = bkg.BGsubtraction(data, meta, 
                                     log, 0, 
                                     meta.isplots_S1)
        else:
            # do not show plots except for the last group
            data = bkg.BGsubtraction(data, meta, 
                                     log, 0, 0)
        all_data[:, ngrp, :, :] = data['flux'].values
    input_model.data = all_data

    return input_model


def mask_trace(input_model, log, meta):
    '''Function that returns a mask centered on the trace.
    used prior to GLBS for curved traces.

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
        The input group-level data product with trace mask object
    '''
    log.writelog('  Masking Curved Trace.')

    all_data = input_model.data

    ngrp = all_data.shape[1]
    ncol = all_data.shape[3]
    nrow = all_data.shape[2]
    trace_mask = np.ones_like(all_data[0, 0, :, :])

    # take a median across groups and smooth                                                                                                                                                                                                           
    med_data = np.nanmedian(all_data[:, ngrp-1, :, :], axis=0)
    smt_data = spn.median_filter(med_data, size=(1, meta.window_len))

    # find the max in a column and smooth
    column_max = np.nanargmax(smt_data, axis=0)
    smooth_coms = sgn.medfilt(column_max, kernel_size=meta.window_len)

    # extend the upper and lower boundaries if requested
    if meta.ignore_low is not None:
        smooth_coms[:meta.ignore_low] = smooth_coms[meta.ignore_low]
    if meta.ignore_hi is not None:
        smooth_coms[meta.ignore_hi:] = smooth_coms[meta.ignore_hi]
    
    # now create mask based on smooth_coms center.                                                                                                                                                                                                         
    for nc in range(ncol):
        mask_low = np.nanmax([0, smooth_coms[nc]-meta.expand_mask])
        mask_hih = np.nanmin([smooth_coms[nc]+meta.expand_mask, nrow-1])

        trace_mask[mask_low:mask_hih, nc] = np.nan

    input_model.trace_mask = trace_mask

    return input_model

    
def custom_ref_pixel(input_model, log, meta):
    '''Function that performs reference pixel
    correction for PRISM

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
        The input group-level data product after ref pix correction
    '''

    all_data = input_model.data
    dq = input_model.groupdq

    nrow = all_data.shape[2]
    row_ind_ref = np.append(np.arange(nrow)[:meta.npix_top], 
                            np.arange(nrow)[nrow-meta.npix_bot:])
    evn_row_ref = row_ind_ref[np.where(row_ind_ref % 2 == 0)[0]]
    odd_row_ref = row_ind_ref[np.where(row_ind_ref % 2 == 1)[0]]

    row_ind = np.arange(nrow)
    evn_ind = np.where(row_ind % 2 == 0)[0]
    odd_ind = np.where(row_ind % 2 == 1)[0]

    for nint in range(all_data.shape[0]):
        for ngrp in range(all_data.shape[1]):
            intgrp_data = all_data[nint, ngrp, :, :]                                                                                                                                                                                                    
            dqmask = np.where((dq[nint, ngrp, :, :] % 2 == 1) | 
                              (dq[nint, ngrp, :, :] == 2))
            intgrp_data[dqmask] = np.nan
            
            if hasattr(meta, 'masktrace') and meta.masktrace:
                intgrp_data *= meta.trace_mask

            odd_med = np.nanmedian(intgrp_data[odd_row_ref, :])
            evn_med = np.nanmedian(intgrp_data[evn_row_ref, :])

            all_data[nint, ngrp, odd_ind, :] -= odd_med
            all_data[nint, ngrp, evn_ind, :] -= evn_med
    input_model.data = all_data

    return input_model

