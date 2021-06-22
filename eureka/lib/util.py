import numpy as np
from importlib import reload
import multiprocessing as mp
from . import sort_nicely as sn
import os


def readfiles(meta):
    """
    Reads in the files saved in topdir + datadir and saves them into a list

    Args:
        meta: metadata object

    Returns:
        meta: metadata object but adds segment_list to metadata containing the sorted data fits files
    """
    meta.segment_list = []
    for fname in os.listdir(meta.topdir + meta.datadir):
        if fname.endswith(meta.suffix + '.fits'):
            meta.segment_list.append(meta.topdir + meta.datadir +'/'+ fname)
    meta.segment_list = sn.sort_nicely(meta.segment_list)
    return meta


def trim(data, meta):
    """
    Removes the edges of the data arrays

    Args:
        dat: Data object
        md: Metadata object

    Returns:
        subdata arrays with trimmed edges depending on xwindow and ywindow which have been set in the S3 ecf
    """
    data.subdata = data.data[:, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    data.suberr  = data.err[:, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    data.subdq   = data.dq[:, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    data.subwave = data.wave[meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    data.subv0   = data.v0[:, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    meta.subny = meta.ywindow[1] - meta.ywindow[0]
    meta.subnx = meta.xwindow[1] - meta.xwindow[0]

    return data, meta


def check_nans(data, mask, log):
    """
    Checks where the data array has NaNs

    Args:
        data: a data array (e.g. data, err, dq, ...)
        mask: input mask
        log: log file where NaNs will be mentioned if existent

    Returns:
        mask: output mask where 0 will be written where the input data array has NaNs
    """
    if np.sum(np.isnan(data)) > 0:
        log.writelog("  WARNING: array has NaNs.  Your subregion is probably off the edge of the detector subarray. Masking NaN region and continuing, but you should probably stop and reconsider your choices.")
        inan = np.where(np.isnan(data))
        #subdata[inan]  = 0
        mask[inan]  = 0
    return mask


def BGsubtraction(data, meta, log, isplots):
    """
    Does background subtraction using inst.fit_bg & optspex.fitbg

    Args:
        dat: Data object
        md: Metadata object
        log: log file
        isplots: amount of plots saved; set in ecf

    Returns:
        Corrects subdata with the background
    """
    n_int, bg_y1, bg_y2, subdata, submask = meta.n_int, meta.bg_y1, meta.bg_y2, data.subdata, data.submask

    # Load instrument module
    exec('from ..S3_data_reduction import ' + meta.inst + ' as inst', globals())
    reload(inst)


    # Write background
    def writeBG(arg):
        bg_data, bg_mask, n = arg
        subbg[n] = bg_data
        submask[n] = bg_mask
        return

    # Compute background for each integration
    log.writelog('  Performing background subtraction')
    subbg = np.zeros((subdata.shape))
    if meta.ncpu == 1:
        # Only 1 CPU
        for n in range(n_int):
            # Fit sky background with out-of-spectra data
            writeBG(inst.fit_bg(subdata[n], submask[n], bg_y1, bg_y2, meta.bg_deg, meta.p3thresh, n, isplots))
    else:
        # Multiple CPUs
        pool = mp.Pool(meta.ncpu)
        for n in range(n_int):
            res = pool.apply_async(inst.fit_bg,
                                   args=(subdata[n], submask[n], bg_y1, bg_y2, meta.bg_deg, meta.p3thresh, n, isplots),
                                   callback=writeBG)
        pool.close()
        pool.join()
        res.wait()

    # Calculate variance
    # bgerr       = np.std(bg, axis=1)/np.sqrt(np.sum(mask, axis=1))
    # bgerr[np.where(np.isnan(bgerr))] = 0.
    # v0[np.where(np.isnan(v0))] = 0.   # FINDME: v0 is all NaNs
    # v0         += np.mean(bgerr**2)
    # variance    = abs(data) / gain + ev.v0    # FINDME: Gain reference file: 'crds://jwst_nircam_gain_0056.fits'
    # variance    = abs(subdata*submask) / gain + v0

    # 9.  Background subtraction
    # Perform background subtraction
    subdata -= subbg

    data.subbg, data.submask, data.subdata = subbg, submask, subdata

    return data
