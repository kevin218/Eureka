import numpy as np
from importlib import reload
import multiprocessing as mp
from . import sort_nicely as sn
import os

def readfiles(ev):
    ev.segment_list = []
    for fname in os.listdir(ev.topdir + ev.datadir):
        if fname.endswith(ev.suffix + '.fits'):
            ev.segment_list.append(ev.topdir + ev.datadir +'/'+ fname)
    ev.segment_list = sn.sort_nicely(ev.segment_list)
    return ev

def check_nans(data, mask, log):
    if np.sum(np.isnan(data)) > 0:
        log.writelog("  WARNING: array has NaNs.  Your subregion is probably off the edge of the detector subarray. Masking NaN region and continuing, but you should probably stop and reconsider your choices.")
        inan = np.where(np.isnan(data))
        #subdata[inan]  = 0
        mask[inan]  = 0
    return mask

def trim(dat, md):
    dat.subdata = dat.data[:, md.ywindow[0]:md.ywindow[1], md.xwindow[0]:md.xwindow[1]]
    dat.suberr = dat.err[:, md.ywindow[0]:md.ywindow[1], md.xwindow[0]:md.xwindow[1]]
    dat.subdq = dat.dq[:, md.ywindow[0]:md.ywindow[1], md.xwindow[0]:md.xwindow[1]]
    dat.subwave = dat.wave[md.ywindow[0]:md.ywindow[1], md.xwindow[0]:md.xwindow[1]]
    dat.subv0 = dat.v0[:, md.ywindow[0]:md.ywindow[1], md.xwindow[0]:md.xwindow[1]]
    md.subny = md.ywindow[1] - md.ywindow[0]
    md.subnx = md.xwindow[1] - md.xwindow[0]

    return dat, md



def BGsubtraction(dat, md, log, isplots):

    n_int, bg_y1, bg_y2, subdata, submask = md.n_int, md.bg_y1, md.bg_y2, dat.subdata, dat.submask

    # Load instrument module
    exec('from ..S3_data_reduction import ' + md.inst + ' as inst', globals())
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
    if md.ncpu == 1:
        # Only 1 CPU
        for n in range(n_int):
            # Fit sky background with out-of-spectra data
            writeBG(inst.fit_bg(subdata[n], submask[n], bg_y1, bg_y2, md.bg_deg, md.p3thresh, n, isplots))
    else:
        # Multiple CPUs
        pool = mp.Pool(md.ncpu)
        for n in range(n_int):
            res = pool.apply_async(inst.fit_bg,
                                   args=(subdata[n], submask[n], bg_y1, bg_y2, md.bg_deg, md.p3thresh, n, isplots),
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

    dat.subbg, dat.submask, dat.subdata = subbg, submask, subdata

    return dat
