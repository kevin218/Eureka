import numpy as np
from importlib import reload
import multiprocessing as mp

def check_nans(data, mask, log):
    if np.sum(np.isnan(data)) > 0:
        log.writelog("  WARNING: array has NaNs.  Your subregion is probably off the edge of the detector subarray. Masking NaN region and continuing, but you should probably stop and reconsider your choices.")
        inan = np.where(np.isnan(data))
        #subdata[inan]  = 0
        mask[inan]  = 0
    return mask

def trim(ev, data,err, dq, wave, v0):
    subdata = data[:, ev.ywindow[0]:ev.ywindow[1], ev.xwindow[0]:ev.xwindow[1]]
    suberr = err[:, ev.ywindow[0]:ev.ywindow[1], ev.xwindow[0]:ev.xwindow[1]]
    subdq = dq[:, ev.ywindow[0]:ev.ywindow[1], ev.xwindow[0]:ev.xwindow[1]]
    subwave = wave[ev.ywindow[0]:ev.ywindow[1], ev.xwindow[0]:ev.xwindow[1]]
    subv0 = v0[:, ev.ywindow[0]:ev.ywindow[1], ev.xwindow[0]:ev.xwindow[1]]
    subny = ev.ywindow[1] - ev.ywindow[0]
    subnx = ev.xwindow[1] - ev.xwindow[0]

    return subdata, suberr, subdq, subwave, subv0, subny, subnx

def BGsubtraction(ev, log, n_int, bg_y1, bg_y2,subdata, submask, isplots):
    # Load instrument module
    exec('from ..S3_data_reduction import ' + ev.inst + ' as inst', globals())
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
    if ev.ncpu == 1:
        # Only 1 CPU
        for n in range(n_int):
            # Fit sky background with out-of-spectra data
            writeBG(inst.fit_bg(subdata[n], submask[n], bg_y1, bg_y2, ev.bg_deg, ev.p3thresh, n, isplots))
    else:
        # Multiple CPUs
        pool = mp.Pool(ev.ncpu)
        for n in range(n_int):
            res = pool.apply_async(inst.fit_bg,
                                   args=(subdata[n], submask[n], bg_y1, bg_y2, ev.bg_deg, ev.p3thresh, n, isplots),
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

    return subbg, submask, subdata
