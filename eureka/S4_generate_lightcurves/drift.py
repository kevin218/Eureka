import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
from tqdm import tqdm
from ..lib import gaussian as g
from . import plots_s4

# Measure spectrum drift over all frames and all non-destructive reads.
def spec1D(spectra, meta, log):
    '''Measures the 1D spectrum drift over all integrations.

    Parameters
    ----------
    spectra:    ndarray
        2D array of flux values (nint, nx).
    meta:   MetaClass
        The metadata object.
    log:    logedit.Logedit
        The open log in which notes from this step can be added.

    Returns
    -------
    meta:   MetaClass
        The updated metadata object.

    Notes
    -----
    History:

    - Dec 2013 KBS
        Written for HST.
    - Jun 2021 KBS
        Updated for JWST.
    '''
    if meta.drift_postclip != None:
        meta.drift_postclip = -meta.drift_postclip
    meta.drift1d    = np.zeros(meta.n_int)
    meta.driftmask   = np.zeros(meta.n_int,dtype=int)
    ref_spec        = np.copy(spectra[meta.drift_iref,meta.drift_preclip:meta.drift_postclip])
    # correlate.py sometimes performs better when the mean is subtracted
    if meta.sub_mean:
        #Zero-mean for cross correlation
        ref_spec-= np.mean(ref_spec[meta.drift_range:-meta.drift_range][np.where(np.isnan(ref_spec[meta.drift_range:-meta.drift_range]) == False)])
    ref_spec[np.where(np.isnan(ref_spec) == True)] = 0
    nx          = len(ref_spec)
    for n in tqdm(range(meta.n_int)):
        fit_spec    = np.copy(spectra[n,meta.drift_preclip:meta.drift_postclip])
        #Trim data to achieve accurate cross correlation without assumptions over interesting region
        #http://stackoverflow.com/questions/15989384/cross-correlation-of-non-periodic-function-with-numpy
        fit_spec    = fit_spec[meta.drift_range:-meta.drift_range]
        # correlate.py sometimes performs better when the mean is subtracted
        if meta.sub_mean:
            fit_spec     -= np.mean(fit_spec[np.where(np.isnan(fit_spec) == False)])
        fit_spec[np.where(np.isnan(fit_spec) == True)] = 0
        try:
            #vals = np.correlate(ref_spec, fit_spec, mode='valid')
            vals = sps.correlate(ref_spec, fit_spec, mode='valid', method='fft')
            if meta.isplots_S4 >= 5:
                plots_s4.cc_spec(meta, ref_spec, fit_spec, nx, n)
                plots_s4.cc_vals(meta, vals, n)
            argmax      = np.argmax(vals)
            subvals     = vals[argmax-meta.drift_hw:argmax+meta.drift_hw+1]
            params, err = g.fitgaussian(subvals/subvals.max(), guess=[meta.drift_hw/5., meta.drift_hw*1., 1])
            meta.drift1d[n]= len(vals)//2 - params[1] - argmax + meta.drift_hw
            #meta.drift1d[n]= len(vals)/2 - params[1] - argmax + meta.drift_hw
            meta.driftmask[n] = 1
        except:
            log.writelog(f'  Cross correlation failed. Integration {n} marked as bad.')


    return meta
