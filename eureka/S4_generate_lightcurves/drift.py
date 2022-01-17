import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
from tqdm import tqdm
from ..lib import gaussian as g
from . import plots_s4
from astropy.convolution import convolve, Box1DKernel

def highpassfilt(signal, highpassWidth):
    '''Run a signal through a highpass filter to remove high frequency signals.

    This function can be used to compute the continuum of a signal to be subtracted.

    Parameters
    ----------
    signal: ndarray (1D)
        1D array of values
    highpassWidth: int
        The width of the boxcar filter to use.

    Returns
    -------
    smoothed_signal:    ndarray (1D)
        An array containing the smoothed signal.

    Notes
    -----
    History:
    
    - 14 Feb 2018 Lisa Dang
        Written for early version of SPCA
    - 23 Sep 2019 Taylor Bell
        Generalized upon the code
    - 02 Nov 2021 Taylor Bell
        Added to Eureka!
    '''
    g = Box1DKernel(highpassWidth)
    return convolve(signal, g, boundary='extend')

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
    - Oct 18, 2021 Taylor Bell
        Minor tweak to cc_spec inputs.
    - Nov 02, 2021 Taylor Bell
        Added option for subtraction of continuum using a highpass
        filter before cross-correlation.
    '''
    if meta.drift_postclip != None:
        meta.drift_postclip = -meta.drift_postclip
    meta.drift1d    = np.zeros(meta.n_int)
    meta.driftmask   = np.zeros(meta.n_int,dtype=int)
    ref_spec        = np.copy(spectra[meta.drift_iref,meta.drift_preclip:meta.drift_postclip])
    if meta.sub_continuum:
        # Subtract off the continuum as computed using a highpass filter
        ref_spec -= highpassfilt(ref_spec, meta.highpassWidth)
        ref_spec = ref_spec[int(np.ceil(meta.highpassWidth/2)):]
    if meta.sub_mean:
        #Zero-mean for cross correlation
        # correlate.py sometimes performs better when the mean is subtracted
        ref_spec-= np.mean(ref_spec[meta.drift_range:-meta.drift_range][np.where(np.isnan(ref_spec[meta.drift_range:-meta.drift_range]) == False)])
    ref_spec[np.where(np.isnan(ref_spec) == True)] = 0
    for n in tqdm(range(meta.n_int)):
        fit_spec    = np.copy(spectra[n,meta.drift_preclip:meta.drift_postclip])
        #Trim data to achieve accurate cross correlation without assumptions over interesting region
        #http://stackoverflow.com/questions/15989384/cross-correlation-of-non-periodic-function-with-numpy
        fit_spec    = fit_spec[meta.drift_range:-meta.drift_range]
        # correlate.py sometimes performs better when the mean is subtracted
        if meta.sub_continuum:
            # Subtract off the continuum as computed using a highpass filter
            fit_spec -= highpassfilt(fit_spec, meta.highpassWidth)
            fit_spec = fit_spec[int(np.ceil(meta.highpassWidth/2)):]
        if meta.sub_mean:
            fit_spec     -= np.mean(fit_spec[np.where(np.isnan(fit_spec) == False)])
        fit_spec[np.where(np.isnan(fit_spec) == True)] = 0
        try:
            vals = sps.correlate(ref_spec, fit_spec, mode='valid', method='fft')
            if meta.isplots_S4 >= 5:
                plots_s4.cc_spec(meta, ref_spec, fit_spec, n)
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
