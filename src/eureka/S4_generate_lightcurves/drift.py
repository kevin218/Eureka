import numpy as np
import scipy.signal as sps
from tqdm import tqdm
from ..lib import gaussian as g
from . import plots_s4
from astropy.convolution import convolve, Box1DKernel


def highpassfilt(signal, highpassWidth):
    '''Run a signal through a highpass filter to remove high frequency signals.

    This function can be used to compute the continuum of a signal to be
    subtracted.

    Parameters
    ----------
    signal : ndarray (1D)
        1D array of values.
    highpassWidth : int
        The width of the boxcar filter to use.

    Returns
    -------
    smoothed_signal : ndarray (1D)
        An array containing the smoothed signal.
    '''
    g = Box1DKernel(highpassWidth)
    return convolve(signal, g, boundary='extend')


def spec1D(spectra, meta, log, mask=None):
    '''Measures the 1D spectrum drift over all integrations.

    Measure spectrum drift over all frames and all non-destructive reads.

    Parameters
    ----------
    spectra : ndarray
        2D array of flux values (nint, nx).
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    mask : ndarray (1D); optional
        A mask array to use if spectra is not a masked array. Defaults to None
        in which case only the invalid values of spectra will be masked.

    Returns
    -------
    drift1d : ndarray
        1D array of spectrum drift values.
    driftwidth : ndarray
        1D array of the widths of the Gaussians fitted to the CCFs.
    driftmask : ndarray
        1D masked array, where True is masked.
    '''
    spectra = np.ma.masked_invalid(np.ma.copy(spectra))
    spectra = np.ma.masked_where(mask, spectra)

    if meta.drift_postclip is not None:
        meta.drift_postclip = -meta.drift_postclip
    drift1d = np.zeros(meta.n_int)
    driftwidth = np.zeros(meta.n_int)
    driftmask = np.zeros(meta.n_int, dtype=bool)
    ref_spec = np.ma.copy(spectra[meta.drift_iref,
                                  meta.drift_preclip:meta.drift_postclip])
    if meta.sub_continuum:
        # Subtract off the continuum as computed using a highpass filter
        ref_spec -= highpassfilt(ref_spec, meta.highpassWidth)
        ref_spec = ref_spec[int(np.ceil(meta.highpassWidth/2)):]
    if meta.sub_mean:
        # Zero-mean for cross correlation
        # correlate.py sometimes performs better when the mean is subtracted
        ref_spec -= np.ma.mean(ref_spec[meta.drift_range:-meta.drift_range])
    ref_spec[np.isnan(ref_spec)] = 0
    iterfn = range(meta.n_int)
    if meta.verbose:
        iterfn = tqdm(iterfn)
    for n in iterfn:
        fit_spec = np.ma.copy(spectra[n,
                                      meta.drift_preclip:meta.drift_postclip])
        # Trim data to achieve accurate cross correlation without assumptions
        # over interesting region
        # http://stackoverflow.com/questions/15989384/cross-correlation-of-non-periodic-function-with-numpy
        fit_spec = fit_spec[meta.drift_range:-meta.drift_range]
        # correlate.py sometimes performs better when the mean is subtracted
        if meta.sub_continuum:
            # Subtract off the continuum as computed using a highpass filter
            fit_spec -= highpassfilt(fit_spec, meta.highpassWidth)
            fit_spec = fit_spec[int(np.ceil(meta.highpassWidth/2)):]
        if meta.sub_mean:
            fit_spec -= np.ma.mean(fit_spec)
        fit_spec[np.isnan(fit_spec)] = 0
        try:
            vals = sps.correlate(ref_spec, fit_spec, mode='valid',
                                 method='fft')
            if meta.isplots_S4 >= 5 and n < meta.nplots:
                plots_s4.cc_spec(meta, ref_spec, fit_spec, n)
                plots_s4.cc_vals(meta, vals, n)
            argmax = np.ma.argmax(vals)
            subvals = vals[argmax-meta.drift_hw:argmax+meta.drift_hw+1]
            params, err = g.fitgaussian(subvals/subvals.max(),
                                        guess=[meta.drift_hw/5.,
                                               meta.drift_hw*1., 1])
            drift1d[n] = len(vals)//2-params[1]-argmax+meta.drift_hw
            # meta.drift1d[n] = len(vals)/2-params[1]-argmax+meta.drift_hw
            driftwidth[n] = params[0]

        except:
            # FINDME: Need change this bare except to only
            # catch the specific exception
            log.writelog(f'  Cross correlation failed. Integration {n} marked '
                         f'as bad.')
            driftmask[n] = True

    return drift1d, driftwidth, driftmask
