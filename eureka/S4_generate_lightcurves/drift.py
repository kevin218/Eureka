import numpy as np
#import scipy.ndimage.interpolation as spni
import matplotlib.pyplot as plt
import gaussian as g


# Measure spectrum drift over all frames and all non-destructive reads.
def spec1D(spectra, meta, log):
    '''
    Measures the 1D spectrum drift over all integrations.

    Parameters
    ----------
    spectra     : 2D array of flux values (nint, nx)
    meta        : MetaData object

    Returns
    -------
    meta        : Updated MetaData object

    History
    -------
    Written for HST by KBS          Dec 2013
    Updated for JWST by KBS         Jun 2021

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
    for n in range(meta.n_int):
        fit_spec    = np.copy(spectra[n,meta.drift_preclip:meta.drift_postclip])
        #Trim data to achieve accurate cross correlation without assumptions over interesting region
        #http://stackoverflow.com/questions/15989384/cross-correlation-of-non-periodic-function-with-numpy
        fit_spec    = fit_spec[meta.drift_range:-meta.drift_range]
        # correlate.py sometimes performs better when the mean is subtracted
        if meta.sub_mean:
            fit_spec     -= np.mean(fit_spec[np.where(np.isnan(fit_spec) == False)])
        fit_spec[np.where(np.isnan(fit_spec) == True)] = 0
        #try:
        vals        = np.correlate(ref_spec, fit_spec, mode='valid')
        if meta.isplots_S4 >= 5:
            plt.figure(4500)
            plt.clf()
            plt.plot(range(nx), ref_spec, '-')
            plt.plot(range(meta.drift_range,nx-meta.drift_range), fit_spec, '-')
            #plt.savefig()
            plt.figure(4501)
            plt.clf()
            plt.plot(range(-meta.drift_range,meta.drift_range+1), vals, '.')
            #plt.savefig()
            plt.pause(0.1)
        argmax      = np.argmax(vals)
        subvals     = vals[argmax-meta.drift_hw:argmax+meta.drift_hw+1]
        params, err = g.fitgaussian(subvals/subvals.max(), guess=[meta.drift_hw/5., meta.drift_hw*1., 1])
        meta.drift1d[n]= len(vals)/2 - params[1] - argmax + meta.drift_hw
        #drift1d[n,m,i]    = nx//2. - argmax - params[1] + width
        '''
        vals        = np.correlate(ref_spec, fit_spec, mode='valid')
        params, err = g.fitgaussian(vals, guess=[width/5., width*1., vals.max()-np.median(vals)])
        drift[n,m,i]    = len(vals)/2 - params[1]
        #FINMDE
        plt.figure(4)
        plt.clf()
        plt.plot(vals/vals.max(),'o')
        ymin,ymax=plt.ylim()
        plt.vlines(params[1], ymin, ymax, colors='k')
        plt.figure(5)
        plt.clf()
        plt.plot(range(nx),ref_spec,'-k')
        plt.plot(range(validRange,nx-validRange), fit_spec,'-r')
        plt.pause(0.1)
        '''
        meta.driftmask[n] = 1
        #except:
        #    log.writelog(f'  Cross correlation failed. Integration {n} marked as bad.')


    return meta
