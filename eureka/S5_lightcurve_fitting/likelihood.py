import numpy as np
from scipy.stats import norm

def ln_like(theta, lc, model, pmin, pmax, freenames):
    """Compute the log-likelihood.

    Parameters
    ----------
    theta: ndarray
        The current estimate of the fitted parameters
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    pmin: ndarray
        The lower-bound for uniform priors.
    pmax: ndarray
        The upper-bound for uniform priors.
    freenames: iterable
        The names of the fitted parameters.

    Returns
    -------
    ln_like_val: ndarray
        The log-likelihood value at the position theta.

    Notes
    -----
    History:

    - December 29-30, 2021 Taylor Bell
        Moved code to separate file, added documentation.
    """
    # params[ifreepars] = freepars
    ilow = np.where(theta < pmin)
    ihi = np.where(theta > pmax)
    theta[ilow] = pmin[ilow]
    theta[ihi] = pmax[ihi]
    model.update(theta, freenames)
    model_lc = model.eval()
    residuals = (lc.flux - model_lc) #/ lc.unc
    ln_like_val = (-0.5 * (np.sum((residuals / lc.unc) ** 2+ np.log(2.0 * np.pi * (lc.unc) ** 2))))
    if len(ilow[0]) + len(ihi[0]) > 0:
        ln_like_val = -np.inf
    return ln_like_val

def lnprior(theta, pmin, pmax):
    """Compute the log-prior.

    Parameters
    ----------
    theta: ndarray
        The current estimate of the fitted parameters
    pmin: ndarray
        The lower-bound for uniform priors.
    pmax: ndarray
        The upper-bound for uniform priors.

    Returns
    -------
    lnprior_prob: ndarray
        The log-prior probability value at the position theta.

    Notes
    -----
    History:

    - December 29-30, 2021 Taylor Bell
        Moved code to separate file, added documentation.
    """
    lnprior_prob = 0.
    n = len(theta)
    for i in range(n):
        if np.logical_or(theta[i] < pmin[i],
                                theta[i] > pmax[i]): lnprior_prob += - np.inf
    return lnprior_prob

def lnprob(theta, lc, model, pmin, pmax, freenames):
    """Compute the log-probability.

    Parameters
    ----------
    theta: ndarray
        The current estimate of the fitted parameters
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    pmin: ndarray
        The lower-bound for uniform priors.
    pmax: ndarray
        The upper-bound for uniform priors.
    freenames:
        The names of the fitted parameters.

    Returns
    -------
    ln_prob_val: ndarray
        The log-probability value at the position theta.

    Notes
    -----
    History:

    - December 29-30, 2021 Taylor Bell
        Moved code to separate file, added documentation.
    """
    ln_like_val = ln_like(theta, lc, model, pmin, pmax, freenames)
    lp = lnprior(theta, pmin, pmax)
    lnprob = ln_like_val + lp
    if not np.isfinite(lnprob):
        lnprob = -np.inf
    return lnprob

#PRIOR TRANSFORMATION TODO: ADD GAUSSIAN PRIORS
def transform_uniform(x, a, b):
    return a + (b - a) * x

def transform_normal(x, mu, sigma):
    return norm.ppf(x, loc=mu, scale=sigma)

def ptform(theta, pmin, pmax):
    p = np.zeros_like(theta)
    n = len(theta)
    for i in range(n):
        p[i] = transform_uniform(theta[i], pmin[i], pmax[i])
    return p

def computeRedChiSq(lc, model, meta, freenames):
    """Compute the reduced chi-squared value.

    Parameters
    ----------
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    meta: MetaObject
        The metadata object.
    freenames: iterable
        The names of the fitted parameters.

    Returns
    -------
    chi2red: float
        The reduced chi-squared value.

    Notes
    -----
    History:

    - December 29-30, 2021 Taylor Bell
        Moved code to separate file, added documentation.
    """
    model_lc = model.eval()
    residuals = (lc.flux - model_lc) #/ lc.unc
    chi2 = np.sum((residuals / lc.unc) ** 2)
    chi2red = chi2 / (len(lc.unc) - len(freenames))

    if meta.run_verbose:
        print('Reduced Chi-squared: ', chi2red)

    return chi2red

def computeRMS(data, maxnbins=None, binstep=1, isrmserr=False):
    """Compute the root-mean-squared and standard error of data for various bin sizes.

    Parameters
    ----------
    data: ndarray
        The residuals after fitting.
    maxnbins: int, optional
        The maximum number of bins. Use None to default to 10 points per bin.
    binstep: int, optional
        Bin step size.
    isrmserr: bool
        True if return rmserr, else False.

    Returns
    -------
    rms: ndarray
        The RMS for each bin size.
    stderr: ndarray
        The standard error for each bin size.
    binsz: ndarray
        The different bin sizes.
    rmserr: ndarray, optional
        The uncertainty in the RMS.

    Notes
    -----
    History:

    - December 29-30, 2021 Taylor Bell
        Moved code to separate file, added documentation.
    """
    # bin data into multiple bin sizes
    npts = data.size
    if maxnbins is None:
        maxnbins = npts / 10.
    binsz = np.arange(1, maxnbins + binstep, step=binstep, dtype=int)
    nbins = np.zeros(binsz.size, dtype=int)
    rms = np.zeros(binsz.size)
    rmserr = np.zeros(binsz.size)
    for i in range(binsz.size):
        nbins[i] = int(np.floor(data.size / binsz[i]))
        bindata = np.zeros(nbins[i], dtype=float)
        # bin data
        # ADDED INTEGER CONVERSION, mh 01/21/12
        for j in range(nbins[i]):
            bindata[j] = data[j * binsz[i]:(j + 1) * binsz[i]].mean()
        # get rms
        rms[i] = np.sqrt(np.mean(bindata ** 2))
        rmserr[i] = rms[i] / np.sqrt(2. * nbins[i])
    # expected for white noise (WINN 2008, PONT 2006)
    stderr = (data.std() / np.sqrt(binsz)) * np.sqrt(nbins / (nbins - 1.))
    if isrmserr is True:
        return rms, stderr, binsz, rmserr
    else:
        return rms, stderr, binsz
