import numpy as np
from scipy.stats import norm
import pdb
from copy import deepcopy

def ln_like(theta, lc, model, freenames):
    """Compute the log-likelihood.

    Parameters
    ----------
    theta: ndarray
        The current estimate of the fitted parameters
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
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
    - January 22, 2022 Megan Mansfield
        Adding ability to do a single shared fit across all channels
    """

    model.update(theta, freenames)
    model_lc = model.eval()
    residuals = (lc.flux - model_lc)
    if "scatter_ppm" in freenames:
        ind = np.where(freenames=="scatter_ppm")
        lc.unc_fit = theta[ind]*1e-6
    else:
        lc.unc_fit = deepcopy(lc.unc)
    ln_like_val = (-0.5 * (np.sum((residuals / lc.unc_fit) ** 2+ np.log(2.0 * np.pi * (lc.unc_fit) ** 2))))
    return ln_like_val

def lnprior(theta, prior1, prior2, priortype):
    """Compute the log-prior.

    Parameters
    ----------
    theta: ndarray
        The current estimate of the fitted parameters
    prior1: ndarray
        The lower-bound for uniform/log uniform priors, or mean for normal priors.
    prior2: ndarray
        The upper-bound for uniform/log uniform priors, or std. dev. for normal priors.
    priortype: ndarray
        Keywords indicating the type of prior for each free parameter.

    Returns
    -------
    lnprior_prob: ndarray
        The log-prior probability value at the position theta.

    Notes
    -----
    History:

    - December 29-30, 2021 Taylor Bell
        Moved code to separate file, added documentation.
    - February 23-25, 2022 Megan Mansfield
        Added log-uniform and Gaussian priors.
    """
    lnprior_prob = 0.
    n = len(theta)
    for i in range(n):
        if priortype[i]=='U':
            if np.logical_or(theta[i] < prior1[i], theta[i] > prior2[i]):
                lnprior_prob += - np.inf
        elif priortype[i]=='N':
            lnprior_prob -= 0.5*(np.sum(((theta[i] - prior1[i])/prior2[i])**2 + np.log(2.0*np.pi*(prior2[i])**2)))
        elif priortype[i]=='LU':
            if np.logical_or(np.log(theta[i]) < prior1[i], np.log(theta[i]) > prior2[i]):
                lnprior_prob += - np.inf
        else:
            raise ValueError("PriorType must be 'U', 'LU', or 'N'")
    return lnprior_prob

def lnprob(theta, lc, model, prior1, prior2, priortype, freenames):
    """Compute the log-probability.

    Parameters
    ----------
    theta: ndarray
        The current estimate of the fitted parameters
    lc: eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    model: eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    prior1: ndarray
        The lower-bound for uniform/log uniform priors, or mean for normal priors.
    prior2: ndarray
        The upper-bound for uniform/log uniform priors, or std. dev. for normal priors.
    priortype: ndarray
        Keywords indicating the type of prior for each free parameter.
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
    - February 23-25, 2022 Megan Mansfield
        Added log-uniform and Gaussian priors.
    """
    ln_like_val = ln_like(theta, lc, model, freenames)
    lp = lnprior(theta, prior1, prior2, priortype)
    lnprob = ln_like_val + lp
    if not np.isfinite(lnprob):
        lnprob = -np.inf
    return lnprob

def transform_uniform(x, a, b):
    return a + (b - a) * x

def transform_log_uniform(x, a, b):
    return a*(b/a)**x

def transform_normal(x, mu, sigma):
    return norm.ppf(x, loc=mu, scale=sigma)

def ptform(theta, prior1, prior2, priortype):
    """Compute the prior transform for nested sampling.

    Parameters
    ----------
    theta: ndarray
        The current estimate of the fitted parameters
    prior1: ndarray
        The lower-bound for uniform/log uniform priors, or mean for normal priors.
    prior2: ndarray
        The upper-bound for uniform/log uniform priors, or std. dev. for normal priors.
    priortype: ndarray
        Keywords indicating the type of prior for each free parameter.
    freenames:
        The names of the fitted parameters.

    Returns
    -------
    p: ndarray
        The prior transform.

    Notes
    -----
    History:

    - February 23-25, 2022 Megan Mansfield
        Added log-uniform and Gaussian priors.    
    """
    p = np.zeros_like(theta)
    n = len(theta)
    for i in range(n):
        if priortype[i]=='U':
            p[i] = transform_uniform(theta[i], prior1[i], prior2[i])
        elif priortype[i]=='LU':
            p[i] = transform_log_uniform(theta[i], prior1[i], prior2[i])
        elif priortype[i]=='N':
            p[i] = transform_normal(theta[i], prior1[i], prior2[i])
        else:
            raise ValueError("PriorType must be 'U', 'LU', or 'N'")
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
    chi2 = np.sum((residuals / lc.unc_fit) ** 2)
    chi2red = chi2 / (len(lc.flux) - len(freenames))

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
        bindata = np.ma.zeros(nbins[i], dtype=float)
        # bin data
        # ADDED INTEGER CONVERSION, mh 01/21/12
        for j in range(nbins[i]):
            bindata[j] = np.ma.mean(data[j * binsz[i]:(j + 1) * binsz[i]])
        # get rms
        rms[i] = np.sqrt(np.ma.mean(bindata ** 2))
        rmserr[i] = rms[i] / np.sqrt(2. * nbins[i])
    # expected for white noise (WINN 2008, PONT 2006)
    stderr = (np.ma.std(data) / np.sqrt(binsz)) * np.sqrt(nbins / (nbins - 1.))
    if isrmserr is True:
        return rms, stderr, binsz, rmserr
    else:
        return rms, stderr, binsz
