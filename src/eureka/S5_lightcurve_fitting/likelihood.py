import numpy as np
import copy
from scipy.stats import norm

from ..lib.split_channels import get_trim


def update_uncertainty(theta, nints, unc, freenames, nchannel_fitted):
    """Compute updated uncertainty array when inflating errors during fitting.

    Parameters
    ----------
    theta : np.array
        The
    nints : list
        The number of integrations for each channel being fit at once.
    unc : np.array
        The initial guessed uncertainty array.
    freenames : list
        The names of the fitted parameters.
    nchannel_fitted : int
        The total number of fitted channels.

    Returns
    -------
    unc_fit : np.array
        The updated values for the uncertainty array.
    """
    # Make a copy so we don't edit in place
    unc_fit = copy.deepcopy(unc)

    if "scatter_mult" in freenames:
        for chan in range(nchannel_fitted):
            trim1, trim2 = get_trim(nints, chan)
            if chan > 0 and f'scatter_mult_ch{chan}' in freenames:
                loc = np.where(f'scatter_mult_ch{chan}' == np.array(freenames))
            else:
                loc = np.where('scatter_mult' == np.array(freenames))
            scatter_mult = theta[loc]
            unc_fit[trim1:trim2] *= scatter_mult
    elif "scatter_ppm" in freenames:
        for chan in range(nchannel_fitted):
            trim1, trim2 = get_trim(nints, chan)
            if chan > 0 and f'scatter_ppm_ch{chan}' in freenames:
                loc = np.where(f'scatter_ppm_ch{chan}' == np.array(freenames))
            else:
                loc = np.where('scatter_ppm' == np.array(freenames))
            scatter_ppm = theta[loc]
            unc_fit[trim1:trim2] = scatter_ppm*1e-6

    return unc_fit


def ln_like(theta, lc, model, freenames):
    """Compute the log-likelihood.

    Parameters
    ----------
    theta : ndarray
        The current estimate of the fitted parameters.
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit.
    freenames : iterable
        The names of the fitted parameters.

    Returns
    -------
    ln_like_val : ndarray
        The log-likelihood value at the position theta.

    Notes
    -----
    History:

    - December 29-30, 2021 Taylor Bell
        Moved code to separate file, added documentation.
    - January 22, 2022 Megan Mansfield
        Adding ability to do a single shared fit across all channels
    - February, 2022 Eva-Maria Ahrer
        Adding GP likelihood
    """
    model.update(theta)
    model_lc = model.eval()
    lc.unc_fit = update_uncertainty(theta, lc.nints, lc.unc, freenames,
                                    lc.nchannel_fitted)

    if model.GP:
        ln_like_val = 0
        for m in model.components:
            if m.modeltype == 'GP':
                ln_like_val += m.loglikelihood(model_lc)
    else:
        residuals = (lc.flux - model_lc)
        ln_like_val = (-0.5*(np.ma.sum((residuals/lc.unc_fit)**2
                       + np.ma.log(2.0 * np.pi * (lc.unc_fit) ** 2))))
    return ln_like_val


def lnprior(theta, prior1, prior2, priortype, freenames):
    """Compute the log-prior.

    Parameters
    ----------
    theta : ndarray
        The current estimate of the fitted parameters.
    prior1 : ndarray
        The lower-bound for uniform/log uniform priors, or mean for
        normal priors.
    prior2 : ndarray
        The upper-bound for uniform/log uniform priors, or std. dev. for
        normal priors.
    priortype : ndarray
        Keywords indicating the type of prior for each free parameter.
    freenames : iterable
        The names of the fitted parameters.

    Returns
    -------
    lnprior_prob : ndarray
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
    for i in range(len(theta)):
        if (priortype[i] == 'U' and np.logical_or(theta[i] < prior1[i],
                                                  theta[i] > prior2[i])):
            return -np.inf
        elif (priortype[i] == 'LU' and
              np.logical_or(np.log(theta[i]) < prior1[i],
                            np.log(theta[i]) > prior2[i])):
            return -np.inf
        elif priortype[i] == 'N':
            lnprior_prob -= (0.5*(np.sum(((theta[i]-prior1[i])/prior2[i])**2
                             + np.log(2.0*np.pi*(prior2[i])**2))))
        elif priortype[i] not in ['U', 'LU', 'N']:
            raise ValueError("PriorType must be 'U', 'LU', or 'N'")

        # Force scatter_ppm and scatter_mult to be positive
        if "scatter_ppm" in freenames:
            ind = [i for i in np.arange(len(freenames))
                   if freenames[i][0:11] == "scatter_ppm"]
            for chan in range(len(ind)):
                if theta[ind[chan]] <= 0:
                    # Force scatter_ppm to be > 0
                    return -np.inf
        elif "scatter_mult" in freenames:
            ind = [i for i in np.arange(len(freenames))
                   if freenames[i][0:12] == "scatter_mult"]
            for chan in range(len(ind)):
                if theta[ind[chan]] <= 0:
                    # Force scatter_ppm to be > 0
                    return -np.inf

    return lnprior_prob


def lnprob(theta, lc, model, prior1, prior2, priortype, freenames):
    """Compute the log-probability.

    Parameters
    ----------
    theta : ndarray
        The current estimate of the fitted parameters.
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object.
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit.
    prior1 : ndarray
        The lower-bound for uniform/log uniform priors, or mean for
        normal priors.
    prior2 : ndarray
        The upper-bound for uniform/log uniform priors, or std. dev. for
        normal priors.
    priortype : ndarray
        Keywords indicating the type of prior for each free parameter.
    freenames : iterable
        The names of the fitted parameters.

    Returns
    -------
    ln_prob_val : ndarray
        The log-probability value at the position theta.

    Notes
    -----
    History:

    - December 29-30, 2021 Taylor Bell
        Moved code to separate file, added documentation.
    - February 23-25, 2022 Megan Mansfield
        Added log-uniform and Gaussian priors.
    """
    lp = lnprior(theta, prior1, prior2, priortype, freenames)
    if not np.isfinite(lp):
        return -np.inf
    ln_like_val = ln_like(theta, lc, model, freenames)
    lnprob = ln_like_val + lp
    if not np.isfinite(lnprob):
        return -np.inf
    else:
        return lnprob


def transform_uniform(x, a, b):
    """The uniform prior transform function needed for dynesty.

    Parameters
    ----------
    x : float
        The position at which to calculate the prior.
    a : float
        The lower limit.
    b : float
        The upper limit.

    Returns
    -------
    float
        The uniform prior transform.
    """
    return a + (b - a) * x


def transform_log_uniform(x, a, b):
    """The log-uniform prior transform function needed for dynesty.

    Parameters
    ----------
    x : float
        The position at which to calculate the prior.
    a : float
        The log lower limit.
    b : float
        The log upper limit.

    Returns
    -------
    float
        The log-uniform prior transform.
    """
    return a*(b/a)**x


def transform_normal(x, mu, sigma):
    """The normal prior transform function needed for dynesty.

    Parameters
    ----------
    x : float
        The position at which to calculate the prior.
    mu : float
        The prior mean.
    sigma : float
        The prior standard deviation.

    Returns
    -------
    float
        The normal prior transform.
    """
    return norm.ppf(x, loc=mu, scale=sigma)


def ptform(theta, prior1, prior2, priortype):
    """Compute the prior transform for nested sampling.

    Parameters
    ----------
    theta : ndarray
        The current estimate of the fitted parameters.
    prior1 : ndarray
        The lower-bound for uniform/log uniform priors, or mean for
        normal priors.
    prior2 : ndarray
        The upper-bound for uniform/log uniform priors, or std. dev. for
        normal priors.
    priortype : ndarray
        Keywords indicating the type of prior for each free parameter.

    Returns
    -------
    p : ndarray
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
        if priortype[i] == 'U':
            p[i] = transform_uniform(theta[i], prior1[i], prior2[i])
        elif priortype[i] == 'LU':
            p[i] = transform_log_uniform(theta[i], prior1[i], prior2[i])
        elif priortype[i] == 'N':
            p[i] = transform_normal(theta[i], prior1[i], prior2[i])
        else:
            raise ValueError("PriorType must be 'U', 'LU', or 'N'")
    return p


def computeRedChiSq(lc, log, model, meta, freenames):
    """Compute the reduced chi-squared value.

    Parameters
    ----------
    lc : eureka.S5_lightcurve_fitting.lightcurve.LightCurve
        The lightcurve data object
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    model : eureka.S5_lightcurve_fitting.models.CompositeModel
        The composite model to fit
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    freenames : iterable
        The names of the fitted parameters.

    Returns
    -------
    chi2red : float
        The reduced chi-squared value.

    Notes
    -----
    History:

    - December 29-30, 2021 Taylor Bell
        Moved code to separate file, added documentation.
    - February, 2022 Eva-Maria Ahrer
        Added GP functionality
    """
    model_lc = model.eval(incl_GP=True)
    residuals = (lc.flux - model_lc)
    chi2 = np.ma.sum((residuals / lc.unc_fit) ** 2)
    chi2red = chi2 / (np.sum(~np.ma.getmaskarray(lc.flux)) - len(freenames))

    log.writelog(f'Reduced Chi-squared: {chi2red}', mute=(not meta.verbose))

    return chi2red


def computeRMS(data, maxnbins=None, binstep=1, isrmserr=False):
    """Compute the root-mean-squared and standard error for various bin sizes.

    Parameters
    ----------
    data : ndarray
        The residuals after fitting.
    maxnbins : int; optional
        The maximum number of bins. Use None to default to 10 points per bin.
    binstep : int; optional
        Bin step size. Defaults to 1.
    isrmserr : bool
        True if return rmserr, else False. Defaults to False.

    Returns
    -------
    rms : ndarray
        The RMS for each bin size.
    stderr : ndarray
        The standard error for each bin size.
    binsz : ndarray
        The different bin sizes.
    rmserr : ndarray; optional
        The uncertainty in the RMS. Only returned if isrmserr==True.

    Notes
    -----
    History:

    - December 29-30, 2021 Taylor Bell
        Moved code to separate file, added documentation.
    """
    data = np.ma.masked_invalid(np.ma.copy(data))

    # bin data into multiple bin sizes
    npts = data.size
    if maxnbins is None:
        maxnbins = npts / 10.
    binsz = np.arange(1, maxnbins + binstep, step=binstep, dtype=int)
    nbins = np.zeros(binsz.size, dtype=int)
    rms = np.ma.zeros(binsz.size)
    rmserr = np.ma.zeros(binsz.size)
    for i in range(binsz.size):
        nbins[i] = int(np.floor(data.size / binsz[i]))
        bindata = np.ma.zeros(nbins[i], dtype=float)
        # bin data
        # ADDED INTEGER CONVERSION, mh 01/21/12
        for j in range(nbins[i]):
            bindata[j] = np.ma.mean(data[j * binsz[i]:(j + 1) * binsz[i]])
        # get rms
        rms[i] = np.ma.sqrt(np.ma.mean(bindata ** 2))
        rmserr[i] = rms[i] / np.sqrt(2. * nbins[i])
    # expected for white noise (WINN 2008, PONT 2006)
    stderr = (np.ma.std(data) / np.sqrt(binsz)) * np.sqrt(nbins / (nbins - 1.))
    if isrmserr is True:
        return rms, stderr, binsz, rmserr
    else:
        return rms, stderr, binsz
