import numpy as np
from scipy.stats import norm

def ln_like(theta, lc, model, pmin, pmax, freenames):
    # params[ifreepars] = freepars
    ilow = np.where(theta < pmin)
    ihi = np.where(theta > pmax)
    theta[ilow] = pmin[ilow]
    theta[ihi] = pmax[ihi]
    model.update(theta, freenames)
    model_lc = model.eval()
    residuals = (lc.flux - model_lc) #/ lc.unc
    ln_like_val = (-0.5 * (np.sum((residuals / lc.unc) ** 2+ np.log(2.0 * np.pi * (lc.unc) ** 2))))
    if len(ilow[0]) + len(ihi[0]) > 0: ln_like_val = -np.inf
    return ln_like_val

def lnprior(theta, pmin, pmax):
    lnprior_prob = 0.
    n = len(theta)
    for i in range(n):
        if np.logical_or(theta[i] < pmin[i],
                                theta[i] > pmax[i]): lnprior_prob += - np.inf
    return lnprior_prob

def lnprob(theta, lc, model, pmin, pmax, freenames):
    ln_like_val = ln_like(theta, lc, model, pmin, pmax, freenames)
    lp = lnprior(theta, pmin, pmax)
    return ln_like_val + lp

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
	model_lc = model.eval()
	residuals = (lc.flux - model_lc) #/ lc.unc
	chi2 = np.sum((residuals / lc.unc) ** 2)
	chi2red = chi2 / (len(lc.unc) - len(freenames))
	
	if meta.run_verbose:
		print('red. Chi2: ', chi2red)

	return chi2red

# COMPUTE ROOT-MEAN-SQUARE AND STANDARD ERROR OF DATA FOR VARIOUS BIN SIZES
def computeRMS(data, maxnbins=None, binstep=1, isrmserr=False):
    # data    = fit.normresiduals
    # maxnbin = maximum # of bins
    # binstep = Bin step size

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
