import numpy as np

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
