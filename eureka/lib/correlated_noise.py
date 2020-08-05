

import numpy as np


# COMPUTE ROOT-MEAN-SQUARE AND STANDARD ERROR OF DATA FOR VARIOUS BIN SIZES
def computeRMS(data, maxnbins=None, binstep=1, isrmserr=False):
    #data    = fit.normresiduals
    #maxnbin = maximum # of bins
    #binstep = Bin step size
    
    # bin data into multiple bin sizes
    npts    = data.size
    if maxnbins is None:
        maxnbins = npts/10.
    binsz   = np.arange(1, maxnbins+binstep, step=binstep, dtype=int)
    nbins   = np.zeros(binsz.size, dtype=int)
    rms     = np.zeros(binsz.size)
    rmserr  = np.zeros(binsz.size)
    for i in range(binsz.size):
        nbins[i] = int(np.floor(data.size/binsz[i]))
        bindata   = np.zeros(nbins[i], dtype=float)
        # bin data
        # ADDED INTEGER CONVERSION, mh 01/21/12
        for j in range(nbins[i]):
            bindata[j] = data[j*binsz[i]:(j+1)*binsz[i]].mean()
        # get rms
        rms[i]    = np.sqrt(np.mean(bindata**2))
        rmserr[i] = rms[i]/np.sqrt(2.*nbins[i])
    # expected for white noise (WINN 2008, PONT 2006)
    stderr = (data.std()/np.sqrt(binsz))*np.sqrt(nbins/(nbins - 1.))
    if isrmserr is True:
        return rms, stderr, binsz, rmserr
    else:
        return rms, stderr, binsz

# Compute standard error
def computeStdErr(datastd, datasize, binsz):
    #datastd  = fit.normresiduals.std()
    #datasize = fit.normresiduals.size
    #binsz    = array of bins
    
    nbins   = np.zeros(binsz.size, dtype=int)
    for i in range(binsz.size):
        nbins[i] = int(np.floor(datasize/binsz[i]))
    stderr = (datastd/np.sqrt(binsz))*np.sqrt(nbins/(nbins - 1.))
    return stderr

'''
    # plot results
    plt.figure(1)
    plt.clf()
    plt.axes([0.12, 0.12, 0.82, 0.82])
    plt.loglog(binsz, rms, color='black', lw=1.5, label='RMS')    # our noise
    plt.loglog(binsz, stderr, color='red', ls='-', lw=2, label='Std. Err.') # expected noise
    plt.xlim(0, binsz[-1]*2)
    plt.ylim(rms[-1]/2., rms[0]*2.)
    plt.xlabel("Bin Size", fontsize=14)
    plt.ylabel("RMS", fontsize=14)
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.legend()
    plt.text(binsz[-1], rms[0], "Channel {0}".format(chan+1), fontsize=20,
             ha='right')
    plt.savefig("wa012bs{0}1-noisecorr.ps".format(chan+1))
    plt.title("Channel {0} Noise Correlation".format(chan+1), fontsize=25)
    plt.savefig("wa012bs{0}1-noisecorr.png".format(chan+1))
'''
