import numpy as np


def gelmanrubin(chain, nchains):
    """Perform convergence test of Gelman & Rubin (1992) on a MCMC chain.

    Parameters
    ----------
    chain : ndarray
        A vector of parameter samples from a MCMC routine.
    nchains : scalar
        The number of chains to split the original chain into. The
        length of `chain` WILL BE MODIFIED if NOT evenly divisible by
        `nchains`.

    Returns
    -------
    psrf : scalar
        The potential scale reduction factor of the chain. If the
        chain has converged, this should be close to unity. If it is
        much greater than 1, the chain has not converged and requires
        more samples.
    """
    # Shorten chain by 'remainder' so that chain length is evenly divisible
    # by nchains
    remainder = int(chain.size % nchains)
    if remainder != 0:
        chain = chain[:-remainder]

    # get length of each chain and reshape
    nchains = int(nchains)
    chainlen = int(chain.size/nchains)
    chains = chain.reshape(nchains, chainlen)

    # calculate W (within-chain variance)
    W = np.mean(chains.var(axis=1))

    # calculate B (between-chain variance)
    means = chains.mean(axis=1)
    mmean = means.mean()
    B = (chainlen/(nchains - 1.))*np.sum((means - mmean)**2)

    # calculate V (posterior marginal variance)
    V = W*((chainlen - 1.)/chainlen) + B*((nchains + 1.)/(chainlen*nchains))

    # calculate potential scale reduction factor (PSRF)
    psrf = np.sqrt(V/W)

    return psrf


def convergetest(pars, nchains):
    """Driver routine for gelmanrubin.

    Perform convergence test of Gelman & Rubin (1992) on a MCMC chain.

    Parameters
    ----------
    pars : ndarray
        A 2D array containing a separate parameter MCMC chain per row.
    nchains : scalar
        The number of chains to split the original chain into. The
        length of each chain MUST be evenly divisible by `nchains`.

    Returns
    -------
    psrf : ndarray
        The potential scale reduction factors of the chain.  If the
        chain has converged, each value should be close to unity.  If
        they are much greater than 1, the chain has not converged and
        requires more samples.  The order of psrfs in this vector are
        in the order of the free parameters.
    meanpsrf : scalar
        The mean of `psrf`.  This should be ~1 if your chain has
        converged.

    Examples
    --------
    Consider four MCMC runs that has already been loaded.  The individual fits
    are located in the `fit` list.  These are for channels 1-4.

    .. highlight:: python
    .. code-block:: python

        >>> import gelmanrubin
        >>> import numpy as np
        >>> # channels 1/3 free parameters
        >>> ch13pars = np.concatenate((fit[0].allparams[fit[0].freepars],
        >>>                            fit[2].allparams[fit[2].freepars]))

        >>> # channels 2/4 free parameters
        >>> ch24pars = np.concatenate((fit[1].allparams[fit[1].freepars],
        >>>                            fit[3].allparams[fit[3].freepars]))

        >>> # number of chains to split into
        >>> nchains = 4

        >>> # test for convergence
        >>> ch13conv  = gelmanrubin.convergetest(ch13pars, nchains)
        >>> ch24conv  = gelmanrubin.convergetest(ch24pars, nchains)

        >>> # show results
        >>> print(ch13conv)
        (array([1.02254252, 1.00974035, 1.04838778, 1.0017869 , 1.7869707,
                2.15683239, 1.00506215, 1.00235165, 1.06784124, 1.04075207,
                1.01452032]), 1.1960716427734874)
        >>> print(ch24conv)
        (array([1.01392515, 1.00578357, 1.03285576, 1.13138702, 1.0001787,
                3.52118005, 1.10592542, 1.05514509, 1.00101459]),
                1.3185994837687156)
    """
    # allocate placeholder for results
    nfpar = pars.shape[0]
    psrf = np.zeros(nfpar)
    # calculate psrf for each parameter
    for i in range(nfpar):
        chain = pars[i].flatten()
        # chain = pars[i]
        psrf[i] = gelmanrubin(chain, nchains)

    return psrf, psrf.mean()
