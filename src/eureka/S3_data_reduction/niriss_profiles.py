"""
A library of custom weighted profiles
to fit to the NIRISS orders to complete
the optimal extraction of the data.
"""

import numpy as np
from scipy.special import gamma
from astropy.modeling.models import Gaussian1D, Moffat1D

__all__ = ['moffat_2poly_piecewise', 'moffat_1poly_piecewise',
           'gaussian_1poly_piecewise', 'gaussian_2poly_piecewise',
           'generalized_normal']


def generalized_normal(x, mu,alpha,beta,scale):
    """
    Generalized normal distribution.

    Parameters
    ----------
    x : np.ndarray
       X values to evaluate the distribution.
       over.
    mu : float
       Mean/center value of the distribution.
    alpha : float
       Sets the scale/standard deviation of the
       distribution.
    beta : float
       Sets the shape of the distribution.
       Beta > 2 becomes boxy. Beta < 2 becomes
       peaky. Beta = 2 is a normal Gaussian.
    scale : float
       A value to scale the distribution by.
    """
    gfunc = gamma(1.0/beta)
    frac  = beta / (2 * alpha * gfunc)
    exp   = (np.abs(x-mu)/alpha)**beta
    return frac * np.exp(-exp)*scale


def moffat_2poly_piecewise(args, x):
    """
    A piece-wise function consisting of 2 Moffat profiles
    connected with a 2D polynomial to mimic the bat-shaped
    profile of NIRISS.

    Parameters
    ----------
    args : np.ndarray
       A list or array of parameters for the fits.
    x : np.ndarray
       X values to evaluate the shape over.
    """
    x0,x1,mu1,alpha1,gamma1,f1,a,b,c,mu2,alpha2,gamma2,f2 = args
    model = np.piecewise(x, [((x<x0)),
                             ((x>=x0) & (x<x1))],
                         [lambda x: Moffat1D(x, x_0=mu1, alpha=alpha1,
                                             gamma=gamma1, amplitude=f1),
                          lambda x: a*x**2+b*x+c,
                          lambda x: Moffat1D(x, x_0=mu2, alpha=alpha2,
                                             gamma=gamma2,
                                             amplitude=f2)]
                        )
    return model


def moffat_1poly_piecewise(args, x):
    """
    A piece-wise function consisting of 2 Moffat profiles
    connected with a 1D polynomial to mimic the bat-shaped
    profile of NIRISS.

    Parameters
    ----------
    args : np.ndarray
       A list or array of parameters for the fits.
    x : np.ndarray
       X values to evaluate the shape over.
    """
    x0,x1,mu1,alpha1,gamma1,f1,m,b,mu2,alpha2,gamma2,f2 = args
    model = np.piecewise(x, [((x<x0)),
                             ((x>=x0) & (x<x1))],
                         [lambda x: Moffat1D(x, x_0=mu1, alpha=alpha1,
                                             gamma=gamma1, amplitude=f1),
                          lambda x: m*x+b,
                          lambda x: Moffat1D(x, x_0=mu2, alpha=alpha2,
                                             gamma=gamma2,
                                             amplitude=f2)]
                        )
    return model


def gaussian_1poly_piecewise(args, x):
    """
    A piece-wise function consisting of 2 generalized
    normal distribution profiles
    connected with a 1D polynomial to mimic the bat-shaped
    profile of NIRISS.

    Parameters
    ----------
    args : np.ndarray
       A list or array of parameters for the fits.
    x : np.ndarray
       X values to evaluate the shape over.
    """
    x0,x1,mu1,std1,beta1,scale1,m,b,mu2,std2,beta2,scale2 = args
    model = np.piecewise(x, [((x<x0)),
                             ((x>=x0) & (x<x1))],
                         [lambda x: generalized_normal(x, mu1,std1,beta1,scale1),
                          lambda x: m*x+b,
                          lambda x: generalized_normal(x, mu2,std2,beta2,scale2)]
                        )
    return model


def gaussian_2poly_piecewise(args, x):
    """
    A piece-wise function consisting of 2 generalized
    normal distribution profiles
    connected with a 2D polynomial to mimic the bat-shaped
    profile of NIRISS.

    Parameters
    ----------
    args : np.ndarray
       A list or array of parameters for the fits.
    x : np.ndarray
       X values to evaluate the shape over.
    """
    x0,x1,mu1,std1,beta1,scale1,a,b,c,mu2,std2,beta2,scale2 = args
    model = np.piecewise(x, [((x<x0)),
                             ((x>=x0) & (x<x1))],
                         [lambda x: generalized_normal(x, mu1,std1,beta1,scale1),
                          lambda x: a*x**2+b*x+c,
                          lambda x: generalized_normal(x, mu2,std2,beta2,scale2)]
                        )
    return model
