import numpy as np

__all__ = ['build_gaussian_images', 'build_moffat_images']

def build_gaussian_images(data, A, B, sig, mu1, mu2, return_together=True):
    """
    Builds the two order profiles simultaneously and approximates as a
    Gaussian profile.

    Parameters
    ----------
    data : np.ndarray
       The images.
    A : float
       Sets the scaling of the profile for the first order.
    B : float
       Sets the scaling of the profile for the second order.
    sig : float
       Sets the width of the profile for the orders.
    mu1 : float
       Sets the mean of the profile for the first order.
    mu2 : float
       Sets the mean of the profile for the second order.
    return_together : bool, optional
       Allows users to return a combined profile for both orders, or individual
       profiles. Default is `True`.

    Returns
    -------
    model : np.ndarray
       Best-fit profile with both the first and second orders.
    sigma : np.ndarray
       Describes how well-fit the profile is to the data.
    f1x : np.ndarray
       Best-fit profile with only the first order. Returns if
       `return_together = False`.
    f2x : np.ndarray
       Best-fit profile with only the second order. Returns if
       `return_together = False`.
    """

    y = np.transpose(np.full((len(A), data.shape[-1], data.shape[0]),
                         np.arange(0,data.shape[0],1)), axes=(0,2,1))

    sig = np.full(y.T.shape,sig).T

    A = np.full(y.T.shape, A).T
    exp = np.exp(-(y-mu1)**2/(2.0 * sig**2))
    gauss = 1.0/(2.0*np.pi*np.sqrt(sig)) * exp
    f1x = mu1*gauss*A

    B = np.full(y.T.shape, B).T
    exp = np.exp(-(y-mu2)**2/(2.0 * sig**2))
    gauss = 1.0/(2.0*np.pi*np.sqrt(sig)) * exp
    f2x = mu2*gauss*B

    model = f1x+f2x
    sigma = np.nansum( np.sqrt( (model-data)**2.0), axis=(1,2))

    if return_together:
       return model, sigma
    else:
       return f1x, f2x, sigma


def build_moffat_images(data, A, alpha, gamma, mu1, mu2, return_together=True):
    """
    Builds the two order profiles simultaneously and approximates as a
    Moffat profile.

    Parameters
    ----------
    data : np.ndarray
       The images.
    A : float
       Sets the scaling of the profile for the first order.
    alpha : float
       Sets the width of the profile for the orders.
    gamma : float
       Sets the boxiness of the profile for the orders.
    mu1 : float
       Sets the mean of the profile for the first order.
    mu2 : float
       Sets the mean of the profile for the second order.
    return_together : bool, optional
       Allows users to return a combined profile for both orders, or individual
       profiles. Default is `True`.

    Returns
    -------
    m : np.ndarray
       Describes how well-fit the profile is to the data.
    m1 : np.ndarray
       Best-fit profile with only the first order. Returns if
       `return_together = False`.
    m2 : np.ndarray
       Best-fit profile with only the second order. Returns if
       `return_together = False`.
    """
    def moffat(x, A, alpha, gamma):
       frac = 1 + (y - x)**2.0 / gamma**2.0
       return A * frac**(-alpha)

    y = np.transpose(np.full((len(alpha), data.shape[-1], data.shape[0]),
                         np.arange(0,data.shape[0],1)), axes=(0,2,1))

    A = np.full(y.T.shape, A).T
    alpha = np.full(y.T.shape, alpha).T
    gamma = np.full(y.T.shape, gamma).T

    m1 = moffat(mu1, A, alpha, gamma)
    m2 = moffat(mu2, A, alpha, gamma)

    if return_together:
       return m1 + m2
    else:
       return m1, m2
