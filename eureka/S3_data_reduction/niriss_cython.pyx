import numpy as np

__all__ = ['build_gaussian_images', 'build_moffat_images']

def build_gaussian_images(data, A, B, sig, mu1, mu2, return_together=True):

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
