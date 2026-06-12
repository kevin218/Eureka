import numpy as np


def build_image_models(data, A, B, sig, mu1, mu2, return_together=True):

    y = np.ma.transpose(np.full((len(A), data.shape[-1], data.shape[0]),
                                np.arange(0, data.shape[0], 1)),
                        axes=(0, 2, 1))

    sig = np.full(y.T.shape, sig).T

    A = np.full(y.T.shape, A).T
    exp = np.ma.exp(-(y-mu1)**2/(2.0*sig**2))
    gauss = 1.0/(2.0*np.pi*np.ma.sqrt(sig))*exp
    f1x = mu1*gauss*A

    B = np.full(y.T.shape, B).T
    exp = np.ma.exp(-(y-mu2)**2/(2.0*sig**2))
    gauss = 1.0/(2.0*np.pi*np.ma.sqrt(sig))*exp
    f2x = mu2*gauss*B

    model = f1x+f2x
    sigma = np.ma.sum(np.ma.sqrt((model-data)**2.0), axis=(1, 2))

    if return_together:
        return model, sigma
    else:
        return f1x, f2x, sigma
