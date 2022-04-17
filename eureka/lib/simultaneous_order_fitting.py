import numpy as np
import scipy.optimize as so


import pyximport
pyximport.install()
from . import niriss_profiles as profiles


__all__ = ['fit_orders_fast', 'fit_orders']


def fit_orders(data, tab):
    """
    Creates a 2D image optimized to fit the data. Currently
    runs with a Gaussian profile, but will look into other 
    more realistic profiles at some point. This routine  
    is a bit slow, but fortunately, you only need to run it
    once per observations.

    Parameters
    ----------
    data : object
    which_table : int, optional
       Sets with table of initial y-positions for the
       orders to use. Default is 2.  

    Returns
    -------
    ord1 : np.ndarray
    ord2 : np.ndarray
    """
    print("Go grab some food. This routing could take up to 30 minutes.")

    def construct_guesses(A, B, sig, length=10):
        As   = np.linspace(A[0],   A[1],   length)  # amplitude of gaussian for first order
        Bs   = np.linspace(B[0],   B[1],   length)  # amplitude of gaussian for second order
        sigs = np.linspace(sig[0], sig[1], length)  # std of gaussian profile  
        combos = np.array(list(itertools.product(*[As,Bs,sigs]))) # generates all possible combos
        return combos

    pos1, pos2 = tab['order_1'], tab['order_2']

    # Good initial guesses 
    combos = construct_guesses([0.1,30], [0.1,30], [1,40])

    # generates length x length x length number of images and fits to the data
    img1, sigout1 = profiles.build_gaussian_images(data.median,
                                                   combos[:,0], combos[:,1],
                                                   combos[:,2],
                                                   pos1, pos2)
    
    # Iterates on a smaller region around the best guess
    best_guess = combos[np.argmin(sigout1)]
    combos = construct_guesses( [best_guess[0]-0.5, best_guess[0]+0.5],
                                [best_guess[1]-0.5, best_guess[1]+0.5],
                                [best_guess[2]-0.5, best_guess[2]+0.5] )

    # generates length x length x length number of images centered around the previous
    #   guess to optimize the image fit
    img2, sigout2 = profiles.build_gaussian_images(data.median,
                                                   combos[:,0], combos[:,1],
                                                   combos[:,2],
                                                   pos1, pos2)

    # creates a 2D image for the first and second orders with the best-fit gaussian
    #    profiles   
    final_guess = combos[np.argmin(sigout2)]
    ord1, ord2, _ = profiles.build_gaussian_images(data.median,
                                                        [final_guess[0]],
                                                        [final_guess[1]],
                                                        [final_guess[2]],
                                                        pos1, pos2,
                                                        return_together=False)
    return ord1[0], ord2[0]



def fit_orders_fast(data, tab, profile='gaussian'):
    """
    A faster method to fit a 2D mask to the NIRISS data.
    Very similar to `fit_orders`, but works with
    `scipy.optimize.leastsq`.  

    Parameters
    ----------
    data : object
    which_table : int, optional
       Sets with table of initial y-positions for the 
       orders to use. Default is 2.

    Returns
    -------
    ord1 : np.ndarray
    ord2 : np.ndarray
    """

    def residuals(params, data, y1_pos, y2_pos):
        """ Calcualtes residuals for best-fit profile. """
        nonlocal profile

        A, B, sig1 = params
        # Produce the model:
        if profile.lower() == 'gaussian':
            model,_ = profiles.build_gaussian_images(data, [A], [B], [sig1], y1_pos, y2_pos)
        elif profile.lower() == 'moffat':
            model,_ = profiles.build_moffat_images(data, [A], [B], [sig1], y1_pos, y2_pos)
            # Calculate residuals: 
        res = (model[0] - data)
        return res.flatten()

    pos1, pos2 = tab['order_1'], tab['order_2']
    
     # fits the mask
    if profile.lower()=='gaussian':
        x0=[2,3,30]
    elif profile.lower()=='moffat':
        x0=[]
    else:
        print('profile shape not implemented. using gaussian')
        profile='gaussian'
        x0=[2,3,30]

    results = so.least_squares( residuals,
                                x0=np.array(x0),
                                args=(data.median, pos1, pos2),
                                xtol=1e-11, ftol=1e-11, max_nfev=1e3
                                )
    
    # creates the final mask
    if profile.lower() == 'gaussian':
        out_img1,out_img2,_= profiles.build_gaussian_images(data.median,
                                                            results.x[0:1],
                                                            results.x[1:2],
                                                            results.x[2:3],
                                                            pos1,
                                                            pos2,
                                                            return_together=False)
    return out_img1[0], out_img2[0]
