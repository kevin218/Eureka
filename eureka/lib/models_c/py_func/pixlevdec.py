import numpy as np

def pixlevdec(ci, P_hat):

    del_S = np.sum((np.reshape(P_hat, (len(P_hat), len(ci)))*ci), axis=1)
    
    return del_S
