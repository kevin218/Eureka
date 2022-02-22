import numpy as np

def flatfield3(c, P_hat, etc=[]):

    #print("length of c: " + str(len(c)))
    midpt = P_hat.shape[1]//2    #integer division
    del_S = np.sum((np.reshape(P_hat[:,midpt-1:midpt+2,midpt-1:midpt+2], (len(P_hat), 9))*c[:9]), axis=1) + c[9]
    
    return del_S/del_S.mean()
