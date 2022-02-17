
import numpy as np
import pywt
'''
sigma = 2.
residuals = np.random.normal(0,sigma,5)

for wavelet in pywt.wavelist():
    etc=[wavelet]
    params = [np.var(residuals), 0.0, 1.]
    a = red.rednoise(params, residuals, etc)
    params = [np.var(residuals)*1.01, 0.0, 1.]
    low = np.exp(0.5*(a-red.rednoise(params, residuals, etc)))
    params = [np.var(residuals)*.99, 0.0, 1.]
    hi  = np.exp(0.5*(a-red.rednoise(params, residuals, etc)))
    if low <= 1 and hi <= 1 or wavelet =='db2':
        print(etc, low, hi)

(['bior1.1'], 0.76491703158920565, 0.9848434384441499)
(['db1'], 0.76491703158920565, 0.9848434384441499)
(['haar'], 0.76491703158920565, 0.9848434384441499)
(['rbio1.1'], 0.76491703158920565, 0.9848434384441499)

'''
def rednoise(params, residuals, etc):
    '''
    Estimate -2*log-likelihood (chi-squared) assuming data contains red noise.
    
    
    '''
    # Unpack white and red variances and correlated-noise factor gamma
    wvar  = params[0]**2
    rvar  = params[1]**2
    gamma = params[2]
    # 
    wavelet = etc[0]
    g = 1./(2*np.log(2))   #Assumes gamma = 1
    
    #FINDME: Pad residuals with zeros
    newresiduals = np.zeros(int(2**np.ceil(np.log2(len(residuals)))))
    newresiduals[:len(residuals)] = residuals
    
    # Perform wavelet decomposition
    dec = pywt.wavedec(newresiduals, wavelet)
    
    # Calculate log likelihood with scaling coefficients
    scoef = dec[0]
    svar  = rvar*2**(-gamma)*g + wvar
    #L1 = 1./(2*np.pi*svar)**(len(scoef)/2.)*np.exp(-0.5*np.sum(scoef**2,axis=0)/svar)
    #logL1 = np.log(L1)
    logL1 = np.sum(scoef**2)/svar + len(scoef)*np.log(2*np.pi*svar)
    
    # Calculate likelihood with wavelet coefficient
    wcoef = dec[1:]
    logL2 = 0
    for m in range(1,len(wcoef)):
        bigwvar  = rvar*2**(-gamma*(m+1)) + wvar
        #L2 *= 1./(2*np.pi*bigwvar)**(len(wcoef[m])/2.)*np.exp(-0.5*np.sum(wcoef[m]**2,axis=0)/bigwvar)
        #logL2 += np.log(L2)
        logL2 += np.sum(wcoef[m]**2)/bigwvar + len(wcoef[m])*np.log(2*np.pi*bigwvar)
    
    return logL1+logL2
