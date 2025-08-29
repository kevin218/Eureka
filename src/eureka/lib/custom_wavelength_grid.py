"""
Created on Thu Feb 13 10:54:44 2025

@author: eganaa1
"""
import numpy as np
import glob


order1 = glob.glob('/Users/ashtar1/Data/JWST/NIRISS/HATP30b/SOSS/Stage4/order1_R36.txt')[0]

fil = np.genfromtxt(order1)

wmin = fil[:,0]
wmax = fil[:,1]

#print(wmin, wmax)

#find the min and max wavelengths in the S4 log files from the white light curve analysis

#
R = 36

def get_max_wavelength(wave_min, R):
    wave_max = wave_min*(1+2*R)/(2*R-1)
    return wave_max


# order1_wmin = 0.86
# order1_wmax = 2.80
order1_wmin = 0.855846774
order1_wmax = 2.823692369

wave_pairs = {'order1':[order1_wmin, order1_wmax]}


for key in wave_pairs.keys():
    new_wmax=0
    wmin, wmax = wave_pairs[key]
    wmins = []
    wmaxs = []
    print(key, wmin, wmax)
    while new_wmax <= wmax:
        new_wmax = get_max_wavelength(wmin, R)
        wmins.append(wmin)
        
        if new_wmax >= wmax:
            wmaxs.append(wmax)
            
        else:
            wmaxs.append(new_wmax)
            
            wmin = new_wmax
    print(wmins[:5])
    print(wmaxs[:5])        
    wmin_str = ''
    for i in wmins:
        wmin_str+= str(i) + ', ' 
        
    wmax_str = ''
    for i in wmaxs:
        wmax_str+= str(i) + ', ' 
           
    with open(f'{key}_R{R}.txt','w') as f:
        print(key)
        for m, x in zip(wmins, wmaxs):
            f.write(f'{m:.8f} {x:.8f}' + '\n')
            
    f.close()