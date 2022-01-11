import sys
sys.path.append('../..')
import eureka.S5_lightcurve_fitting.s5_fit as s5

eventlabel = 'template'

s5_meta = s5.fitJWST(eventlabel)
