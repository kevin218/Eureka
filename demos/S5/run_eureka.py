import sys
sys.path.append('../..')
import eureka.S5_lightcurve_fitting.s5_fit as s5

eventlabel = 'template'

s5_meta, lc_model = s5.fitJWST(eventlabel)
