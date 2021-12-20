import sys
sys.path.append('../..')
import eureka.S5_lightcurve_fitting.s5_fit as s5

eventlabel = 'template'

ev5 = s5.fitJWST(eventlabel)
