import sys
sys.path.append('../..')
import eureka.S5_lightcurve_fitting.s5_fit as s5

meta = '/Users/megan/Documents/Code/Eureka/Eureka/demos/S3/S3_2021-10-11_15-10-58_wasp43b/S4_2021-10-11_15-23-10_6chan/S4_wasp43b_Meta_Save.dat'
eventlabel = 'template'

ev5 = s5.fitJWST(eventlabel, s4_meta=meta)
