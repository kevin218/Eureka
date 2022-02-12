import sys
sys.path.append('../..')
import eureka.S5_lightcurve_fitting.s5_fit as s5

eventlabel = 'template'

if __name__ == '__main__':
	s5_meta = s5.fitJWST(eventlabel)
