import sys
sys.path.append('../..')
import eureka.S6_spectra_plotting as s6

eventlabel = 'template'

if __name__ == '__main__':
	s6_meta = s6.fitJWST(eventlabel)
