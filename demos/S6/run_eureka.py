import sys
sys.path.insert(0, '../../')
import eureka.S6_planet_spectra.s6_spectra as s6

eventlabel = 'template'

if __name__ == '__main__':
	s6_meta = s6.plot_spectra(eventlabel)
