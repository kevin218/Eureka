import sys
sys.path.append('../../')

import eureka.S1_detector_processing.s1_process as s1

eventlabel = 'nirx_template'
# eventlabel = 'miri_template'

if __name__ == '__main__':
	s1_meta = s1.rampfitJWST(eventlabel)