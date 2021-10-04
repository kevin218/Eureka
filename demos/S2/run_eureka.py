import sys
sys.path.append('../..')
from eureka.S2_calibrations.s2_calibrate import EurekaS2Pipeline

eventlabel = 'miri_lrs_template'
# eventlabel = 'nirspec_fs_template'
# eventlabel = 'nircam_wfss_template'

s2 = EurekaS2Pipeline()
ev2 = s2.run_eurekaS2(eventlabel)
