import sys
sys.path.append('../..')
from eureka.S2_calibrations.s2_calibrate import EurekaS2Pipeline
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4

# eventlabel = 'miri_lrs_template'
# eventlabel = 'nirspec_fs_template'
eventlabel = 'nircam_wfss_template'

s2 = EurekaS2Pipeline()
s2_meta = s2.run_eurekaS2(eventlabel)

s3_meta = s3.reduceJWST(eventlabel, s2_meta=s2_meta)

s4_meta = s4.lcJWST(eventlabel, s3_meta=s3_meta)
