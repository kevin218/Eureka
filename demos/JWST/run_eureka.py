import sys
sys.path.append('../../')
import eureka.S2_calibrations.s2_calibrate as s2
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4
import eureka.S5_lightcurve_fitting.s5_fit as s5

# eventlabel = 'imaging_template'
# eventlabel = 'miri_lrs_template'
# eventlabel = 'nirspec_fs_template'
eventlabel = 'nircam_wfss_template'

if __name__ == '__main__':
	s2_meta = s2.calibrateJWST(eventlabel)

	s3_meta = s3.reduceJWST(eventlabel, s2_meta=s2_meta)

	s4_meta = s4.lcJWST(eventlabel, s3_meta=s3_meta)

	s5_meta = s5.fitJWST(eventlabel, s4_meta=s4_meta)
