import os
import eureka.lib.plots
import eureka.S1_detector_processing.s1_process as s1
import eureka.S2_calibrations.s2_calibrate as s2
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4cal_StellarSpectra.s4cal_StellarSpec as s4cal

# eventlabel = 'imaging_template'
# eventlabel = 'miri_lrs_template'
eventlabel = 'nirspec_fs_template'
# eventlabel = 'nircam_wfss_template'
ecf_path = '.'+os.sep

if __name__ == '__main__':
    # To skip one or more stages that were already run,
    # just comment them out below

    meta = s1.rampfitJWST(eventlabel, ecf_path=ecf_path)

    meta = s2.calibrateJWST(eventlabel, ecf_path=ecf_path)

    spec, meta = s3.reduce(eventlabel, ecf_path=ecf_path)

    meta, spec, ds = s4cal.medianCalSpec(eventlabel, ecf_path=ecf_path)
