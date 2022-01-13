import sys
sys.path.append('../../')
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4
import eureka.S5_lightcurve_fitting.s5_fit as s5

eventlabel = 'template'

s3_meta = s3.reduceJWST(eventlabel)

s4_meta = s4.lcJWST(eventlabel, s3_meta=s3_meta)

s5_meta, lc_model = s5.fitJWST(eventlabel, s4_meta=s4_meta)
