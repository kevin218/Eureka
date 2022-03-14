import sys
sys.path.append('../../')
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4
import eureka.S5_lightcurve_fitting.s5_fit as s5

eventlabel = 'wfc3'
ecf_path = './'

if __name__ == '__main__':
	s3_meta = s3.reduceJWST(eventlabel, ecf_path=ecf_path)

	s4_meta = s4.lcJWST(eventlabel, ecf_path=ecf_path, s3_meta=s3_meta)

	s5_meta = s5.fitJWST(eventlabel, ecf_path=ecf_path, s4_meta=s4_meta)
