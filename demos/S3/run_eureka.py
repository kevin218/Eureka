import sys
sys.path.append('../..')
from importlib import reload
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4

eventlabel = 'template'

s3_meta = s3.reduceJWST(eventlabel)

# s4_meta = s4.lcJWST(s3_meta.eventlabel, s3_meta.workdir, meta=s3_meta)
