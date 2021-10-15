from ....S3_data_reduction import s3_reduce as s3
from ....S4_generate_lightcurves import s4_genLC as s4

eventlabel = 'template'

s3_meta = s3.reduceJWST(eventlabel)

s4_meta = s4.lcJWST(eventlabel, s3_meta=s3_meta)
