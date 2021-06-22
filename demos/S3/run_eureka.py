import sys, os, time
#sys.path.append('../..')
#sys.path.append('/Users/stevekb1/Documents/code/Eureka/Eureka')
#sys.path.append('/Users/kreidberg/Desktop/Projects/OpenSource/Eureka/')
sys.path.append('/home/zieba/Desktop/Projects/Open_source/Eureka')
from importlib import reload
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4

eventlabel = 'wasp43b'

reload(s3)
ev3 = s3.reduceJWST(eventlabel)

reload(s4)
ev4 = s4.lcJWST(ev3.eventlabel, ev3.workdir, meta=ev3)
