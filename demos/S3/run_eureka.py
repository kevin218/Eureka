
import sys, os, time
#sys.path.append('../..')
#sys.path.append('/Users/stevekb1/Documents/code/Eureka/Eureka')
sys.path.append('/home/zieba/Desktop/Projects/Open_source/Eureka')

from importlib import reload
import eureka.S3_data_reduction.s3_reduce as s3
reload(s3)

eventlabel = 'wasp43b'

ev = s3.reduceJWST(eventlabel, isplots=1)
