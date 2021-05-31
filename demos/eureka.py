
import sys, os, time
#sys.path.append('/Users/stevekb1/Documents/code/Eureka/Eureka/eureka/S3_data_reduction')
#sys.path.append('/Users/stevekb1/Documents/code/Eureka/Eureka/eureka/lib')
sys.path.append('/home/zieba/Desktop/Projects/Open_source/Eureka/eureka/S3_data_reduction')
sys.path.append('/home/zieba/Desktop/Projects/Open_source/Eureka/eureka/lib')

from importlib import reload
import s3_reduce as s3
reload(s3)

eventlabel = 'wasp43b'

ev = s3.reduceJWST(eventlabel, isplots=7)
