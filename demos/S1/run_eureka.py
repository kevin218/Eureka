import sys
sys.path.append('../..')
from eureka.S1_detector_processing.s1_process import EurekaS1Pipeline

eventlabel = 'nirx_template'
# eventlabel = 'miri_template'

if __name__ == '__main__':
s1 = EurekaS1Pipeline()
ev2 = s1.run_eurekaS1(eventlabel)