import sys
sys.path.append('../..')
from importlib import reload
import eureka.S4_generate_lightcurves.s4_genLC as s4

eventlabel = 'template'

s4_meta = s4.lcJWST(eventlabel)
