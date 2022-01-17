import sys
sys.path.append('../../')
import eureka.S4_generate_lightcurves.s4_genLC as s4
import eureka.S5_lightcurve_fitting.s5_fit as s5

eventlabel = 'template'

s4_meta = s4.lcJWST(eventlabel)

s5_meta, lc_model = s5.fitJWST(eventlabel, s4_meta=s4_meta)
