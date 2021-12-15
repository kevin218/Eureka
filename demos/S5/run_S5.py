import sys, os, time
sys.path.append('../..')
#sys.path.append('/Users/stevekb1/Documents/code/Eureka/Eureka')
#sys.path.append('/Users/kreidberg/Desktop/Projects/OpenSource/Eureka/')
sys.path.append('/Users/megan/Documents/Code/Eureka/Eureka')
from importlib import reload
import eureka.S5_lightcurve_fitting.s5_fit as s5
import eureka.S5_lightcurve_fitting.parameters as p
reload(p)

meta = '/Users/megan/Documents/Code/Eureka/Eureka/demos/S3/S3_2021-10-11_15-10-58_wasp43b/S4_2021-10-11_15-23-10_6chan/S4_wasp43b_Meta_Save.dat'
eventlabel = 'wasp43b'
fit_par= './s5_fit_par.ecf'
run_par_file = './s5.ecf'
run_par = p.Parameters(param_file=run_par_file)
workdir = run_par.workdir.value
speclc_dir = run_par.speclc_dir.value
reload(s5)
ev5 = s5.fitJWST(eventlabel, workdir, speclc_dir, fit_par, run_par_file, meta=meta)
