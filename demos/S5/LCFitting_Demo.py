
import os, sys
import numpy as np
#from bokeh.io import output_notebook
#from bokeh.plotting import show
from importlib import reload
#from hotsoss import plotting as plt
sys.path.append('../..')
import eureka
import pdb

# Get the orbital parameters
from eureka.S5_lightcurve_fitting.utils import get_target_data
wasp107b_params, url = get_target_data('WASP-107b')

#load in run parameters from ecf file
def make_dict(table):
    return {x['parameter']: x['value'] for x in table}


# Generate the simulated light curve
from eureka.S5_lightcurve_fitting.simulations import simulate_lightcurve
npts = 1000
snr  = 4000.
#wasp107b_unc = np.ones(npts)/snr
wasp107b_time, wasp107b_flux, wasp107b_unc, wasp107b_par = simulate_lightcurve('WASP-107b', snr=snr, npts=npts, plot=False)

# Create NIRISS/SOSS filter object
#from svo_filters import Filter
#gr700xd = Filter('NIRISS.GR700XD.1', n_bins=15)

# Plot the light curve
#wasp107b_wave = np.linspace(gr700xd.wave_min, gr700xd.wave_max, 2048).value
#wasp107b_spec = plt.plot_time_series_spectra(wasp107b_wave, wasp107b_flux)
#show(wasp107b_spec)

import eureka.S5_lightcurve_fitting.lightcurve as lc
reload(lc)
wasp107b_lc = lc.LightCurve(wasp107b_time, wasp107b_flux[0], unc=wasp107b_unc[0], name='WASP-107b')

# from eureka.lib import readECF as rd
# reload(rd)
# ecf = rd.read_ecf('s5.ecf')


# Set the intial parameters
import eureka.S5_lightcurve_fitting.parameters as p
reload(p)

params = p.Parameters(param_file='s5_fit_par.ecf')
params.limb_dark = 'quadratic', 'independent'
params.transittype = 'primary', 'independent'

# Make the transit model
import eureka.S5_lightcurve_fitting.models as m
reload(m)
t_model = m.TransitModel(parameters=params, name='transit', fmt='r--')
model = m.CompositeModel([t_model])

wasp107b_lc.fit(model, fitter='lsq')

wasp107b_lc.plot(draw=True)

#FINDME: getting this to work
params2=p.Parameters(param_file='s5.ecf')
# Plot it
#t_model.plot(wasp107b_time, draw=True)
'''
newparams = []
for arg, val in params.dict.items():
    newparams.append(val[0])
newparams[0] = 0.3
t_model.update(newparams)
t_model.plot(wasp107b_time, draw=True)
'''
# Needed to install numdifftools
# pip install numdifftools

# Perform fit using LMFIT
# Create a new model instance from the best fit parameters
#wasp107b_lc.fit(t_model, fitter='lmfit', method='powell')
#wasp107b_lc2.fit(t_model, fitter='lmfit', method='least_squares')
#wasp107b_lc2.fit(t_model, fitter='lmfit', method='differential_evolution')

# Plot it
#wasp107b_lc2.plot()
# Fit doesn't work!!!


############
# Code below runs, but still doesn't fit the data
##############
"""
import lmfit
initialParams = lmfit.Parameters()
all_params = [i for j in [model.components[n].parameters.dict.items()
              for n in range(len(model.components))] for i in j]

# Group the different variable types
param_list = []
indep_vars = {}
for item in all_params:
    name, param = item
    if param[1] == 'free':
        param[1] = True
        param_list.append(tuple([name]+param))
    elif param[1] == 'fixed':
        param[1] = False
        param_list.append(tuple([name]+param))
    else:
        indep_vars[name] = param[0]
indep_vars['time'] = wasp107b_lc.time
# Get values from input parameters.Parameters instances
initialParams.add_many(*param_list)

# Create the lightcurve model
lcmodel = lmfit.Model(model.eval)
lcmodel.independent_vars = indep_vars.keys()

data = wasp107b_lc.flux
uncertainty = np.abs(wasp107b_lc.unc)


result = lcmodel.fit(data, weights=1./uncertainty, params=initialParams,
                     limb_dark='quadratic',transittype='primary',time=wasp107b_lc.time)
result.params

result2 = lcmodel.fit(data, weights=1/uncertainty, params=initialParams,
                    **indep_vars, method='least_squares')
result2.params
"""
