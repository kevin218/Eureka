import os
import numpy as np
from bokeh.io import output_notebook
from bokeh.plotting import show
from hotsoss import plotting as plt

# Get the orbital parameters
from exoctk.utils import get_target_data
wasp107b_params, url = get_target_data('WASP-107b')

# Generate the simulated light curve
from exoctk.lightcurve_fitting.simulations import simulate_lightcurve
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

from exoctk.lightcurve_fitting.lightcurve import LightCurve
wasp107b_lc = LightCurve(wasp107b_time, wasp107b_flux[0], unc=wasp107b_unc[0], name='WASP-107b')

# Set the intial parameters
from exoctk.lightcurve_fitting.parameters import Parameters
params = Parameters()
params.rp = wasp107b_par['Rp/Rs']+0.02, 'free', 0.1, 0.2
params.per = wasp107b_par['orbital_period'], 'fixed'
params.t0 = wasp107b_par['transit_time']-0.01, 'free', wasp107b_par['transit_time']-0.1, wasp107b_par['transit_time']+0.1
params.inc = wasp107b_par['inclination'], 'free', 80., 90.
params.a = wasp107b_par['a/Rs'], 'free', 10., 25.
params.ecc = wasp107b_par['eccentricity'], 'fixed'
params.w = 90, 'fixed'    #wasp107b_par['omega'], 'fixed'
params.limb_dark = 'quadratic', 'independent'
params.transittype = 'primary', 'independent'
params.u1 = 0.1, 'free', 0., 1.
params.u2 = 0.1, 'free', 0., 1.

# Make the transit model
from exoctk.lightcurve_fitting.models import TransitModel
t_model = TransitModel(parameters=params, name='transit', fmt='r--')

# Plot it
t_model.plot(wasp107b_time, draw=True)

# Needed to install numdifftools
# pip install numdifftools

# Create a new model instance from the best fit parameters
#wasp107b_lc.fit(t_model, fitter='lmfit', method='powell')
wasp107b_lc.fit(t_model, fitter='lmfit', method='least_squares')
wasp107b_lc.fit(t_model, fitter='lmfit', method='differential_evolution')

# Plot it
wasp107b_lc.plot()
# Fit doesn't work!!!


##########################
from exoctk.lightcurve_fitting.models import CompositeModel
model = CompositeModel([t_model])

import lmfit
initialParams = lmfit.Parameters()
all_params = [i for j in [model.components[n].parameters.list
              for n in range(len(model.components))] for i in j]

# Group the different variable types
param_list = []
indep_vars = {}
for param in all_params:
    param = list(param)
    if param[2] == 'free':
        param[2] = True
        param_list.append(tuple(param))
    elif param[2] == 'fixed':
        param[2] = False
        param_list.append(tuple(param))
    else:
        indep_vars[param[0]] = param[1]
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
