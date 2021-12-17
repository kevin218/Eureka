import numpy as np
from ..lib import manageevent as me
from ..lib import readECF as rd
import glob, os
from ..lib import sort_nicely as sn
import matplotlib.pyplot as plt
from importlib import reload

def fitJWST(eventlabel, workdir, speclc_dir, fit_par, run_par_file, meta):
    # load savefile
    if meta == None:
        meta = me.load(speclc_dir + '/S4_' + eventlabel + '_Meta_Save.dat')

    #Load Eureka! control files and stire values in Event object
    #FINDME: FINISH THIS. ONCE S4 RUNS, CONNECT S5 TO S4.

    #read in run params
    from . import parameters as p
    reload(p)
    run_par=p.Parameters(param_file=run_par_file)

    # Create directories for Stage 4 processing
    files = glob.glob(os.path.join(speclc_dir + "/speclc", "*.txt")) #FINDME: REPLACE THIS PART TO ACCEPT STAGE 4 OUTPUT
    files = sn.sort_nicely(files)
    if run_par.run_verbose.value:
        print(files)
    t0_offset = run_par.toffset.value
    for f in [files[0]]: #FINDME: REPLACE
        t_bjdtdb, flux, flux_err = np.loadtxt(f, skiprows=1).T
        t_bjdtdb = t_bjdtdb - t0_offset
        flux = flux / np.median(flux[:200])
        flux_err = flux_err/800000000/3

        from . import lightcurve as lc
        reload(lc)
        wasp43b_lc = lc.LightCurve(t_bjdtdb, flux, unc=flux_err, name='WASP-43b')

        # Get the orbital parameters
        from .utils import get_target_data
        wasp43b_params, url = get_target_data('WASP-43b')

        # Set the intial parameters
        params = p.Parameters(param_file=fit_par)
        if run_par.run_verbose.value:
            print(params)

        # Make the transit model
        from . import models as m
        reload(m)
        modellist=[]
        if 'transit' in run_par.run_myfuncs.value:
            t_model = m.TransitModel(parameters=params, name='transit', fmt='r--')
            modellist.append(t_model)
        if 'polynomial' in run_par.run_myfuncs.value:
            t_polynom = m.PolynomialModel(parameters=params, name='polynom', fmt='r--')
            modellist.append(t_polynom)
        model = m.CompositeModel(modellist)

        if 'lsq' in run_par.fit_method.value:
            wasp43b_lc.fit(model, fitter='lsq', **run_par.dict)
        if 'mcmc' in run_par.fit_method.value:
            wasp43b_lc.fit(model, fitter='emcee', **run_par.dict)
        if 'nested' in run_par.fit_method.value:
            wasp43b_lc.fit(model, fitter='dynesty', **run_par.dict)
        if run_par.run_show_plot.value:
            wasp43b_lc.plot(draw=True)
