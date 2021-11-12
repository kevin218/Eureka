import numpy as np
from ..lib import manageevent as me
from ..lib import readECF as rd
import glob, os
from ..lib import sort_nicely as sn
import matplotlib.pyplot as plt
from importlib import reload
import pdb


def fitJWST(eventlabel, workdir, speclc_dir, fit_par, run_par_file, meta):
    # load savefile
    if meta == None:
        meta = me.load(speclc_dir + '/S4_' + eventlabel + '_Meta_Save.dat')

    #read in run params
    import eureka.S5_lightcurve_fitting.parameters as p
    reload(p)
    run_par=p.Parameters(param_file=run_par_file)
    #pdb.set_trace()
    # Create directories for Stage 4 processing
    files = glob.glob(os.path.join(speclc_dir + "/speclc", "*.txt"))
    files = sn.sort_nicely(files)
    if run_par.run_verbose.value:
        print(files)
    t0_offset = run_par.toffset.value#59694
    for f in [files[0]]:
        t_bjdtdb, flux, flux_err = np.loadtxt(f, skiprows=1).T
        t_bjdtdb = t_bjdtdb - t0_offset
        flux = flux / np.median(flux[:200])
        flux_err = flux_err/800000000/3

        import eureka.S5_lightcurve_fitting.lightcurve as lc
        reload(lc)
        wasp43b_lc = lc.LightCurve(t_bjdtdb, flux, unc=flux_err, name='WASP-43b')

        # Get the orbital parameters
        from eureka.S5_lightcurve_fitting.utils import get_target_data
        wasp43b_params, url = get_target_data('WASP-43b')

        # Set the intial parameters
        params = p.Parameters(param_file=fit_par)
        # params.rp = wasp43b_params['Rp/Rs'] + 0.02, 'free', 0.1, 0.2
        # params.per = wasp43b_params['orbital_period'], 'fixed'
        # params.t0 = 59694.15-t0_offset, 'free', 59694.12-t0_offset, 59694.18-t0_offset
        # print(params.t0)
        # params.inc = wasp43b_params['inclination'], 'free', 80., 90.
        # params.a = wasp43b_params['a/Rs'], 'free', 2., 15.
        # params.ecc = wasp43b_params['eccentricity'], 'fixed'
        # params.w = 90, 'fixed'  # wasp43b_par['omega'], 'fixed'
        # params.limb_dark = 'quadratic', 'independent'
        # params.transittype = 'primary', 'independent'
        # params.u1 = 0.7, 'free', 0., 1
        # params.u2 = 0.25, 'free', 0., 1
        #pdb.set_trace()
        #params=params.__add__(run_par)
        if run_par.run_verbose.value:
            print(params)
        #params.c0 = np.median(flux[:200]), 'free', np.median(flux[:200])*0.5, np.median(flux[:200])*1.5
        #params.c1 = 0., 'fixed'
        #params.c2 = 0., 'fixed'
        # Make the transit model
        import eureka.S5_lightcurve_fitting.models as m
        reload(m)
        t_model = m.TransitModel(parameters=params, name='transit', fmt='r--')
        #t_polynom = m.PolynomialModel(parameters=params, name='polynom', fmt='r--')
        #model = m.CompositeModel([t_model, t_polynom])
        model = m.CompositeModel([t_model])
        if run_par.run_lsq.value:
            wasp43b_lc.fit(model, fitter='lsq')
        elif run_par.run_mcmc.value:
            wasp43b_lc.fit(model, fitter='emcee')
        elif run_par.run_nested.value:
            wasp43b_lc.fit(model, fitter='dynesty')
        if run_par.run_show_plot.value:
            wasp43b_lc.plot(draw=True)
