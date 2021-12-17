import numpy as np
import matplotlib.pyplot as plt
import glob, os
from ..lib import manageevent as me
from ..lib import readECF as rd
from ..lib import sort_nicely as sn
from . import parameters as p
from . import lightcurve as lc
from . import models as m
from .utils import get_target_data

class MetaClass:
    '''A class to hold Eureka! metadata.
    '''

    def __init__(self):
        return

def fitJWST(eventlabel, s4_meta=None):
    
    # Initialize a new metadata object
    s5_meta = MetaClass()
    s5_meta.eventlabel = eventlabel
    
    # Load Eureka! control file and store values in Event object
    ecffile = 'S5_' + eventlabel + '.ecf'
    ecf = rd.read_ecf(ecffile)
    rd.store_ecf(s5_meta, ecf)
    
    # load savefile
    if s4_meta == None:
        s4_meta = me.load(s5_meta.inputdir + '/S4_' + eventlabel + '_Meta_Save.dat')
    
    #Load Eureka! control files and stire values in Event object
    #FINDME: FINISH THIS. ONCE S4 RUNS, CONNECT S5 TO S4.
    
    # Create directories for Stage 4 processing
    files = glob.glob(os.path.join(s5_meta.inputdir + "/speclc", "*.txt")) #FINDME: REPLACE THIS PART TO ACCEPT STAGE 4 OUTPUT
    files = sn.sort_nicely(files)
    if s5_meta.run_verbose:
        print(files)
    t0_offset = s5_meta.toffset
    for f in [files[0]]: #FINDME: REPLACE
        t_bjdtdb, flux, flux_err = np.loadtxt(f, skiprows=1).T
        t_bjdtdb = t_bjdtdb - t0_offset
        flux = flux / np.median(flux[:200])
        flux_err = flux_err/800000000/3
        
        lc_model = lc.LightCurve(t_bjdtdb, flux, unc=flux_err, name='WASP-43b')
        
        # Set the intial parameters
        params = p.Parameters(param_file=s5_meta.fit_par)
        if s5_meta.run_verbose:
            print(params)
        
        # Make the transit model
        modellist=[]
        if 'transit' in s5_meta.run_myfuncs:
            t_model = m.TransitModel(parameters=params, name='transit', fmt='r--')
            modellist.append(t_model)
        if 'polynomial' in s5_meta.run_myfuncs:
            t_polynom = m.PolynomialModel(parameters=params, name='polynom', fmt='r--')
            modellist.append(t_polynom)
        model = m.CompositeModel(modellist)
        
        if 'lsq' in s5_meta.fit_method:
            lc_model.fit(model, s5_meta, fitter='lsq', **s5_meta)
        if 'mcmc' in s5_meta.fit_method:
            lc_model.fit(model, s5_meta, fitter='emcee', **s5_meta)
        if 'nested' in s5_meta.fit_method:
            lc_model.fit(model, s5_meta, fitter='dynesty', **s5_meta)
        if s5_meta.isplots_S5 > 1:
            lc_model.plot(s5_meta, raw=True)
    
    return
