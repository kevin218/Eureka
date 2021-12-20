import numpy as np
import matplotlib.pyplot as plt
import glob, os
from ..lib import manageevent as me
from ..lib import readECF as rd
from ..lib import sort_nicely as sn
from ..lib import util
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
        # Search for the S2 output metadata in the inputdir provided in 
        # First just check the specific inputdir folder
        rootdir = os.path.join(s5_meta.topdir, *s5_meta.inputdir.split(os.sep))
        if rootdir[-1]!='/':
            rootdir += '/'
        files = glob.glob(rootdir+'S4_'+s5_meta.eventlabel+'*_Meta_Save.dat')
        if len(files)==0:
            # There were no metadata files in that folder, so let's see if there are in children folders
            files = glob.glob(rootdir+'**/S4_'+s5_meta.eventlabel+'*_Meta_Save.dat', recursive=True)
            files = sn.sort_nicely(files)

        if len(files)==0:
            # There may be no metafiles in the inputdir - raise an error and give a helpful message
            raise AssertionError('Unable to find an output metadata file from Eureka!\'s S4 step '
                                +'in the inputdir: \n"{}"!'.format(rootdir))

        elif len(files)>1:
            # There may be multiple runs - use the most recent but warn the user
            print('WARNING: There are multiple metadata save files in your inputdir: \n"{}"\n'.format(rootdir)
                 +'Using the metadata file: \n{}\n'.format(files[-1])
                 +'and will consider aperture ranges listed there. If this metadata file is not a part,\n'
                 +'of the run you intended, please provide a more precise folder for the metadata file.')

        fname = files[-1] # Pick the last file name (should be the most recent or only file)
        fname = fname[:-4] # Strip off the .dat ending

        s4_meta = me.loadevent(fname)
        
    s4_outputdir = s4_meta.outputdir[len(s4_meta.topdir):]
    if s4_outputdir[0]=='/':
        s4_outputdir = s4_outputdir[1:]
    # Overwrite the inputdir with the exact output directory from S4
    s5_meta.inputdir = s4_outputdir
    s5_meta.old_datetime = s4_meta.datetime
    s5_meta.datetime = None
    s5_meta.inputdir_raw = s5_meta.inputdir
    s5_meta.outputdir_raw = s5_meta.outputdir
    # Create directories for Stage 5 processing outputs
    tempfolder = s5_meta.outputdir_raw
    s5_meta.outputdir_raw = s5_meta.inputdir_raw
    s5_meta.inputdir = util.pathdirectory(s4_meta, 'S4', s4_meta.runs[0], old_datetime=s5_meta.old_datetime)
    s5_meta.outputdir_raw = tempfolder
    run = util.makedirectory(s5_meta, 'S5')
    s5_meta.outputdir = util.pathdirectory(s5_meta, 'S5', run)
    
    t0_offset = s5_meta.toffset
    t_bjdtdb=s4_meta.bjdtdb - t0_offset
    for channel in range(s4_meta.nspecchan):
        flux = s4_meta.lcdata[channel,:]
        flux_err = s4_meta.lcerr[channel,:]
        
        #FINDME: these two lines are because we don't have a constant offset model implemented yet. Will remove later
        flux = flux / np.median(flux[:200])
        flux_err = flux_err/800000000/3
        
        lc_model = lc.LightCurve(t_bjdtdb, flux, unc=flux_err, name=eventlabel)
        
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
            lc_model.fit(model, s5_meta, fitter='lsq')
        if 'mcmc' in s5_meta.fit_method:
            lc_model.fit(model, s5_meta, fitter='emcee')
        if 'nested' in s5_meta.fit_method:
            lc_model.fit(model, s5_meta, fitter='dynesty')
        if s5_meta.isplots_S5 > 1:
            lc_model.plot(s5_meta, draw=True)
    
    return
