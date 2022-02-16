import numpy as np
import os, glob, time
from ..lib import manageevent as me
from ..lib import readECF as rd
from ..lib import util, logedit
from ..lib import sort_nicely as sn

#FINDME: Keep reload statements for easy testing
from importlib import reload

class MetaClass:
    '''A class to hold Eureka! metadata.
    '''

    def __init__(self):
        return

def plot_spectra(eventlabel, s5_meta=None):
    '''Gathers together diferent wavelength fits and makes transmission/emission spectra.

    Parameters
    ----------
    eventlabel: str
        The unique identifier for these data.
    s5_meta:    MetaClass
        The metadata object from Eureka!'s S5 step (if running S5 and S6 sequentially).

    Returns
    -------
    meta:   MetaClass
        The metadata object with attributes added by S6.

    Notes
    -------
    History:

    - Feb 14, 2022 Taylor Bell
        Original version
    '''
    print("\nStarting Stage 6: Light Curve Fitting\n")

    # Initialize a new metadata object
    meta = MetaClass()
    meta.eventlabel = eventlabel

    # Load Eureka! control file and store values in Event object
    ecffile = 'S6_' + eventlabel + '.ecf'
    ecf = rd.read_ecf(ecffile)
    rd.store_ecf(meta, ecf)

    # load savefile
    if s4_meta == None:
        s4_meta = read_s5_meta(meta)

    meta = load_general_s5_meta_info(meta, s5_meta)

    if (not meta.s5_allapers) or (not meta.allapers):
        # The user indicated in the ecf that they only want to consider one aperture
        # in which case the code will consider only the one which made s4_meta.
        # Alternatively, S4 was run without allapers, so S6's allapers will only conside that one
        meta.spec_hw_range = [meta.spec_hw,]
        meta.bg_hw_range = [meta.bg_hw,]

    if meta.testing_S6:
        # Only fit a single channel while testing
        chanrng = [0]
    else:
        chanrng = range(meta.nspecchan)

    # Create directories for Stage 6 outputs
    meta.runs_s6 = []
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:
            run = util.makedirectory(meta, 'S6', ap=spec_hw_val, bg=bg_hw_val)
            meta.runs_s6.append(run)

    run_i = 0
    old_meta = meta
    for spec_hw_val in meta.spec_hw_range:
        for bg_hw_val in meta.bg_hw_range:
            
            t0 = time.time()
            
            meta = load_specific_s5_meta_info(old_meta, run_i, spec_hw_val, bg_hw_val)
            
            # Get the directory for Stage 6 processing outputs
            meta.outputdir = util.pathdirectory(meta, 'S6', meta.runs_s6[run_i], ap=spec_hw_val, bg=bg_hw_val)
            run_i += 1
            
            # Copy existing S4 log file and resume log
            meta.s6_logname  = meta.outputdir + 'S6_' + meta.eventlabel + ".log"
            log         = logedit.Logedit(meta.s6_logname, read=meta.s5_logname)
            log.writelog(f"Input directory: {meta.inputdir}")
            log.writelog(f"Output directory: {meta.outputdir}")
            
            # Copy ecf (and update outputdir in case S6 is being called sequentially with S5)
            log.writelog('Copying S6 control file')
            new_ecfname = meta.outputdir + ecffile.split('/')[-1]
            with open(new_ecfname, 'w') as new_file:
                with open(ecffile, 'r') as file:
                    for line in file.readlines():
                        if len(line.strip())==0 or line.strip()[0]=='#':
                            new_file.write(line)
                        else:
                            line_segs = line.strip().split()
                            if line_segs[0]=='inputdir':
                                new_file.write(line_segs[0]+'\t\t/'+meta.inputdir+'\t'+' '.join(line_segs[2:])+'\n')
                            else:
                                new_file.write(line)
            
            # Do S6 stuff
            
            # Calculate total time
            total = (time.time() - t0) / 60.
            log.writelog('\nTotal time (min): ' + str(np.round(total, 2)))
            
            # Save results
            log.writelog('Saving results')
            me.saveevent(meta, meta.outputdir + 'S6_' + meta.eventlabel + "_Meta_Save", save=[])
            
            log.closelog()
    
    return meta

def read_s5_meta(meta):

    # Search for the S5 output metadata in the inputdir provided in
    # First just check the specific inputdir folder
    rootdir = os.path.join(meta.topdir, *meta.inputdir.split(os.sep))
    if rootdir[-1]!='/':
        rootdir += '/'
    files = glob.glob(rootdir+'S5_'+meta.eventlabel+'*_Meta_Save.dat')
    if len(files)==0:
        # There were no metadata files in that folder, so let's see if there are in children folders
        files = glob.glob(rootdir+'**/S5_'+meta.eventlabel+'*_Meta_Save.dat', recursive=True)
        files = sn.sort_nicely(files)

    if len(files)==0:
        # There may be no metafiles in the inputdir - raise an error and give a helpful message
        raise AssertionError('Unable to find an output metadata file from Eureka!\'s S5 step '
                            +'in the inputdir: \n"{}"!'.format(rootdir))

    elif len(files)>1:
        # There may be multiple runs - use the most recent but warn the user
        print('WARNING: There are multiple metadata save files in your inputdir: \n"{}"\n'.format(rootdir)
                +'Using the metadata file: \n{}\n'.format(files[-1])
                +'and will consider aperture ranges listed there. If this metadata file is not a part\n'
                +'of the run you intended, please provide a more precise folder for the metadata file.')

    fname = files[-1] # Pick the last file name (should be the most recent or only file)
    fname = fname[:-4] # Strip off the .dat ending

    s5_meta = me.loadevent(fname)

    return s5_meta

def load_general_s5_meta_info(meta, s5_meta):

    # Need to remove the topdir from the outputdir
    s5_outputdir = s5_meta.outputdir[len(s5_meta.topdir):]
    if s5_outputdir[0]=='/':
        s5_outputdir = s5_outputdir[1:]
    if s5_outputdir[-1]!='/':
        s5_outputdir += '/'
    s5_allapers = s5_meta.allapers

    # Overwrite the temporary meta object made above to be able to find s5_meta
    meta = s5_meta

    # Load Eureka! control file and store values in the S4 metadata object
    ecffile = 'S6_' + meta.eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(meta, ecf)

    # Overwrite the inputdir with the exact output directory from S5
    meta.inputdir = s5_outputdir
    meta.old_datetime = s5_meta.datetime # Capture the date that the
    meta.datetime = None # Reset the datetime in case we're running this on a different day
    meta.inputdir_raw = meta.inputdir
    meta.outputdir_raw = meta.outputdir

    meta.s5_allapers = s5_allapers

    return meta

def load_specific_s5_meta_info(meta, run_i, spec_hw_val, bg_hw_val):
    # Do some folder swapping to be able to reuse this function to find the correct S5 outputs
    tempfolder = meta.outputdir_raw
    meta.outputdir_raw = '/'.join(meta.inputdir_raw.split('/')[:-2])
    meta.inputdir = util.pathdirectory(meta, 'S5', meta.runs_s5[run_i], old_datetime=meta.old_datetime, ap=spec_hw_val, bg=bg_hw_val)
    meta.outputdir_raw = tempfolder

    # Read in the correct S5 metadata for this aperture pair
    tempfolder = meta.inputdir
    meta.inputdir = meta.inputdir[len(meta.topdir):]
    new_meta = read_s5_meta(meta)
    meta.inputdir = tempfolder

    # Load S6 Eureka! control file and store values in the S5 metadata object
    ecffile = 'S6_' + meta.eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(new_meta, ecf)

    # Save correctly identified folders from earlier
    new_meta.inputdir = meta.inputdir
    new_meta.outputdir = meta.outputdir
    new_meta.inputdir_raw = meta.inputdir_raw
    new_meta.outputdir_raw = meta.outputdir_raw

    new_meta.runs_s6 = meta.runs_s6
    new_meta.datetime = meta.datetime

    return new_meta
