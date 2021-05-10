#! /usr/bin/env python

# Generic Stage 3 reduction pipeline

"""
# Proposed Steps
# -------- -----
# 1.  Read in all data frames and header info from Stage 2 data products
# 2.  Record JD and other relevant header information
# 3.  Apply light-time correction (if necessary)
# 4.  Calculate trace and 1D+2D wavelength solutions (if necessary)
# 5.  Make flats, apply flat field correction (Stage 2)
# 6.  Manually mask regions
# 7.  Compute difference frames OR slopes (Stage 1)
# 8.  Perform outlier rejection of BG region
# 9.  Background subtraction
# 10. Compute 2D drift, apply rough (integer-pixel) correction
# 11. Full-frame outlier rejection for time-series stack of NDRs
# 12. Apply sub-pixel 2D drift correction
# 13. Extract spectrum through summation
# 14. Compute median frame
# 15. Optimal spectral extraction
# 16. Save Stage 3 data products
# 17. Produce plots
"""

import sys, os, time
sys.path.append('/Users/stevekb1/Documents/code/Eureka/Eureka/eureka/S3_data_reduction')
sys.path.append('/Users/stevekb1/Documents/code/Eureka/Eureka/eureka/lib')
from importlib import reload
import numpy as np
import logedit
import readECF as rd
import manageevent as me

class Event():
  def __init__(self):

    # initialize Univ
    #Univ.__init__(self)
    #self.initpars(ecf)
    #self.foo = 2
    return

def reduceJWST(eventlabel, isplots=False):
    '''
    Reduces data images and calculated optimal spectra.

    Parameters
    ----------
    eventlabel  : str, Unique label for this dataset
    isplots     : boolean, Set True to produce plots

    Returns
    -------
    ev          : Event object

    Remarks
    -------


    History
    -------
    Written by Kevin Stevenson      May 2021

    '''

    t0      = time.time()

    # Initialize event object
    ev              = Event()
    ev.eventlabel   = eventlabel

    # Create directories for Stage 3 processing
    datetime= time.strftime('%Y-%m-%d_%H-%M-%S')
    ev.dirname = 'S3_' + datetime + '_' + ev.eventlabel
    if not os.path.exists(ev.dirname):
        os.makedirs(ev.dirname)
    if not os.path.exists(ev.dirname+"/figs"):
        os.makedirs(ev.dirname+"/figs")

    # Load Eureka! control file and store values in Event object
    ecffile = 'S3_' + eventlabel + '.ecf'
    ecf     = rd.read_ecf(ecffile)
    rd.store_ecf(ev, ecf)

    # Open new log file
    ev.logname  = './'+ev.dirname + '/S3_' + ev.eventlabel + ".log"
    log         = logedit.Logedit(ev.logname)
    log.writelog("\nStarting Stage 3 Reduction")


    # Calculate total time
    total = (time.time() - t0)/60.
    log.writelog('\nTotal time (min): ' + str(np.round(total,2)))


    # Save results
    log.writelog('Saving results...')
    me.saveevent(ev, ev.dirname + '/S3_' + ev.eventlabel + "_Save", save=[])

    log.closelog()
    return ev
