import numpy as np

# Define event object
class event_init:
    def __init__(self):
        
        # Location and name of file for flat fielding
        #self.flatfile  = "/home/kevin/Documents/UChicago/data/ancil/flats/WFC3.IR.G141.flat.2.fits"
        self.flatfile  = '/grp/hst/wfc3a/automated_outputs/cal_ir_make_spatial_scan/sedFFcube-both.fits'

        # Variables that might need to change
        self.ncpu       = 1         # Number of CPUs
        self.iref       = [2,1]     # Index a reference frame for given scan direction
        self.inormspec  = [-10,-1]  # Index of frames to normalize against in Figure 3000
        
        # Variables that are roughly optimized and shouldn't be changed
        self.expand     = 1         # Factor by which to increase resolution
        self.flatsigma  = 30        # Sigma cutoff
        self.diffthresh = 6         # Sigma cutoff
        self.sigthresh  = [4,4]     # Sigma cutoff
        self.p3thresh   = 6         # Sigma cutoff
        self.p5thresh   = 8         # Sigma cutoff
        self.p7thresh   = 8         # Sigma cutoff
        self.fittype    = 'smooth'  # Optimal spectral extraction
        self.window_len = 11        # Optimal spectral extraction
        self.deg        = None      # Optimal spectral extraction
        self.bgdeg      = 0         # Fit background with polynomial of given degree
        
        # Save root directory
        self.savedir    = '/grp/hst/wfc3a/automated_outputs/cal_ir_make_spatial_scan/daily_outputs/'
        
        return

