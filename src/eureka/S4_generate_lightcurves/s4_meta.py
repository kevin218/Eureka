import numpy as np

from ..lib.readECF import MetaClass


class S4MetaClass(MetaClass):
    '''A class to hold Eureka! S4 metadata.

    This class loads a Stage 4 Eureka! Control File (ecf) and lets you
    query the parameters and values.

    Notes
    -----
    History:

    - 2024-04 Taylor J Bell
        Made specific S4 class based on MetaClass
    '''

    def __init__(self, folder=None, file=None, eventlabel=None, **kwargs):
        '''Initialize the MetaClass object.

        Parameters
        ----------
        folder : str; optional
            The folder containing an ECF file to be read in. Defaults to None
            which resolves to './'.
        file : str; optional
            The ECF filename to be read in. Defaults to None which first tries
            to find the filename using eventlabel and stage, and if that fails
            results in an empty MetaClass object.
        eventlabel : str; optional
            The unique identifier for these data.
        **kwargs : dict
            Any additional parameters to be loaded into the MetaClass after
            the ECF has been read in

        Notes
        -----
        History:

        - 2024-04 Taylor J Bell
            Initial version.
        '''
        super.__init__(folder, file, eventlabel, stage=4, **kwargs)

    def set_defaults(self):
        '''Set Stage 4 specific defaults for generic instruments.

        Notes
        -----
        History:

        - 2024-04 Taylor J Bell
            Initial version setting defaults for any instrument.
        '''
        # Repeat Stage 4 for each of the aperture sizes run from Stage 3?
        self.allapers = getattr(self, 'allapers', False)

        # Make sure the inst and filt attributes are at least initialized
        self.inst = getattr(self, 'inst', None)
        self.filter = getattr(self, 'filter', None)

        # Make sure the S3 expand parameter is defined
        # (to allow resuming from old analyses)
        self.expand = getattr(self, 'expand', 1)

        # Spectral binning/trimming control
        self.nspecchan = getattr(self, 'nspecchan', None)
        self.compute_white = getattr(self, 'compute_white', True)
        self.wave_min = getattr(self, 'wave_min', None)
        self.wave_max = getattr(self, 'wave_max', None)
        self.wave_hi = getattr(self, 'wave_hi', None)
        self.wave_low = getattr(self, 'wave_hi', None)

        # Manually mask pixel columns by index number
        self.mask_columns = getattr(self, 'mask_columns', [])

        # Parameters for drift tracking/correction of 1D spectra
        self.recordDrift = getattr(self, 'recordDrift', False)
        self.correctDrift = getattr(self, 'correctDrift', False)
        self.drift_preclip = getattr(self, 'drift_preclip', 0)
        self.drift_postclip = getattr(self, 'drift_postclip', 100)
        self.drift_range = getattr(self, 'drift_range', 11)
        self.drift_hw = getattr(self, 'drift_hw', 5)
        self.drift_iref = getattr(self, 'drift_iref', -1)
        self.sub_mean = getattr(self, 'sub_mean', True)
        self.sub_continuum = getattr(self, 'sub_continuum', True)
        self.highpassWidth = getattr(self, 'highpassWidth', 10)

        # Parameters for sigma clipping
        self.clip_unbinned = getattr(self, 'clip_unbinned', False)
        self.clip_binned = getattr(self, 'clip_binned', True)
        if self.clip_unbinned or self.clip_binned:
            # Require these parameters to be explicitly set since there isn't
            # really a generically safe value
            self.sigma = getattr(self, 'sigma')
            self.box_width = getattr(self, 'box_width')
        self.maxiters = getattr(self, 'maxiters', 20)
        self.boundary = getattr(self, 'boundary', 'fill')
        self.fill_value = getattr(self, 'fill_value', 'mask')

        # HST/WFC3 temporal binning (sum together all reads from one scan)
        self.sum_reads = getattr(self, 'sum_reads', True)

        # Limb-darkening parameters
        self.compute_ld = getattr(self, 'compute_ld', False)
        if self.compute_ld == 'spam':
            # Require the file to be specified if relevant
            self.spam_file = getattr(self, 'spam_file')
        elif self.compute_ld == 'exotic-ld':
            self.custom_si_grid = getattr(self, 'custom_si_grid', None)
            if self.custom_si_grid is None:
                self.exotic_ld_file = getattr(self, 'exotic_ld_file', None)
                if self.exotic_ld_file is None:
                    # Require the file to be specified if relevant            
                    self.inst_filter = getattr(self, 'inst_filter')  # FINDME: This should never need to be manually passed in - we should be able to use meta.inst and meta.filter.
                # Require the following to be specified if relevant
                self.exotic_ld_direc = getattr(self, 'exotic_ld_direc')
                self.exotic_ld_grid = getattr(self, 'exotic_ld_grid')
                self.metallicity = getattr(self, 'metallicity')
                self.teff = getattr(self, 'teff')
                self.logg = getattr(self, 'logg')

        # Diagnostics
        self.isplots_S4 = getattr(self, 'isplots_S4', 3)
        self.nplots = getattr(self, 'nplots', None)
        self.vmin = getattr(self, 'vmin', 0.97)
        self.vmax = getattr(self, 'vmax', 1.03)
        self.time_axis = getattr(self, 'time_axis', 'y')
        if self.time_axis not in ['y', 'x']:
            print("WARNING: meta.time_axis is not one of ['y', 'x']!"
                  " Using 'y' by default.")
            self.time_axis = 'y'
        self.hide_plots = getattr(self, 'hide_plots', False)
        self.verbose = getattr(self, 'verbose', True)
        
        # Project directory
        self.topdir = getattr(self, 'topdir')  # Must be provided in the ECF

        # Directories relative to topdir
        self.inputdir = getattr(self, 'inputdir', 'Stage3')
        self.outputdir = getattr(self, 'outputdir', 'Stage4')
