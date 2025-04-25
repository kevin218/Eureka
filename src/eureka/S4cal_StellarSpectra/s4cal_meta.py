from eureka.lib.readECF import MetaClass


class S4cal_MetaClass(MetaClass):
    '''A class to hold Eureka! S4cal metadata.

    This class loads a Stage 4cal Eureka! Control File (ecf) and lets you
    query the parameters and values.
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
        '''
        super().__init__(folder, file, eventlabel, stage='4cal', **kwargs)

    def set_defaults(self):
        '''Set Stage 4cal specific defaults for generic instruments.
        '''
        # System parameters
        self.rprs = getattr(self, 'rprs', None)
        self.period = getattr(self, 'period', None)
        self.t0 = getattr(self, 't0')
        self.time_offset = getattr(self, 'time_offset', 0)
        self.inc = getattr(self, 'inc', None)
        self.ars = getattr(self, 'ars', None)

        self.t14 = getattr(self, 't14', None)
        self.t23 = getattr(self, 't23', None)
        self.base_dur = getattr(self, 'base_dur', None)

        # Aperture correction
        self.apcorr = getattr(self, 'apcorr', 1.0)
        if self.apcorr is None:
            self.apcorr = 1.0

        # Outlier detection
        self.smoothing = getattr(self, 'smoothing', 0)
        self.sigma_thresh = getattr(self, 'sigma_thresh', [4, 4, 4])

        # Diagnostics
        self.isplots_S4cal = getattr(self, 'isplots_S4cal', 3)
        self.nbin_plot = getattr(self, 'nbin_plot', 100)
        self.s4cal_plotErrorType = getattr(self, 's4cal_plotErrorType',
                                           'stderr')
        if self.s4cal_plotErrorType not in ['stderr', 'stddev']:
            raise ValueError("Unknown s4cal_plotErrorType value: "
                             f"{self.s4cal_plotErrorType} is not one of "
                             "'stderr' or 'stddev'")
        self.hide_plots = getattr(self, 'hide_plots', False)
        self.verbose = getattr(self, 'verbose', True)

        # Project directory
        self.topdir = getattr(self, 'topdir')  # Must be provided in the ECF

        # Directories relative to topdir
        self.inputdir = getattr(self, 'inputdir', 'Stage3')
        self.outputdir = getattr(self, 'outputdir', 'Stage4cal')
