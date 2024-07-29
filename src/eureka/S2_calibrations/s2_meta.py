from ..lib.readECF import MetaClass


class S2MetaClass(MetaClass):
    '''A class to hold Eureka! S2 metadata.

    This class loads a Stage 2 Eureka! Control File (ecf) and lets you
    query the parameters and values.

    Notes
    -----
    History:

    - 2024-03 Taylor J Bell
        Made specific S2 class based on MetaClass
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

        - 2024-03 Taylor J Bell
            Initial version.
        '''
        super().__init__(folder, file, eventlabel, stage=2, **kwargs)

    def set_defaults(self):
        '''Set Stage 2 specific defaults for generic instruments.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial version setting defaults for any instrument.
        '''
        # Data file suffix
        self.suffix = getattr(self, 'suffix', 'rateints')

        # Generic pipeline steps that'd usually be run for TSO data
        # Run by default for instruments that call the function, but not all
        # instruments call it so True is a safe default
        self.skip_extract_2d = getattr(self, 'skip_extract_2d', False)
        self.skip_srctype = getattr(self, 'skip_srctype', False)
        self.skip_flat_field = getattr(self, 'skip_flat_field', True)
        # Recommended to skip to get better uncertainties out of our Stage 3
        self.skip_photom = getattr(self, 'skip_photom', True)
        # No need to run this time consuming step by default
        self.skip_extract_1d = getattr(self, 'skip_extract_1d', True)

        # Steps that are not run by default for any TSO data
        self.skip_msaflagopen = getattr(self, 'skip_msaflagopen', True)
        self.skip_nsclean = getattr(self, 'skip_nsclean', True)
        self.skip_imprint = getattr(self, 'skip_imprint', True)
        self.skip_bkg_subtract = getattr(self, 'skip_bkg_subtract', True)
        self.skip_master_background = getattr(self, 'skip_master_background',
                                              True)
        self.skip_wavecorr = getattr(self, 'skip_wavecorr', True)
        self.skip_straylight = getattr(self, 'skip_straylight', True)
        self.skip_fringe = getattr(self, 'skip_fringe', True)
        self.skip_pathloss = getattr(self, 'skip_pathloss', True)
        self.skip_barshadow = getattr(self, 'skip_barshadow', True)
        self.skip_wfss_contam = getattr(self, 'skip_wfss_contam', True)
        self.skip_residual_fringe = getattr(self, 'skip_residual_fringe', True)
        self.skip_pixel_replace = getattr(self, 'skip_pixel_replace', True)
        self.skip_resample = getattr(self, 'skip_resample', True)
        self.skip_cube_build = getattr(self, 'skip_cube_build', True)

        # Diagnostics
        self.testing_S2 = getattr(self, 'testing_S2', False)
        self.hide_plots = getattr(self, 'hide_plots', True)
        self.verbose = getattr(self, 'verbose', True)

        # Project directory
        # Must be provided in the ECF
        self.topdir = getattr(self, 'topdir')

        # Directories relative to topdir
        self.inputdir = getattr(self, 'inputdir', 'Stage1')
        self.outputdir = getattr(self, 'outputdir', 'Stage2')

    def set_MIRI_defaults(self):
        '''Set Stage 2 specific defaults for MIRI.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial empty version setting defaults for MIRI.
        '''
        return

    def set_NIRCam_defaults(self):
        '''Set Stage 2 specific defaults for NIRCam.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial empty version setting defaults for NIRCam.
        '''
        self.tsgrism_extract_height = getattr(self, 'tsgrism_extract_height',
                                              None)
        return

    def set_NIRSpec_defaults(self):
        '''Set Stage 2 specific defaults for NIRSpec.

        Notes
        -----
        History:

        - 2024-07 Taylor J Bell
            Initial version setting defaults for NIRSpec.
        '''
        self.slit_y_low = getattr(self, 'slit_y_low', -1)
        self.slit_y_high = getattr(self, 'slit_y_high', 50)
        self.waverange_start = getattr(self, 'waverange_start', 6e-08)
        self.waverange_end = getattr(self, 'waverange_end', 6e-06)
        return

    def set_NIRISS_defaults(self):
        '''Set Stage 2 specific defaults for NIRISS.

        Notes
        -----
        History:

        - 2024-03 Taylor J Bell
            Initial empty version setting defaults for NIRISS.
        '''
        return
