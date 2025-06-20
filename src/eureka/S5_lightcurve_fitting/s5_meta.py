import numpy as np
try:
    import starry
    starry.config.quiet = True
    starry.config.lazy = True
except ModuleNotFoundError:
    # PyMC3 hasn't been installed
    pass

from ..lib.readECF import MetaClass


class S5MetaClass(MetaClass):
    '''A class to hold Eureka! S5 metadata.

    This class loads a Stage 5 Eureka! Control File (ecf) and lets you
    query the parameters and values.

    Notes
    -----
    History:

    - 2024-06 Taylor J Bell
        Made specific S5 class based on MetaClass
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

        - 2024-06 Taylor J Bell
            Initial version.
        '''
        super().__init__(folder, file, eventlabel, stage=5, **kwargs)

    def set_defaults(self):
        '''Set Stage 5 specific defaults for generic instruments.

        '''
        # Make sure the S3 expand parameter is defined
        # (to allow resuming from old analyses)
        self.expand = getattr(self, 'expand', 1)

        self.ncpu = getattr(self, 'ncpu', 4)

        # Joint fit of multiple white lightcurves?
        self.multwhite = getattr(self, 'multwhite', False)

        # Repeat Stage 5 for each of the aperture sizes run from Stage 3?
        self.allapers = getattr(self, 'allapers', False)

        if not self.allapers:
            # The user indicated in the ecf that they only want to consider one
            # aperture in which case the code will consider only the one which
            # made s4_meta. Alternatively, if S4 was run without allapers, S5
            # will already only consider that one
            self.spec_hw_range = [self.spec_hw, ]
            self.bg_hw_range = [self.bg_hw, ]
        else:
            self.spec_hw_range = getattr(self, 'spec_hw_range',
                                         [self.spec_hw,])
            self.bg_hw_range = getattr(self, 'bg_hw_range',
                                       [self.spec_hw,])
        # Make sure hw_range attributes are lists
        if not isinstance(self.spec_hw_range, (list, np.ndarray)):
            self.spec_hw_range = [self.spec_hw_range, ]
        if not isinstance(self.bg_hw_range, (list, np.ndarray)):
            self.bg_hw_range = [self.bg_hw_range, ]

        # Manual clipping in time
        self.manual_clip = getattr(self, 'manual_clip', None)

        # Fitted model details - must be passed in through the ECF
        self.fit_par = getattr(self, 'fit_par')
        self.fit_method = getattr(self, 'fit_method')
        self.run_myfuncs = getattr(self, 'run_myfuncs')

        # Fitted model details - can use some safe defaults
        self.num_planets = getattr(self, 'num_planets', 1)
        self.compute_ltt = getattr(self, 'compute_ltt', None)
        self.force_positivity = getattr(self, 'force_positivity', False)
        # The following is only relevant for the starry model
        self.mutualOccultations = getattr(self, 'mutualOccultations', True)

        # Use of modelled LD coefficients
        self.use_generate_ld = getattr(self, 'use_generate_ld', None)
        self.ld_file = getattr(self, 'ld_file', None)
        self.ld_file_white = getattr(self, 'ld_file_white', None)
        if not all([self.use_generate_ld is None, self.ld_file is None,
                    self.ld_file_white is None]):
            # Only set this parameter to True if relevant
            self.recenter_ld_prior = getattr(self, 'recenter_ld_prior', True)
        else:
            # Set this to False if not relevant
            self.recenter_ld_prior = getattr(self, 'recenter_ld_prior', False)

        # Use of modelled spot contrast coefficients
        self.spotcon_file = getattr(self, 'spotcon_file', None)
        self.spotcon_file_white = getattr(self, 'spotcon_file_white', None)
        if not all([self.spotcon_file is None,
                    self.spotcon_file_white is None]):
            # Only set this parameter to True if relevant
            self.recenter_spotcon_prior = getattr(
                self, 'recenter_spotcon_prior', True)
        else:
            # Default to False since it ends up being checked later
            self.recenter_spotcon_prior = False

        # Catwoman convergence-aiding parameters
        self.catwoman_fac = getattr(self, 'catwoman_fac', None)
        self.catwoman_max_err = getattr(self, 'catwoman_max_err', 1.0)

        # General fitter, fitparams CSV file to resume from
        self.old_fitparams = getattr(self, 'old_fitparams', None)

        # lsq inputs
        self.lsq_method = getattr(self, 'lsq_method', 'Powell')
        self.lsq_tol = getattr(self, 'lsq_tol', 1e-7)
        self.lsq_maxiter = getattr(self, 'lsq_maxiter', None)

        # emcee inputs
        self.old_chain = getattr(self, 'old_chain', None)
        self.lsq_first = getattr(self, 'lsq_first', False)
        if 'emcee' in self.fit_method:
            # Must be provided in the ECF if relevant
            self.run_nsteps = getattr(self, 'run_nsteps')
            self.run_nwalkers = getattr(self, 'run_nwalkers')
            self.run_nburn = getattr(self, 'run_nburn')

        # dynesty inputs
        self.run_nlive = getattr(self, 'run_nlive', 'min')
        self.run_bound = getattr(self, 'run_bound', 'multi')
        self.run_sample = getattr(self, 'run_sample', 'auto')
        self.run_tol = getattr(self, 'run_tol', 0.1)

        # PyMC3 NUTS sampler settings
        self.exoplanet_first = getattr(self, 'exoplanet_first', False)
        self.chains = getattr(self, 'chains', 3)
        self.target_accept = getattr(self, 'target_accept', 0.85)
        if 'nuts' in self.fit_method:
            # Must be provided in the ECF if relevant
            self.tune = getattr(self, 'tune')
            self.draws = getattr(self, 'draws')

        # Starry eclipse mapping pixel-sampling parameters
        self.pixelsampling = getattr(self, 'pixelsampling', False)
        self.oversample = getattr(self, 'oversample', 3)
        if self.pixelsampling:
            # Must be provided in the ECF if relevant
            self.ydeg = getattr(self, 'ydeg')
            # Compute the number of pixels used in sampling
            map = starry.Map(ydeg=self.ydeg)
            A = map.get_pixel_transforms(oversample=self.oversample)[3]
            self.npix = A.shape[1]
        else:
            self.ydeg = getattr(self, 'ydeg', None)
            self.npix = 0

        # GP inputs
        self.kernel_inputs = getattr(self, 'kernel_inputs', ['time'])
        self.kernel_class = getattr(self, 'kernel_class', ['Matern32'])
        self.GP_package = getattr(self, 'GP_package', 'celerite')
        self.useHODLR = getattr(self, 'useHODLR', False)

        # Diagnostics
        self.interp = getattr(self, 'interp', False)
        self.isplots_S5 = getattr(self, 'isplots_S5', 3)
        self.nbin_plot = getattr(self, 'nbin_plot', None)
        self.testing_S5 = getattr(self, 'testing_S5', False)
        self.testing_model = getattr(self, 'testing_model', False)
        self.hide_plots = getattr(self, 'hide_plots', True)
        self.verbose = getattr(self, 'verbose', True)

        # Project directory
        self.topdir = getattr(self, 'topdir')  # Must be provided in the ECF

        # Directories relative to topdir
        self.inputdir = getattr(self, 'inputdir', 'Stage4')
        if self.multwhite:
            # Must be provided in the ECF if relevant
            self.inputdirlist = getattr(self, 'inputdirlist')
        self.outputdir = getattr(self, 'outputdir', 'Stage5')
