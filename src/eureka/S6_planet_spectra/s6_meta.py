import numpy as np
from astropy import units

from ..lib.readECF import MetaClass


class S6MetaClass(MetaClass):
    '''A class to hold Eureka! S6 metadata.

    This class loads a Stage 6 Eureka! Control File (ecf) and lets you
    query the parameters and values.

    Notes
    -----
    History:

    - 2024-06 Taylor J Bell
        Made specific S6 class based on MetaClass
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
        super().__init__(folder, file, eventlabel, stage=6, **kwargs)

    def set_defaults(self):
        '''Set Stage 6 specific defaults for generic instruments.

        Notes
        -----
        History:

        - 2024-06 Taylor J Bell
            Initial version setting defaults for any instrument.
        '''
        # Make sure the S3 expand parameter is defined
        # (to allow resuming from old analyses)
        self.expand = getattr(self, 'expand', 1)

        # Repeat Stage 6 for each of the aperture sizes run from Stage 3?
        self.allapers = getattr(self, 'allapers', False)

        if not self.allapers:
            # The user indicated in the ecf that they only want to consider one
            # aperture in which case the code will consider only the one which
            # made s5_meta. Alternatively, if S4 or S5 were run without
            # allapers, S6 will already only consider that one
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

        # Which parameters are being plotted? Must be specified in ECF
        self.y_params = getattr(self, 'y_params')
        # The formatted string you want on the y-label
        self.y_labels = getattr(self, 'y_labels', None)
        # The formatted string for the units you want on the y-label
        self.y_label_units = getattr(self, 'y_label_units', None)
        # Convert to percent, ppm, etc. if requested
        self.y_scalar = getattr(self, 'y_scalar', 1)

        # Make sure these are lists, even if it's just one item
        if isinstance(self.y_params, str):
            self.y_params = [self.y_params]
        if self.y_labels is None:
            self.y_labels = [None for _ in self.y_params]
        elif isinstance(self.y_labels, str):
            self.y_labels = [self.y_labels]
        if self.y_label_units is None:
            self.y_label_units = [None for _ in self.y_params]
        elif isinstance(self.y_label_units, str):
            self.y_label_units = [self.y_label_units]
        if (isinstance(self.y_scalars, int) or
                isinstance(self.y_scalars, float)):
            self.y_scalars = [self.y_scalars]

        # Convert to the user-provided x-axis unit if needed
        self.x_unit = getattr(self, 'x_unit', 'um')
        # Get the x-unit as an astropy unit
        self.x_unit = getattr(units, self.x_unit)
        # Number of time steps used to sample phase variation
        # when computing the phase curve amplitude and offset
        self.pc_nstep = getattr(self, 'pc_nstep', 1000)
        # Sample spacing between independent MCMC steps
        self.pc_stepsize = getattr(self, 'pc_stepsize', 50)
        self.strings_stepsize = getattr(self, 'strings_stepsize', 50)
        # Harmonica strings angle (in degrees) to include in morning/evening
        # limb calculation. An angle of 60 degrees will span -30 to +30 degrees
        # for the morning limb.
        self.strings_angle = getattr(self, 'strings_angle', 60)

        # Tabulating parameters
        self.ncols = getattr(self, 'ncols', 4)

        # This section is relevant if isplots_S6>=3
        # If requested, can we make the scale_height version of the figure?
        self.has_fig6301reqs = np.all([hasattr(self, val) for val in
                                       ['planet_Teq', 'planet_mu',
                                        'planet_Rad', 'planet_Mass',
                                        'star_Rad', 'planet_R0']])

        # Parameters for also plotting a fitted/predicted model
        self.model_spectrum = getattr(self, 'model_spectrum', None)
        if self.model_spectrum is not None:
            self.model_x_unit = getattr(self, 'model_x_unit', 'um')
            self.model_y_scalar = getattr(self, 'model_y_scalar', 1)
            self.model_zorder = getattr(self, 'model_zorder', 0)
            self.model_delimiter = getattr(self, 'model_delimiter', None)
            # Must be specified if relevant
            self.model_y_param = getattr(self, 'model_y_param')

        # Some parameters for saving the outputs
        self.wave_units = getattr(self, 'wave_units', 'microns')
        self.time_units = getattr(self, 'time_units', 'BMJD_TDB')

        # Diagnostics
        self.isplots_S6 = getattr(self, 'isplots_S6', 5)
        self.hide_plots = getattr(self, 'hide_plots', True)

        # Project directory
        self.topdir = getattr(self, 'topdir')  # Must be provided in the ECF

        # Directories relative to topdir
        self.inputdir = getattr(self, 'inputdir', 'Stage5')
        self.outputdir = getattr(self, 'outputdir', 'Stage6')
