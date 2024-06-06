#! /usr/bin/env python

# This is based on the superbias_step from the JWST pipeline v1.8.0
# adapted by Kevin Stevenson, Feb 2023


from jwst.stpipe import Step
from jwst import datamodels
# from jwst.superbias import bias_sub
from . import bias_sub
# import numpy as np
# from functools import partial
# import warnings
import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

__all__ = ["Eureka_SuperBiasStep"]


class Eureka_SuperBiasStep(Step):
    """This step is an alternative to the jwst pipeline superbias.
    Performs super-bias subtraction by subtracting scaled super-bias
    reference data from the input science data model at the group level.
    """

    class_alias = "superbias"

    spec = """

    """

    reference_file_types = ['superbias']

    def process(self, input):

        # Open the input data model
        with datamodels.RampModel(input) as input_model:

            # Get the name of the superbias reference file to use
            self.bias_name = self.get_reference_file(input_model, 'superbias')
            self.log.info('Using SUPERBIAS reference file %s', self.bias_name)

            # Check for a valid reference file
            if self.bias_name == 'N/A':
                self.log.warning('No SUPERBIAS reference file found')
                self.log.warning('Superbias step will be skipped')
                result = input_model.copy()
                result.meta.cal_step.superbias = 'SKIPPED'
                return result

            # Open the superbias ref file data model
            bias_model = datamodels.SuperBiasModel(self.bias_name)

            # Do the bias subtraction
            result = bias_sub.do_correction(input_model, bias_model,
                                            self.s1_meta, self.s1_log)

            # Close the superbias reference file model and
            # set the step status to complete
            bias_model.close()
            result.meta.cal_step.superbias = 'COMPLETE'

        return result
