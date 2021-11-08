
#! /usr/bin/env python

#This is based on the RampFitStep from the JWST pipeline, accessed Oct 2021
#adapted by Eva-Maria Ahrer & Aarynn Carter, Oct 2021

import numpy as np

from jwst.stpipe import Step
from jwst import datamodels

from stcal.ramp_fitting import ramp_fit
from jwst.datamodels import dqflags

from jwst.lib import reffile_utils  
from jwst.lib import pipe_utils

from eureka.S1_detector_processing.Eureka_ramp_fitting import mean_ramp_fit_single

#wanted to call the first three functions from ramp_fit_step but are not initialised with the jwst pipeline
#from jwst.ramp_fitting import ramp_fit_step

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

#do we need this? what is this?
BUFSIZE = 1024 * 300000  # 300Mb cache size for data section 

__all__ = ["RampFitStep"]

class Eureka_RampFitStep(Step):

    """
    This step is alternatively to the pipeline rampfitstep to
    determine the count rate for each pixel.
    """

    spec = """
        int_name = string(default='')
        save_opt = boolean(default=False) # Save optional output
        opt_name = string(default='')
        maximum_cores = option('none', 'quarter', 'half', 'all', default='none') # max number of processes to create
    """

    algorithm = 'differenced' #default
    weighting = 'optimal'  # Only weighting allowed for Build 7.1
    maximum_cores = 1 #default

    reference_file_types = ['readnoise', 'gain']

    def process(self, input):
        with datamodels.RampModel(input) as input_model:
            readnoise_filename = self.get_reference_file(input_model, 'readnoise')
            gain_filename = self.get_reference_file(input_model, 'gain')

            log.info('Using READNOISE reference file: %s', readnoise_filename)
            log.info('Using GAIN reference file: %s', gain_filename)

            with datamodels.ReadnoiseModel(readnoise_filename) as readnoise_model, \
                 datamodels.GainModel(gain_filename) as gain_model:

                # Try to retrieve the gain factor from the gain reference file.
                # If found, store it in the science model meta data, so that it's
                # available later in the gain_scale step, which avoids having to
                # load the gain ref file again in that step.
                if gain_model.meta.exposure.gain_factor is not None:
                    input_model.meta.exposure.gain_factor = gain_model.meta.exposure.gain_factor

                # Get gain arrays, subarrays if desired.
                frames_per_group = input_model.meta.exposure.nframes
                readnoise_2d, gain_2d = get_reference_file_subarrays(
                    input_model, readnoise_model, gain_model, frames_per_group)

            log.info('Using algorithm = %s' % self.algorithm)
            log.info('Using weighting = %s' % self.weighting)

            buffsize = ramp_fit.BUFSIZE
            if pipe_utils.is_tso(input_model) and hasattr(input_model, 'int_times'):
                input_model.int_times = input_model.int_times
            else:
                input_model.int_times = None
            
            if self.algorithm =='differenced':
                #replace this by differenced ramp fit

                image_info, integ_info, opt_info, gls_opt_model = ramp_fit.ramp_fit(input_model, buffsize, self.save_opt, readnoise_2d,\
                                                                                    gain_2d, self.algorithm, self.weighting,\
                                                                                    self.maximum_cores, dqflags.pixel)

        if image_info is not None:
            out_model = create_image_model(input_model, image_info)
            out_model.meta.bunit_data = 'DN/s'
            out_model.meta.bunit_err = 'DN/s'
            out_model.meta.cal_step.ramp_fit = 'COMPLETE'

        if integ_info is not None:
            int_model = create_integration_model(input_model, integ_info)
            int_model.meta.bunit_data = 'DN/s'
            int_model.meta.bunit_err = 'DN/s'
            int_model.meta.cal_step.ramp_fit = 'COMPLETE'

         
        return out_model, int_model


#### NOTE FOR FUTURE (11/05/2021)####
'''
Space telescope is currently changing the ramp fitting structure on Github
# The following functions:


create_optional_results_model()

Will *not* need to be directly included in this file, we can instead simply
import them from ramp_fitting.ramp_fit_step 

i.e. 

from ramp_fitting.ramp_fit_step import xxx

However, this has yet incorporated in the pypi repository, so can't do things straight away. 

'''

def get_reference_file_subarrays(model, readnoise_model, gain_model, nframes):
    """
    Get readnoise array for calculation of variance of noiseless ramps, and
    the gain array in case optimal weighting is to be done. The returned
    readnoise has been multiplied by the gain.
    Parameters
    ----------
    model : data model
        input data model, assumed to be of type RampModel
    readnoise_model : instance of data Model
        readnoise for all pixels
    gain_model : instance of gain Model
        gain for all pixels
    nframes : int
        number of frames averaged per group; from the NFRAMES keyword. Does
        not contain the groupgap.
    Returns
    -------
    readnoise_2d : float, 2D array
        readnoise subarray
    gain_2d : float, 2D array
        gain subarray
    """
    if reffile_utils.ref_matches_sci(model, gain_model):
        gain_2d = gain_model.data
    else:
        log.info('Extracting gain subarray to match science data')
        gain_2d = reffile_utils.get_subarray_data(model, gain_model)

    if reffile_utils.ref_matches_sci(model, readnoise_model):
        readnoise_2d = readnoise_model.data.copy()
    else:
        log.info('Extracting readnoise subarray to match science data')
        readnoise_2d = reffile_utils.get_subarray_data(model, readnoise_model)

    return readnoise_2d, gain_2d


def create_image_model(input_model, image_info):
    """
    Creates an ImageModel from the computed arrays from ramp_fit.
    Parameter
    ---------
    input_model: RampModel
        Input RampModel for which the output ImageModel is created.
    image_info: tuple
        The ramp fitting arrays needed for the ImageModel.
    Parameter
    ---------
    out_model: ImageModel
        The output ImageModel to be returned from the ramp fit step.
    """
    data, dq, var_poisson, var_rnoise, err = image_info

    # Create output datamodel
    out_model = datamodels.ImageModel(data.shape)

    # ... and add all keys from input
    out_model.update(input_model)

    # Populate with output arrays
    out_model.data = data
    out_model.dq = dq
    out_model.var_poisson = var_poisson
    out_model.var_rnoise = var_rnoise
    out_model.err = err

    return out_model


def create_integration_model(input_model, integ_info):
    """
    Creates an ImageModel from the computed arrays from ramp_fit.
    Parameter
    ---------
    input_model: RampModel
        Input RampModel for which the output CubeModel is created.
    integ_info: tuple
        The ramp fitting arrays needed for the CubeModel for each integration.
    Parameter
    ---------
    int_model: CubeModel
        The output CubeModel to be returned from the ramp fit step.
    """
    data, dq, var_poisson, var_rnoise, int_times, err = integ_info
    int_model = datamodels.CubeModel(
        data=np.zeros(data.shape, dtype=np.float32),
        dq=np.zeros(data.shape, dtype=np.uint32),
        var_poisson=np.zeros(data.shape, dtype=np.float32),
        var_rnoise=np.zeros(data.shape, dtype=np.float32),
        err=np.zeros(data.shape, dtype=np.float32))
    int_model.int_times = None
    int_model.update(input_model)  # ... and add all keys from input

    int_model.data = data
    int_model.dq = dq
    int_model.var_poisson = var_poisson
    int_model.var_rnoise = var_rnoise
    int_model.err = err
    int_model.int_times = int_times

    return int_model

############################################################
######## SEE NOTE ON LINE ~110 FOR ABOVE FUNCTIONS #########
############################################################


#######################################
######### CUSTOM FUNCTIONS ############
#######################################
"""
To adjust the pipeline to our needs, we can write functions to
replace specific functions from the massive ols_fit.py file
in the stcal package

OR

We can write entirely new files to replace ols_fit.py in it's
entirety. 

"""