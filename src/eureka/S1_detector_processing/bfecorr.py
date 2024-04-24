#! /usr/bin/env python

# This is based on the RampFitStep from the JWST pipeline, accessed Oct 2021
# adapted by Eva-Maria Ahrer & Aarynn Carter, Oct 2021

import numpy as np

from scipy.signal import convolve2d

from jwst.stpipe import Step
from jwst import datamodels

import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

__all__ = ["Eureka_BfeCorrStep"]


class Eureka_BfeCorrStep(Step):
    """This step is an experimental brighter-fatter effect correction.
    """

    spec = """
        int_name = string(default='')
        save_opt = boolean(default=False) # Save optional output
        opt_name = string(default='')
        maximum_cores = option('none', 'quarter', 'half', 'all', \
default='none') # max number of processes to create
    """

    reference_file_types = []

    def process(self, input):

        # Open the input data model
        with datamodels.RampModel(input) as input_model:
            if not hasattr(self.s1_meta, 'bfe_alpha'):
                self.s1_meta.bfe_alpha = 0.15
            if not hasattr(self.s1_meta, 'bfe_beta'):
                self.s1_meta.bfe_beta = 0.03

            self.log.info('Performing BFE correction step, assuming '
                          f'alpha={self.s1_meta.bfe_alpha}, '
                          f'beta={self.s1_meta.bfe_beta}')

            result = input_model

            # Compute the coefficients
            badpix = result.groupdq % 2 == 1
            data = np.ma.masked_where(badpix, result.data)
            ds_algo_vals = []
            for frame in range(result.data.shape[1]):
                medianFrame = np.ma.median(data[:, frame], axis=0)
                ds_algo_vals.append(ds_algo(medianFrame,
                                            alpha=self.s1_meta.bfe_alpha,
                                            beta=self.s1_meta.bfe_beta))
            ds_algo_vals = np.array(ds_algo_vals)

            # Apply the coefficients
            result.data += ds_algo_vals

        return result


def ds_algo(image, alpha=0.15, beta=0.03):
    kernel = np.zeros_like(image)
    
    ycent = kernel.shape[0]//2-1
    xcent = kernel.shape[1]//2-1
    
    # Central pixel (the pixel itself)
    kernel[ycent,xcent] = 0
    
    # Pixels that share sides
    kernel[ycent+1,xcent] = alpha # Same column (moving in spectral direction)
    kernel[ycent-1,xcent] = alpha # Same column (moving in spectral direction)
    kernel[ycent,xcent+1] = alpha # Same row (moving in spatial direction)
    kernel[ycent,xcent-1] = alpha # Same row (moving in spatial direction)
    
    # Pixels that share corners
    kernel[ycent+1,xcent+1] = beta
    kernel[ycent+1,xcent-1] = beta
    kernel[ycent-1,xcent+1] = beta
    kernel[ycent-1,xcent-1] = beta

    # Apply the 2D convolution
    ds_values = convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0.)

    return ds_values
