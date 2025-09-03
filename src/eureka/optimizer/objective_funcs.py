import numpy as np
import eureka.S1_detector_processing.s1_process as s1
import eureka.S2_calibrations.s2_calibrate as s2
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4
import shutil

"""
Eureka! Optimization Tools: Objective Functions for optimization of
targeted focus parameters in Light Curve Analysis

Description:
-----------
This function is structured to optimize the thresholding differences
across multiple stages. By adjusting
the focus parameter, based on the optimization variable `x`, the
function aims to minimize an objective that combines the
MAD of spectroscopic light curves, and MAD of binned white light curves.

The function integrates the effect of focus parameter across the stages,
processes the light curves, and subsequently computes the fitness value.

Parameters:
----------
x : list
    List of optimization variables. Here, it specifically adjusts the
    focus parameter for the light curve stages.

eventlabel : str
    A label or identifier for the event being analyzed.

s3_meta, s4_meta : objects
    Metadata objects for Stages 3 and 4, respectively. These get
    modified within the function based on the optimization variable `x`.

scaling_MAD_white, scaling_MAD_spec : float
    Scaling factors that are used to weight the respective components in
    the objective function. They ensure that each component's contribution
    to the fitness value is appropriately adjusted.

Outputs:
-------
fitness_value : float
    The computed objective value, which is a measure of the fitness of the
    given set of optimization variables `x`. Lower values are preferred,
    suggesting a better thresholding configuration.

Notes:
-----
- The function updates the focus parameter parameter in the metadata of
    Stages 3, 4, and 5 using the optimization variable `x[0]`.
- After adjusting the focus parameter, the function processes the light
    curves in each stage and computes the required metrics.
- The objective is a weighted sum of the
    MAD of spectroscopic light curves from Stage 4, and MAD of binned white
    light curves from Stage 4.
- The function also performs cleanup by removing directories related to
    each stage after processing.

Author: Reza Ashtari
Date: 08/22/2023
"""

def single(val, eventlabel, meta, s3_meta, s4_meta):

    # Set value of the variable to be optimized
    if hasattr(s3_meta, meta.opt_param_name):
        setattr(s3_meta, meta.opt_param_name, val)
    if hasattr(s4_meta, meta.opt_param_name):
        setattr(s4_meta, meta.opt_param_name, val)

    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
                                       s3_meta=s3_meta)

    if meta.delete_intermediate:
        shutil.rmtree(s3_meta.outputdir)
        shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        meta.scaling_MAD_spec * s4_meta.mad_s4 +
        meta.scaling_MAD_white * s4_meta.mad_s4_binned[0]
    )

    return fitness_value



def double(val, eventlabel, meta, s3_meta, s4_meta):

    # Set values of the two variables to be optimized
    param_names = meta.opt_param_name.split('__')
    assert len(param_names) == len(val) == 2, \
        "Expected two parameters for optimization."
    for p, v in zip(param_names, val):
        if hasattr(s3_meta, p):
            setattr(s3_meta, p, v)
        if hasattr(s4_meta, p):
            setattr(s4_meta, p, v)

    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
                                        s3_meta=s3_meta)

    if meta.delete_intermediate:
        shutil.rmtree(s3_meta.outputdir)
        shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        meta.scaling_MAD_spec * s4_meta.mad_s4 +
        meta.scaling_MAD_white * s4_meta.mad_s4_binned[0]
    )

    return fitness_value


# # Objective Function
# def jump_rejection_threshold_s1(x, eventlabel, s1_meta_GA, s2_meta_GA,
#                                 s3_meta, s4_meta,
#                                 scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 1
#     s1_meta_GA.jump_rejection_threshold = x[0]

#     s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
#     s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s1_meta.outputdir)
#     shutil.rmtree(s2_meta.outputdir)
#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def bg_deg_s1(x, eventlabel, s1_meta_GA, s2_meta_GA, s3_meta, s4_meta,
#               scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 1
#     s1_meta_GA.bg_deg = x[0]

#     s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
#     s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s1_meta.outputdir)
#     shutil.rmtree(s2_meta.outputdir)
#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def bg_method_s1(x, eventlabel, s1_meta_GA, s2_meta_GA, s3_meta, s4_meta,
#                  scaling_MAD_white, scaling_MAD_spec):

#     def remove_apostrophes(s):
#         return s.replace("'", "")

#     # Set bg_method in metadata for each stage based on the input string
#     bg_method_value = remove_apostrophes(x[0])  # Assumes x[0] is string
#     # bg_method_value = x[0]  # Assuming x[0] is a string, e.g. 'std'
#     s1_meta_GA.bg_method = bg_method_value

#     s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
#     s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s1_meta.outputdir)
#     shutil.rmtree(s2_meta.outputdir)
#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     # Calculate the fitness value based on the specified scaling factors
#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def p3thresh_s1(x, eventlabel, s1_meta_GA, s2_meta_GA, s3_meta, s4_meta,
#                 scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 1
#     s1_meta_GA.p3thresh = x[0]

#     s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
#     s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s1_meta.outputdir)
#     shutil.rmtree(s2_meta.outputdir)
#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def window_len_s1(x, eventlabel, s1_meta_GA, s2_meta_GA, s3_meta,
#                   s4_meta, scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 1
#     s1_meta_GA.window_len = x[0]

#     s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
#     s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s1_meta.outputdir)
#     shutil.rmtree(s2_meta.outputdir)
#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def expand_mask_s1(x, eventlabel, s1_meta_GA, s2_meta_GA, s3_meta,
#                    s4_meta, scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 1
#     s1_meta_GA.expand_mask = x[0]

#     s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
#     s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s1_meta.outputdir)
#     shutil.rmtree(s2_meta.outputdir)
#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def xwindow_crop(x, eventlabel, ev, pixel_wave_min, pixel_wave_max, s3_meta,
#                  s4_meta):

#     try:

#         # Define Variables to be optimized
#         # Stage 3
#         s3_meta.xwindow = [ev.xwindow[0] + x[0], ev.xwindow[1] - x[0]]

#         # Check conditions for xwindow values

#         # if s3_meta.xwindow[0] > pixel_wave_min-3 or \
#         #    s3_meta.xwindow[1] < pixel_wave_max+3:

#         if s3_meta.xwindow[0] > pixel_wave_min or \
#            s3_meta.xwindow[1] < pixel_wave_max:

#             raise ValueError("xwindow boundaries violated")

#         s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#         s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                            s3_meta=s3_meta)
#         # shutil.rmtree(s3_meta.outputdir)
#         # shutil.rmtree(s4_meta.outputdir)

#         fitness_value = (
#             0.05 * s3_meta.mad_s3 +
#             0.5 * s4_meta.mad_s4 +
#             1.0 * s4_meta.mad_s4_binned[0]
#         )

#         return fitness_value

#     except:

#         fitness_value = float('inf')

#         return fitness_value

#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)


# def ywindow_crop(x, eventlabel, ev, s3_meta, s4_meta):

#     try:

#         # Define Variables to be optimized
#         # Stage 3
#         s3_meta.ywindow = [ev.ywindow[0] + x[0], ev.ywindow[1] - x[0]]

#         # Check conditions for ywindow values --> Maybe Later, Gator

#         s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#         s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                            s3_meta=s3_meta)
#         # shutil.rmtree(s3_meta.outputdir)
#         # shutil.rmtree(s4_meta.outputdir)

#         fitness_value = (
#             0.05 * s3_meta.mad_s3 +
#             0.5 * s4_meta.mad_s4 +
#             1.0 * s4_meta.mad_s4_binned[0]
#         )

#         return fitness_value

#     except:

#         fitness_value = float('inf')

#         return fitness_value

#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)


# def dqmask(x, eventlabel, s2_inputdir, s3_meta, s4_meta,
#            scaling_MAD_white, scaling_MAD_spec):

#     # Convert the optimization variable to boolean for dqmask
#     dqmask_value = bool(x[0])

#     # Update dqmask in metadata for each stage
#     s3_meta.dqmask = dqmask_value
#     s4_meta.dqmask = dqmask_value

#     s3_meta.inputdir = s2_inputdir

#     # Perform the light curve processing for each stage
#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     # Clean up the directories for each stage
#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     # Calculate the fitness value based on the specified scaling factors
#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def expand(x, eventlabel, s2_inputdir, s3_meta, s4_meta,
#            scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 3
#     s3_meta.expand = x[0]
#     # Stage 4
#     s4_meta.expand = x[0]

#     s3_meta.inputdir = s2_inputdir

#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def bg_thresh(x, eventlabel, s2_inputdir, s3_meta, s4_meta,
#               scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 3
#     s3_meta.bgthresh = x[0]
#     # Stage 4
#     s4_meta.bgthresh = x[0]

#     s3_meta.inputdir = s2_inputdir

#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def bg_deg(x, eventlabel, s2_inputdir, s3_meta, s4_meta,
#            scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 3
#     s3_meta.bg_deg = x[0]
#     # Stage 4
#     s4_meta.bg_deg = x[0]

#     s3_meta.inputdir = s2_inputdir

#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def bg_method(x, eventlabel, s2_inputdir, s3_meta, s4_meta,
#               scaling_MAD_white, scaling_MAD_spec):

#     def remove_apostrophes(s):
#         return s.replace("'", "")

#     # Set bg_method in metadata for each stage based on the input string
#     bg_method_value = remove_apostrophes(x[0])  # Assuming x[0] is a string
#     # bg_method_value = x[0]  # Assuming x[0] is a string, e.g. 'std'
#     s3_meta.bg_method = bg_method_value
#     s4_meta.bg_method = bg_method_value

#     s3_meta.inputdir = s2_inputdir

#     # Perform the light curve processing for each stage
#     s3_spec, s3_meta = s3.reduce(eventlabel,
#                                  input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel,
#                                        input_meta=s4_meta, s3_meta=s3_meta)

#     # Clean up the directories for each stage
#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     # Calculate the fitness value based on the specified scaling factors
#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def p3thresh(x, eventlabel, s2_inputdir, s3_meta, s4_meta,
#              scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 3
#     s3_meta.p3thresh = x[0]
#     # Stage 4
#     s4_meta.p3thresh = x[0]

#     s3_meta.inputdir = s2_inputdir

#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def median_thresh(x, eventlabel, s2_inputdir, s3_meta,
#                   s4_meta, scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 3
#     s3_meta.median_thresh = x[0]
#     # Stage 4
#     s4_meta.median_thresh = x[0]

#     s3_meta.inputdir = s2_inputdir

#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# # Objective Function
# def window_len(x, eventlabel, s2_inputdir, s3_meta, s4_meta,
#                scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 3
#     s3_meta.window_len = x[0]
#     # Stage 4
#     s4_meta.window_len = x[0]

#     s3_meta.inputdir = s2_inputdir

#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# # Objective Function
# def p5thresh(x, eventlabel, s2_inputdir, s3_meta, s4_meta,
#              scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 3
#     s3_meta.p5thresh = x[0]
#     # Stage 4
#     s4_meta.p5thresh = x[0]

#     s3_meta.inputdir = s2_inputdir

#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# # Objective Function
# def p7thresh(x, eventlabel, s2_inputdir, s3_meta, s4_meta,
#              scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 3
#     s3_meta.p7thresh = x[0]
#     # Stage 4
#     s4_meta.p7thresh = x[0]

#     s3_meta.inputdir = s2_inputdir

#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def sigma(x, eventlabel, last_s3_meta_outputdir, s3_meta, s4_meta,
#           scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 4
#     s4_meta.sigma = x[0]

#     s4_meta.inputdir = last_s3_meta_outputdir
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value


# def box_width(x, eventlabel, last_s3_meta_outputdir, s3_meta, s4_meta,
#               scaling_MAD_white, scaling_MAD_spec):

#     # Define Variables to be optimized
#     # Stage 4
#     s4_meta.box_width = x[0]

#     s4_meta.inputdir = last_s3_meta_outputdir
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white * s4_meta.mad_s4_binned[0]
#     )

#     return fitness_value
