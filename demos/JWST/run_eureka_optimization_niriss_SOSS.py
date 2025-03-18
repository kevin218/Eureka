import os
import numpy as np
import pandas as pd
import eureka.lib.plots
import eureka.S1_detector_processing.s1_process as s1
import eureka.S2_calibrations.s2_calibrate as s2
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4

import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import astropy.io.fits as pf

import importlib, shutil, re
import pickle

# import json
from datetime import datetime

from eureka.lib import optimizers
from eureka.lib import objective_funcs

from eureka.S1_detector_processing.s1_meta import S1MetaClass
from eureka.S2_calibrations.s2_meta import S2MetaClass
from eureka.S3_data_reduction.s3_meta import S3MetaClass
from eureka.S4_generate_lightcurves.s4_meta import S4MetaClass


"""
Eureka! Optimization for JWST NIRISS SOSS Data
---------------------------------------------

Description:
This script is designed to optimize the stage 1-4 ECF parameters for JWST NIRISS SOSS observations. 

Inputs:
- Loaded from an input text file, 'optimizer_inputs_niriss_SOSS.txt'.
- Parameters not specified for optimization in the optimizer input text file will assume ECF values by default.

Outputs:
- Optimized parameter values saved in the "best" dictionary.
- Plot of optimization fitness values.
- Optimization results, printing the best ECF values and fitness scores (WLC MAD, 2D MAD, etc.).

Author: Reza Ashtari
Date: 11/22/2024
"""


if __name__ == "__main__":

    ## STEP 0 - Load optimizer parameters from input .txt file
    optimizer_params = optimizers.read_inputs("optimizer_inputs_niriss_SOSS.txt")

    # # Set each parameter as a local variable
    # for k, v in optimizer_params.items():
    #     locals()[k] = v

    # Set each parameter as a local variable (Explicit)
    eventlabel = optimizer_params["eventlabel"]
    order = optimizer_params["order"]
    xwindow_selection = optimizer_params["xwindow_selection"]
    ywindow_selection = optimizer_params["ywindow_selection"]
    spec_hw_selection = optimizer_params["spec_hw_selection"]
    bg_hw_selection = optimizer_params["bg_hw_selection"]
    optimizer = optimizer_params["optimizer"]
    target_fitness = optimizer_params["target_fitness"]
    scaling_MAD_spec = optimizer_params["scaling_MAD_spec"]
    scaling_MAD_white = optimizer_params["scaling_MAD_white"]

    bounds_jump_rejection_threshold_s1 = optimizer_params[
        "bounds_jump_rejection_threshold_s1"
    ]
    if bounds_jump_rejection_threshold_s1 == "auto":
        bounds_jump_rejection_threshold_s1 = [4, 12]

    bounds_dqmask = optimizer_params["bounds_dqmask"]
    if bounds_dqmask == "auto":
        bounds_dqmask = [True, False]

    bounds_bg_thresh = optimizer_params["bounds_bg_thresh"]
    if bounds_bg_thresh == "auto":
        bounds_bg_thresh = [3, 7]

    bounds_bg_hw = optimizer_params["bounds_bg_hw"]
    if bounds_bg_hw == "auto":
        bounds_bg_hw = [17, 27]

    bounds_spec_hw = optimizer_params["bounds_spec_hw"]
    if bounds_spec_hw == "auto":
        bounds_spec_hw = [12, 22]

    # bounds_bg_deg = optimizer_params['bounds_bg_deg']
    # if bounds_bg_deg == 'auto':
    #     bounds_bg_deg = [0, 1]   # or none! <-- not as important for AI-ML, optimizing from S3 on

    bounds_bg_method = optimizer_params["bounds_bg_method"]
    if bounds_bg_method == "auto":
        bounds_bg_method = ["std", "mean", "median"]  # <-- median doesnt seem to work

    bounds_p3thresh = optimizer_params["bounds_p3thresh"]
    if bounds_p3thresh == "auto":
        bounds_p3thresh = [3, 15]

    bounds_median_thresh = optimizer_params["bounds_median_thresh"]
    if bounds_median_thresh == "auto":
        bounds_median_thresh = [3, 7]

    bounds_window_len = optimizer_params["bounds_window_len"]
    if bounds_window_len == "auto":
        bounds_window_len = [1, 21]

    bounds_p7thresh = optimizer_params["bounds_p7thresh"]
    if bounds_p7thresh == "auto":
        bounds_p7thresh = [5, 15]

    bounds_expand = optimizer_params["bounds_expand"]
    if bounds_expand == "auto":
        bounds_expand = [1, 5]

    bounds_sigma = optimizer_params["bounds_sigma"]
    if bounds_sigma == "auto":
        bounds_sigma = [3, 25]

    bounds_box_width = optimizer_params["bounds_box_width"]
    if bounds_box_width == "auto":
        bounds_box_width = [10, 50]

    outputdir_optimization = optimizer_params["outputdir_optimization"]
    loc_sci = optimizer_params["loc_sci"]

    # # Set conditional inputs
    # if xwindow_selection == 'manual':
    #     skip_xwindow_crop = True
    # if ywindow_selection == 'manual':
    #     skip_ywindow_crop = True

    ## STEP 0 - Create "best" Dictionary to save optimized values ##
    best = {}

    # Create dictionaries to keep track of optimization metrics
    history_MAD_white = {}
    history_MAD_spec = {}
    history_MAD_chi2red = {}
    history_fitness_score = {}

    ## STEP 0 - Define Bounds for Parametric and Genetic Optimization ##

    # Define bounds for inputs used in Genetic Algorithm
    min_bounds = np.array(
        [
            bounds_jump_rejection_threshold_s1[0],
            bounds_dqmask[0],
            bounds_expand[0],
            bounds_bg_thresh[0],
            bounds_bg_hw[0],
            bounds_spec_hw[0],
            bounds_bg_method[0],
            bounds_p3thresh[0],
            bounds_median_thresh[0],
            bounds_window_len[0],
            bounds_p7thresh[0],
            bounds_sigma[0],
            bounds_box_width[0],
        ]
    )

    max_bounds = np.array(
        [
            bounds_jump_rejection_threshold_s1[1],
            bounds_dqmask[1],
            bounds_expand[1],
            bounds_bg_thresh[1],
            bounds_bg_hw[1],
            bounds_spec_hw[1],
            bounds_bg_method[1],
            bounds_p3thresh[1],
            bounds_median_thresh[1],
            bounds_window_len[1],
            bounds_p7thresh[1],
            bounds_sigma[1],
            bounds_box_width[1],
        ]
    )

    ## STEP 0 - Load ECF Files ##

    eureka.lib.plots.set_rc(style="eureka", usetex=False, filetype=".png")

    ecf_path = "." + os.sep

    # Load Eureka! control file and store values in Event object
    s1_ecffile = "S1_" + eventlabel + ".ecf"
    s1_meta_GA = S1MetaClass(ecf_path, s1_ecffile, eventlabel)

    s2_ecffile = "S2_" + eventlabel + ".ecf"
    s2_meta_GA = S2MetaClass(ecf_path, s2_ecffile, eventlabel)

    s3_ecffile = "S3_" + eventlabel + ".ecf"
    s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)

    s4_ecffile = "S4_" + eventlabel + ".ecf"
    s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

    # Set xwindow and ywindow values for auto vs. manual box extraction
    if xwindow_selection == "auto":
        s3_meta_GA.xwindow = [6, 2043]  # NIRISS-SOSS

    # # Set src_ypos for auto vs. manual box extraction
    # if order == 1:
    #     s3_meta_GA.order = [1]
    #     s3_meta_GA.src_ypos = [35]
    #     s4_meta_GA.s4_order = 1
    #     s4_meta_GA.wave_min = 0.86
    #     s4_meta_GA.wave_max = 2.8

    # if order == 2:
    #     s3_meta_GA.order = [2]
    #     s3_meta_GA.src_ypos = [90]
    #     s4_meta_GA.s4_order = 2
    #     s4_meta_GA.wave_min = 0.63
    #     s4_meta_GA.wave_max = 1.11

    best["xwindow_LB"] = s3_meta_GA.xwindow[0]
    best["xwindow_UB"] = s3_meta_GA.xwindow[1]

    if ywindow_selection == "auto":
        s3_meta_GA.ywindow = [1, 250]  # NIRISS-SOSS
        best["ywindow_LB"] = s3_meta_GA.ywindow[0]
        best["ywindow_UB"] = s3_meta_GA.ywindow[1]

    # Set spec_hw & bg_hw values for auto vs. manual selection methods
    if spec_hw_selection == "auto":
        s3_meta_GA.spec_hw = 17  # NIRISS-SOSS
    if bg_hw_selection == "auto":
        s3_meta_GA.bg_hw = 22  # NIRISS-SOSS

    ## STEP 0 - Initial Run ##

    print("Initial Run.")

    s3_meta_GA.turbo_optimizer = False
    s4_meta_GA.nspecchan = 1

    if __name__ == "__main__":
        s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
        s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)

        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
        s4_spec, s4_lc, s4_meta = s4.genlc(
            eventlabel, input_meta=s4_meta_GA, s3_meta=s3_meta
        )

        initial_fitness_value = (
            scaling_MAD_spec * s4_meta.mad_s4
            + scaling_MAD_white
            * (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
        )

        history_fitness_score["initial_run"] = initial_fitness_value

    ## STEP 1.0 - Parametric Sweep of "jump_rejection_threshold_s1" ##

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s1_meta_GA = S1MetaClass(ecf_path, s1_ecffile, eventlabel)
        s2_meta_GA = S2MetaClass(ecf_path, s2_ecffile, eventlabel)
        s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
        s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

        # Use 1 spectroscopic channel during optimization
        s4_meta_GA.nspecchan = 1

        # Perform parametric sweep
        best_param, best_fitness_value = optimizers.parametric_sweep_S1(
            objective_funcs.jump_rejection_threshold_s1,
            bounds_jump_rejection_threshold_s1,
            eventlabel,
            s1_meta_GA,
            s2_meta_GA,
            s3_meta_GA,
            s4_meta_GA,
            scaling_MAD_white,
            scaling_MAD_spec,
        )

        best["jump_rejection_threshold_s1"] = best_param[0]

        print("Best parameters: ", best_param[0])
        print("Best fitness: ", best_fitness_value)

        history_fitness_score["jump_rejection_threshold_s1"] = best_fitness_value

    ## STEP 1.0 - Run Stages 1 and 2 once and store Stage 2 directory to avoid re-running Stages 1 and 2 during Stage 3 optimization (Saves Time)

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s1_meta_GA = S1MetaClass(ecf_path, s1_ecffile, eventlabel)
        s2_meta_GA = S2MetaClass(ecf_path, s2_ecffile, eventlabel)

        # Setup Meta / Define Initial Population
        # Stage 1
        s1_meta_GA.jump_rejection_threshold = best["jump_rejection_threshold_s1"]

        s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
        s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
        last_s2_meta_outputdir = s2_meta.outputdir

    ## STEP 1.0 - Parametric Sweep of "dqmask" ##

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
        s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

        # Use 1 spectroscopic channel during optimization
        s4_meta_GA.nspecchan = 1

        # Perform parametric sweep
        best_param, best_fitness_value = optimizers.parametric_sweep_dqmask(
            objective_funcs.dqmask,
            bounds_dqmask,
            eventlabel,
            last_s2_meta_outputdir,
            s3_meta_GA,
            s4_meta_GA,
            scaling_MAD_white,
            scaling_MAD_spec,
        )

        best["dqmask"] = bool(best_param)

        print("Best parameters: ", best_param)
        print("Best fitness: ", best_fitness_value)

        history_fitness_score["dqmask"] = best_fitness_value

    ## STEP 1.0 - Parametric Sweep of "bg_thresh" ##

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
        s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

        # Use 1 spectroscopic channel during optimization
        s4_meta_GA.nspecchan = 1

        # Setup Meta / Define Initial Population
        # Stage 3
        s3_meta_GA.dqmask = best["dqmask"]

        # Stage 4
        s4_meta_GA.dqmask = best["dqmask"]

        # Perform parametric sweep
        best_param, best_fitness_value = optimizers.parametric_sweep_S3(
            objective_funcs.bg_thresh,
            bounds_bg_thresh,
            eventlabel,
            last_s2_meta_outputdir,
            s3_meta_GA,
            s4_meta_GA,
            scaling_MAD_white,
            scaling_MAD_spec,
        )

        best["bg_thresh"] = [best_param[0], best_param[0]]

        print("Best parameters: ", [best_param[0], best_param[0]])
        print("Best fitness: ", best_fitness_value)

        history_fitness_score["bg_thresh"] = best_fitness_value

    ## STEP 1.0 - Parametric Sweep of "bg_hw" & "spec_hw" ##

    # if optimizer == "parametric" or optimizer == "genetic":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
        s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

        # Use 1 spectroscopic channel during optimization
        s4_meta_GA.nspecchan = 1

        # Set spec_hw & bg_hw values for auto vs. manual selection methods
        if spec_hw_selection == "auto":
            s3_meta_GA.spec_hw = 17  # NIRISS-SOSS
        if bg_hw_selection == "auto":
            s3_meta_GA.bg_hw = 22  # NIRISS-SOSS

        # Setup Meta / Define Initial Population
        # Stage 3
        s3_meta_GA.dqmask = best["dqmask"]
        s3_meta_GA.bg_thresh = best["bg_thresh"]

        # Stage 4
        s4_meta_GA.dqmask = best["dqmask"]
        s4_meta_GA.bg_thresh = best["bg_thresh"]

        # Perform parametric sweep
        best_params, best_fitness_value = optimizers.parametric_sweep_double(
            objective_funcs.bg_hw_spec_hw,
            bounds_bg_hw,
            bounds_spec_hw,
            eventlabel,
            last_s2_meta_outputdir,
            s3_meta_GA,
            s4_meta_GA,
            scaling_MAD_white,
            scaling_MAD_spec,
        )

        # Save Results in "best" Dictionary
        best["bg_hw"] = best_params[0]
        best["spec_hw"] = best_params[1]

        # Print Results of Parametric Sweep
        print("Best parameters: ", best_params)
        print("Best fitness: ", best_fitness_value)

        history_fitness_score["spec_hw_bg_hw"] = best_fitness_value

    ## STEP 1.0 - Parametric Sweep of "bg_deg" ##

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        best["bg_deg"] = 0

    ## STEP 1.0 - Parametric Sweep of "bg_method" ##

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
        s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

        # Use 1 spectroscopic channel during optimization
        s4_meta_GA.nspecchan = 1

        # Setup Meta / Define Initial Population
        # Stage 3
        s3_meta_GA.dqmask = best["dqmask"]
        s3_meta_GA.bg_thresh = best["bg_thresh"]
        s3_meta_GA.bg_hw = best["bg_hw"]
        s3_meta_GA.spec_hw = best["spec_hw"]
        s3_meta_GA.bg_deg = best["bg_deg"]

        # Stage 4
        s4_meta_GA.dqmask = best["dqmask"]
        s4_meta_GA.bg_thresh = best["bg_thresh"]
        s4_meta_GA.bg_hw = best["bg_hw"]
        s4_meta_GA.spec_hw = best["spec_hw"]
        s4_meta_GA.bg_deg = best["bg_deg"]

        # Perform parametric sweep
        best_param, best_fitness_value = optimizers.parametric_sweep_bg_method(
            objective_funcs.bg_method,
            bounds_bg_method,
            eventlabel,
            last_s2_meta_outputdir,
            s3_meta_GA,
            s4_meta_GA,
            scaling_MAD_white,
            scaling_MAD_spec,
        )

        best["bg_method"] = best_param

        print("Best parameters: ", best_param)
        print("Best fitness: ", best_fitness_value)

        history_fitness_score["bg_method"] = best_fitness_value

    ## STEP 1.0 - Parametric Sweep of "p3thresh" ##

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
        s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

        # Use 1 spectroscopic channel during optimization
        s4_meta_GA.nspecchan = 1

        # Setup Meta / Define Initial Population
        # Stage 3
        s3_meta_GA.dqmask = best["dqmask"]
        s3_meta_GA.bg_thresh = best["bg_thresh"]
        s3_meta_GA.bg_hw = best["bg_hw"]
        s3_meta_GA.spec_hw = best["spec_hw"]
        s3_meta_GA.bg_deg = best["bg_deg"]
        s3_meta_GA.bg_method = best["bg_method"]

        # Stage 4
        s4_meta_GA.dqmask = best["dqmask"]
        s4_meta_GA.bg_thresh = best["bg_thresh"]
        s4_meta_GA.bg_hw = best["bg_hw"]
        s4_meta_GA.spec_hw = best["spec_hw"]
        s4_meta_GA.bg_deg = best["bg_deg"]
        s4_meta_GA.bg_method = best["bg_method"]

        # Perform parametric sweep
        best_param, best_fitness_value = optimizers.parametric_sweep_S3(
            objective_funcs.p3thresh,
            bounds_p3thresh,
            eventlabel,
            last_s2_meta_outputdir,
            s3_meta_GA,
            s4_meta_GA,
            scaling_MAD_white,
            scaling_MAD_spec,
        )

        best["p3thresh"] = best_param[0]

        print("Best parameters: ", best_param[0])
        print("Best fitness: ", best_fitness_value)

        history_fitness_score["p3thresh"] = best_fitness_value

    ## STEP 1.0 - Parametric Sweep of "median_thresh" ##

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
        s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

        # Use 1 spectroscopic channel during optimization
        s4_meta_GA.nspecchan = 1

        # Setup Meta / Define Initial Population
        # Stage 3
        s3_meta_GA.dqmask = best["dqmask"]
        s3_meta_GA.bg_thresh = best["bg_thresh"]
        s3_meta_GA.bg_hw = best["bg_hw"]
        s3_meta_GA.spec_hw = best["spec_hw"]
        s3_meta_GA.bg_deg = best["bg_deg"]
        s3_meta_GA.bg_method = best["bg_method"]
        s3_meta_GA.p3thresh = best["p3thresh"]

        # Stage 4
        s4_meta_GA.dqmask = best["dqmask"]
        s4_meta_GA.bg_thresh = best["bg_thresh"]
        s4_meta_GA.bg_hw = best["bg_hw"]
        s4_meta_GA.spec_hw = best["spec_hw"]
        s4_meta_GA.bg_deg = best["bg_deg"]
        s4_meta_GA.bg_method = best["bg_method"]
        s4_meta_GA.p3thresh = best["p3thresh"]

        # Perform parametric sweep
        best_param, best_fitness_value = optimizers.parametric_sweep_S3(
            objective_funcs.median_thresh,
            bounds_median_thresh,
            eventlabel,
            last_s2_meta_outputdir,
            s3_meta_GA,
            s4_meta_GA,
            scaling_MAD_white,
            scaling_MAD_spec,
        )

        best["median_thresh"] = best_param[0]

        print("Best parameters: ", best_param[0])
        print("Best fitness: ", best_fitness_value)

        history_fitness_score["median_thresh"] = best_fitness_value

    ## STEP 1.0 - Parametric Sweep of "window_len" ##

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
        s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

        # Use 1 spectroscopic channel during optimization
        s4_meta_GA.nspecchan = 1

        # Setup Meta / Define Initial Population
        # Stage 3
        s3_meta_GA.dqmask = best["dqmask"]
        s3_meta_GA.bg_thresh = best["bg_thresh"]
        s3_meta_GA.bg_hw = best["bg_hw"]
        s3_meta_GA.spec_hw = best["spec_hw"]
        s3_meta_GA.bg_deg = best["bg_deg"]
        s3_meta_GA.bg_method = best["bg_method"]
        s3_meta_GA.p3thresh = best["p3thresh"]
        s3_meta_GA.median_thresh = best["median_thresh"]

        # Stage 4
        s4_meta_GA.dqmask = best["dqmask"]
        s4_meta_GA.bg_thresh = best["bg_thresh"]
        s4_meta_GA.bg_hw = best["bg_hw"]
        s4_meta_GA.spec_hw = best["spec_hw"]
        s4_meta_GA.bg_deg = best["bg_deg"]
        s4_meta_GA.bg_method = best["bg_method"]
        s4_meta_GA.p3thresh = best["p3thresh"]
        s4_meta_GA.median_thresh = best["median_thresh"]

        # Perform parametric sweep for odd numbers
        best_param, best_fitness_value = optimizers.parametric_sweep_odd(
            objective_funcs.window_len,
            bounds_window_len,
            eventlabel,
            last_s2_meta_outputdir,
            s3_meta_GA,
            s4_meta_GA,
            scaling_MAD_white,
            scaling_MAD_spec,
        )

        best["window_len"] = best_param[0]

        print("Best parameters: ", best_param[0])
        print("Best fitness: ", best_fitness_value)

        history_fitness_score["window_len"] = best_fitness_value

    ## STEP 1.0 - Parametric Sweep of "p7thresh" ##

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
        s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

        # Use 1 spectroscopic channel during optimization
        s4_meta_GA.nspecchan = 1

        # Setup Meta / Define Initial Population
        # Stage 3
        s3_meta_GA.dqmask = best["dqmask"]
        s3_meta_GA.bg_thresh = best["bg_thresh"]
        s3_meta_GA.bg_hw = best["bg_hw"]
        s3_meta_GA.spec_hw = best["spec_hw"]
        s3_meta_GA.bg_deg = best["bg_deg"]
        s3_meta_GA.bg_method = best["bg_method"]
        s3_meta_GA.p3thresh = best["p3thresh"]
        s3_meta_GA.median_thresh = best["median_thresh"]
        s3_meta_GA.window_len = best["window_len"]

        # Stage 4
        s4_meta_GA.dqmask = best["dqmask"]
        s4_meta_GA.bg_thresh = best["bg_thresh"]
        s4_meta_GA.bg_hw = best["bg_hw"]
        s4_meta_GA.spec_hw = best["spec_hw"]
        s4_meta_GA.bg_deg = best["bg_deg"]
        s4_meta_GA.bg_method = best["bg_method"]
        s4_meta_GA.p3thresh = best["p3thresh"]
        s4_meta_GA.median_thresh = best["median_thresh"]
        s4_meta_GA.window_len = best["window_len"]

        # Perform parametric sweep
        best_param, best_fitness_value = optimizers.parametric_sweep_p7thresh_S3(
            objective_funcs.p7thresh,
            bounds_p7thresh,
            eventlabel,
            last_s2_meta_outputdir,
            s3_meta_GA,
            s4_meta_GA,
            scaling_MAD_white,
            scaling_MAD_spec,
        )

        best["p7thresh"] = best_param[0]

        print("Best parameters: ", best_param[0])
        print("Best fitness: ", best_fitness_value)

        history_fitness_score["p7thresh"] = best_fitness_value

    ## STEP 1.0 - Parametric Sweep of "expand" ##

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
        s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

        # Use 1 spectroscopic channel during optimization
        s4_meta_GA.nspecchan = 1

        # Setup Meta / Define Initial Population
        # Stage 3
        s3_meta_GA.dqmask = best["dqmask"]
        s3_meta_GA.bg_thresh = best["bg_thresh"]
        s3_meta_GA.bg_hw = best["bg_hw"]
        s3_meta_GA.spec_hw = best["spec_hw"]
        s3_meta_GA.bg_deg = best["bg_deg"]
        s3_meta_GA.bg_method = best["bg_method"]
        s3_meta_GA.p3thresh = best["p3thresh"]
        s3_meta_GA.median_thresh = best["median_thresh"]
        s3_meta_GA.window_len = best["window_len"]
        s3_meta_GA.p7thresh = best["p7thresh"]

        # Stage 4
        s4_meta_GA.dqmask = best["dqmask"]
        s4_meta_GA.bg_thresh = best["bg_thresh"]
        s4_meta_GA.bg_hw = best["bg_hw"]
        s4_meta_GA.spec_hw = best["spec_hw"]
        s4_meta_GA.bg_deg = best["bg_deg"]
        s4_meta_GA.bg_method = best["bg_method"]
        s4_meta_GA.p3thresh = best["p3thresh"]
        s4_meta_GA.median_thresh = best["median_thresh"]
        s4_meta_GA.window_len = best["window_len"]
        s4_meta_GA.p7thresh = best["p7thresh"]

        # Perform parametric sweep
        best_param, best_fitness_value = optimizers.parametric_sweep_S3(
            objective_funcs.expand,
            bounds_expand,
            eventlabel,
            last_s2_meta_outputdir,
            s3_meta_GA,
            s4_meta_GA,
            scaling_MAD_white,
            scaling_MAD_spec,
        )

        best["expand"] = best_param[0]

        print("Best parameters: ", best_param[0])
        print("Best fitness: ", best_fitness_value)

        history_fitness_score["expand"] = best_fitness_value

    ## STEP 1.0 - Run Stage 3 once and store Stage 3 directory to avoid re-running Stage 3 during Stage 4 optimization (Saves Time)

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
        s3_meta_GA.inputdir = last_s2_meta_outputdir

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

        # Setup Meta / Define Initial Population
        # Stage 3
        s3_meta_GA.dqmask = best["dqmask"]
        s3_meta_GA.expand = best["expand"]
        s3_meta_GA.bg_thresh = best["bg_thresh"]
        s3_meta_GA.bg_hw = best["bg_hw"]
        s3_meta_GA.spec_hw = best["spec_hw"]
        s3_meta_GA.bg_deg = best["bg_deg"]
        s3_meta_GA.bg_method = best["bg_method"]
        s3_meta_GA.p3thresh = best["p3thresh"]
        s3_meta_GA.median_thresh = best["median_thresh"]
        s3_meta_GA.window_len = best["window_len"]
        s3_meta_GA.p7thresh = best["p7thresh"]

        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
        # s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA, s3_meta=s3_meta)
        # shutil.rmtree(s3_meta.outputdir)
        # shutil.rmtree(s4_meta.outputdir)
        last_s3_meta_outputdir = s3_meta.outputdir

    ## STEP 1.0 - Parametric Sweep of "sigma" ##

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
        s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

        # Use 1 spectroscopic channel during optimization
        s4_meta_GA.nspecchan = 1

        # Setup Meta / Define Initial Population
        # Stage 3
        s3_meta_GA.dqmask = best["dqmask"]
        s3_meta_GA.expand = best["expand"]
        s3_meta_GA.bg_thresh = best["bg_thresh"]
        s3_meta_GA.bg_hw = best["bg_hw"]
        s3_meta_GA.spec_hw = best["spec_hw"]
        s3_meta_GA.bg_deg = best["bg_deg"]
        s3_meta_GA.bg_method = best["bg_method"]
        s3_meta_GA.p3thresh = best["p3thresh"]
        s3_meta_GA.median_thresh = best["median_thresh"]
        s3_meta_GA.window_len = best["window_len"]
        s3_meta_GA.p7thresh = best["p7thresh"]

        # Stage 4
        s4_meta_GA.dqmask = best["dqmask"]
        s4_meta_GA.expand = best["expand"]
        s4_meta_GA.bg_thresh = best["bg_thresh"]
        s4_meta_GA.bg_hw = best["bg_hw"]
        s4_meta_GA.spec_hw = best["spec_hw"]
        s4_meta_GA.bg_deg = best["bg_deg"]
        s4_meta_GA.bg_method = best["bg_method"]
        s4_meta_GA.p3thresh = best["p3thresh"]
        s4_meta_GA.median_thresh = best["median_thresh"]
        s4_meta_GA.window_len = best["window_len"]
        s4_meta_GA.p7thresh = best["p7thresh"]

        # Perform parametric sweep
        best_param, best_fitness_value = optimizers.parametric_sweep_S4(
            objective_funcs.sigma,
            bounds_sigma,
            eventlabel,
            last_s3_meta_outputdir,
            s3_meta,
            s4_meta_GA,
            scaling_MAD_white,
            scaling_MAD_spec,
        )

        best["sigma"] = best_param[0]

        print("Best parameters: ", best_param[0])
        print("Best fitness: ", best_fitness_value)

        history_fitness_score["sigma"] = best_fitness_value

    ## STEP 1.0 - Parametric Sweep of "box_width" ##

    # if optimizer == "parametric":
    if optimizer == "parametric" and __name__ == "__main__":

        ## Setup Meta ##
        s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
        s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

        # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
        s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

        # Use 1 spectroscopic channel during optimization
        s4_meta_GA.nspecchan = 1

        # Setup Meta / Define Initial Population
        # Stage 3
        s3_meta_GA.dqmask = best["dqmask"]
        s3_meta_GA.expand = best["expand"]
        s3_meta_GA.bg_thresh = best["bg_thresh"]
        s3_meta_GA.bg_hw = best["bg_hw"]
        s3_meta_GA.spec_hw = best["spec_hw"]
        s3_meta_GA.bg_deg = best["bg_deg"]
        s3_meta_GA.bg_method = best["bg_method"]
        s3_meta_GA.p3thresh = best["p3thresh"]
        s3_meta_GA.median_thresh = best["median_thresh"]
        s3_meta_GA.window_len = best["window_len"]
        s3_meta_GA.p7thresh = best["p7thresh"]

        # Stage 4
        s4_meta_GA.dqmask = best["dqmask"]
        s4_meta_GA.expand = best["expand"]
        s4_meta_GA.bg_thresh = best["bg_thresh"]
        s4_meta_GA.bg_hw = best["bg_hw"]
        s4_meta_GA.spec_hw = best["spec_hw"]
        s4_meta_GA.bg_deg = best["bg_deg"]
        s4_meta_GA.bg_method = best["bg_method"]
        s4_meta_GA.p3thresh = best["p3thresh"]
        s4_meta_GA.median_thresh = best["median_thresh"]
        s4_meta_GA.window_len = best["window_len"]
        s4_meta_GA.p7thresh = best["p7thresh"]
        s4_meta_GA.sigma = best["sigma"]

        # Perform parametric sweep
        best_param, best_fitness_value = optimizers.parametric_sweep_S4(
            objective_funcs.box_width,
            bounds_box_width,
            eventlabel,
            last_s3_meta_outputdir,
            s3_meta,
            s4_meta_GA,
            scaling_MAD_white,
            scaling_MAD_spec,
        )

        best["box_width"] = best_param[0]

        print("Best parameters: ", best_param[0])
        print("Best fitness: ", best_fitness_value)

        history_fitness_score["box_width"] = best_fitness_value

    ## STEP 1.2 - Save Best ECF Values ##

    # Setup Meta
    s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile)
    s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile)

    s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
    s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

    # Stage 3
    s3_meta_GA.dqmask = best["dqmask"]
    s3_meta_GA.expand = best["expand"]
    s3_meta_GA.bg_thresh = best["bg_thresh"]
    s3_meta_GA.bg_hw = best["bg_hw"]
    s3_meta_GA.spec_hw = best["spec_hw"]
    s3_meta_GA.bg_deg = best["bg_deg"]
    s3_meta_GA.bg_method = best["bg_method"]
    s3_meta_GA.p3thresh = best["p3thresh"]
    s3_meta_GA.median_thresh = best["median_thresh"]
    s3_meta_GA.window_len = best["window_len"]
    s3_meta_GA.p7thresh = best["p7thresh"]

    # Stage 4
    s4_meta_GA.dqmask = best["dqmask"]
    s4_meta_GA.expand = best["expand"]
    s4_meta_GA.bg_thresh = best["bg_thresh"]
    s4_meta_GA.bg_hw = best["bg_hw"]
    s4_meta_GA.spec_hw = best["spec_hw"]
    s4_meta_GA.bg_deg = best["bg_deg"]
    s4_meta_GA.bg_method = best["bg_method"]
    s4_meta_GA.p3thresh = best["p3thresh"]
    s4_meta_GA.median_thresh = best["median_thresh"]
    s4_meta_GA.window_len = best["window_len"]
    s4_meta_GA.p7thresh = best["p7thresh"]
    s4_meta_GA.sigma = best["sigma"]
    s4_meta_GA.box_width = best["box_width"]

    print(f"Optimized ECF Inputs for Stages 3 & 4 : {best}")

    ## STEP 1.2 - Test the Optimized White Light Curve  - Stages 3 & 4 ##

    ## Setup Meta ##
    s3_meta_GA = S3MetaClass(ecf_path, s3_ecffile, eventlabel)
    s4_meta_GA = S4MetaClass(ecf_path, s4_ecffile, eventlabel)

    s3_meta_GA.turbo_optimizer = False

    s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
    # s3_meta_GA.ywindow = [best['ywindow_LB'], best['ywindow_UB']]
    if "ywindow_LB" in best:
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]
    else:
        s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

    # Use 1 spectroscopic channel during optimization
    s4_meta_GA.nspecchan = 1

    # Overwrite default meta inputs with optimized Stage 3 and Stage 4 inputs
    # Stage 3
    s3_meta_GA.dqmask = best["dqmask"]
    s3_meta_GA.expand = best["expand"]
    s3_meta_GA.bg_thresh = best["bg_thresh"]
    s3_meta_GA.bg_hw = best["bg_hw"]
    s3_meta_GA.spec_hw = best["spec_hw"]
    s3_meta_GA.bg_deg = best["bg_deg"]
    s3_meta_GA.bg_method = best["bg_method"]
    s3_meta_GA.p3thresh = best["p3thresh"]
    s3_meta_GA.median_thresh = best["median_thresh"]
    s3_meta_GA.window_len = best["window_len"]
    s3_meta_GA.p7thresh = best["p7thresh"]
    # Stage 4
    s4_meta_GA.dqmask = best["dqmask"]
    s4_meta_GA.expand = best["expand"]
    s4_meta_GA.bg_thresh = best["bg_thresh"]
    s4_meta_GA.bg_hw = best["bg_hw"]
    s4_meta_GA.spec_hw = best["spec_hw"]
    s4_meta_GA.bg_deg = best["bg_deg"]
    s4_meta_GA.bg_method = best["bg_method"]
    s4_meta_GA.p3thresh = best["p3thresh"]
    s4_meta_GA.median_thresh = best["median_thresh"]
    s4_meta_GA.window_len = best["window_len"]
    s4_meta_GA.p7thresh = best["p7thresh"]
    s4_meta_GA.sigma = best["sigma"]
    s4_meta_GA.box_width = best["box_width"]

    directory = s4_meta_GA.topdir
    # directory = last_outputdir_S4
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Run Test
    if __name__ == "__main__":
        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
        s4_spec, s4_lc, s4_meta = s4.genlc(
            eventlabel, input_meta=s4_meta_GA, s3_meta=s3_meta
        )

    ## STEP 1.2 - Create Run String for Output Path

    def create_run_string(event_label, directory_path):
        # Get today's date in YYYY-MM-DD format
        today_date = datetime.now().strftime("%Y-%m-%d")

        # Initialize the run number to 1
        run_number = 1

        # Check if the directory exists
        if os.path.exists(directory_path):
            # List all files in the directory and find the highest run number
            for file in os.listdir(directory_path):
                if file.startswith(f"Optimized_{today_date}_{event_label}_run"):
                    try:
                        current_run_number = int(file.split("_")[-1][3:])
                        run_number = max(run_number, current_run_number + 1)
                    except ValueError:
                        continue

        # Create the formatted string
        return f"Optimized_{today_date}_{eventlabel}_run{run_number}"

    # Create Run String
    run_string = create_run_string(eventlabel, outputdir_optimization)

    # Update outputdir to run-specific folder
    outputdir_optimization = outputdir_optimization + run_string + "/"

    ## STEP 1.2 - Save best S3-S4 ECF values ##

    # Use pickle.dump to save the dictionary
    if not os.path.exists(outputdir_optimization):
        os.makedirs(outputdir_optimization)

    with open(outputdir_optimization + "best_inputs.pkl", "wb") as f:
        pickle.dump(best, f)

    ## STEP 1.2 - Set save directory for optimized white light curve ##

    outputdir_WLC_S3 = s3_meta.outputdir
    outputdir_WLC_S4 = s4_meta.outputdir

    # Save optimized WLC data for reference
    s3_meta_best_WLC = s3_meta
    s4_meta_best_WLC = s4_meta

    ## STEP 1.2 - Copy the optimizer_inputs.txt file used to the optimization output folder ##

    def copy_and_rename_file(src, dest_dir, dest_filename):
        # Check if destination directory exists; if not, create it
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Copy and rename the file
        shutil.copy(src, os.path.join(dest_dir, dest_filename))

    if __name__ == "__main__":
        source_file = "optimizer_inputs.txt"
        destination_directory = outputdir_optimization
        destination_filename = "optimizer_inputs_copy.txt"

        copy_and_rename_file(source_file, destination_directory, destination_filename)
        print(
            f"'{source_file}' has been copied to '{destination_directory}/{destination_filename}'"
        )

    ## STEP 1.2 - Create output file for optimization results ##

    output_file = outputdir_optimization + "optimization_results.txt"

    with open(output_file, "w") as f:

        f.write("## OPTIMIZATION RESULTS ##\n\n")

        # Write Dictionary Contents
        f.write("## Optimized ECF & EPF Inputs ##\n")
        for key, value in best.items():
            f.write(f"{key}: {value}\n")

        f.write("\n")

        # Write Optimization Results
        f.write("## Optimization Metrics ##\n")
        f.write(
            f"MAD_WLC = {sum(s4_meta_best_WLC.mad_s4_binned) / len(s4_meta_best_WLC.mad_s4_binned)}\n"
        )
        f.write(f"MAD_Spec = {s4_meta_best_WLC.mad_s4}\n")

        # f.write(chi2red_nspecchan)

    print(f"Optimization results saved to {output_file}")

    ## STEP 1.2 - Save results for optimized white light curve ##

    def copy_files_and_subfolders_to_new_directory(src_dirs, dest_dir):
        # Ensure the destination directory exists
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        # Iterate over the source directories and copy each file and folder
        for src_dir, folder_name in src_dirs:
            if os.path.exists(src_dir):
                dest_subdir = os.path.join(dest_dir, folder_name)
                if not os.path.exists(dest_subdir):
                    os.makedirs(dest_subdir)
                shutil.copytree(src_dir, dest_subdir, dirs_exist_ok=True)

    # Assuming these are full path directories
    directories = [
        (outputdir_WLC_S3, "S3"),
        (outputdir_WLC_S4, "S4"),
    ]

    best_directory_white = outputdir_optimization + "White"

    # Copy files and subfolders
    copy_files_and_subfolders_to_new_directory(directories, best_directory_white)

    # Plot Optimization History
    keys = list(history_fitness_score.keys())
    values = list(history_fitness_score.values())

    # Plotting the values
    plt.figure(figsize=(12, 6))
    plt.plot(keys, values, marker="o", linestyle="-", color="b")
    plt.xticks(rotation=45, ha="right")
    if scaling_MAD_white > 0.99 and scaling_MAD_spec < 0.01:
        plt.ylabel("MAD Value - White")
    elif scaling_MAD_white < 0.01 and scaling_MAD_spec > 0.99:
        plt.ylabel("MAD Value - Spectroscopic")
    else:
        plt.ylabel(
            f"Fitness Score ({scaling_MAD_white}*MAD$_{{WLC}}$ + {scaling_MAD_spec}*MAD$_{{spec}}$)"
        )
    plt.xlabel("ECF Parameter")
    plt.title("Optimization History")
    plt.grid(True)  # Add grid
    plt.tight_layout()
    # plt.show()
    filepath_history_plot = outputdir_optimization + "OptimizationHistory.png"

    plt.savefig(filepath_history_plot)
