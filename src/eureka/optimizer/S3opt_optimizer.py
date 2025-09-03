
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
# import astropy.io.fits as pf
# import importlib, re
# from datetime import datetime
# import pickle, shutil, fileinput
# import warnings
# warnings.filterwarnings("ignore")

# import eureka.S1_detector_processing.s1_process as s1
# import eureka.S2_calibrations.s2_calibrate as s2
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4
# from ..S1_detector_processing.s1_meta import S1MetaClass
# from ..S2_calibrations.s2_meta import S2MetaClass
from eureka.S3_data_reduction.s3_meta import S3MetaClass
from eureka.S4_generate_lightcurves.s4_meta import S4MetaClass
from .S3opt_meta import S3optMetaClass
from . import optimizers
from ..lib import logedit
# from ..lib import plots


def initialize_meta(meta, eventlabel, ecf_path=None):
    """Initialize MetaClass objects for the optimization run.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        An S3opt metadata object
    eventlabel : str
        The unique identifier for these data.
    ecf_path : str; optional
        The absolute or relative path to where ecfs are stored. Defaults to
        None which resolves to './'.

    Returns
    -------
    s3_meta : eureka.lib.readECF.MetaClass
        The Stage 3 metadata object.
    s4_meta : eureka.lib.readECF.MetaClass
        The Stage 4 metadata object.
    """
    # Setup Meta objects
    s3_meta = S3MetaClass(folder=ecf_path, eventlabel=eventlabel)
    s4_meta = S4MetaClass(folder=ecf_path, eventlabel=eventlabel)

    # Overwrite certain Meta values
    s3_meta.inputdir = meta.s2_inputdir
    s3_meta.outputdir_raw = os.path.join(meta.outputdir_raw, 'Stage3')
    s4_meta.outputdir_raw = os.path.join(meta.outputdir_raw, 'Stage4')
    s3_meta.isplots_S3 = meta.isplots_S3opt
    s4_meta.isplots_S4 = meta.isplots_S3opt
    s3_meta.verbose = meta.verbose
    s4_meta.verbose = meta.verbose
    s4_meta.nspecchan = 1
    s4_meta.compute_ld = False

    return s3_meta, s4_meta


def optimize(eventlabel, ecf_path=None, initial_run=False):
    """
    Eureka! Optimization for JWST NIRSpec PRISM Data
    ---------------------------------------------

    Description:
    This script is designed to optimize the stage 1-4 ECF parameters for JWST NIRSpec PRISM observations.

    Inputs:
    - Loaded from an input text file, 'optimizer_inputs_nirspec_PRISM.txt'.
    - Parameters not specified for optimization in the optimizer input text file will assume ECF values by default.

    Outputs:
    - Optimized parameter values saved in the "best" dictionary.
    - Plot of optimization fitness values.
    - Optimization results, printing the best ECF values and fitness scores (WLC MAD, 2D MAD, etc.).
    """
    # Load optimizer parameters from ECF
    s3opt_meta = S3optMetaClass(folder=ecf_path, eventlabel=eventlabel)
    s3opt_meta.s2_inputdir = os.path.join(s3opt_meta.topdir, s3opt_meta.inputdir)
    s3opt_meta.s3_logname = s3opt_meta.outputdir + f'S3opt_{eventlabel}.log'
    if not os.path.exists(s3opt_meta.outputdir):
        os.mkdir(s3opt_meta.outputdir)
    log = logedit.Logedit(s3opt_meta.s3_logname)

    # Create dictionaries to keep track of optimization metrics
    best = {}
    history_MAD_white = {}
    history_MAD_spec = {}
    history_MAD_chi2red = {}
    history_fitness_score = {}

    if initial_run:

        # Setup Meta objects
        meta = deepcopy(s3opt_meta)
        s3_meta, s4_meta = initialize_meta(meta, eventlabel, ecf_path=None)

        # s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
        # s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
        s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
                                           s3_meta=s3_meta)

        history_fitness_score["initial_run"] = (
            meta.scaling_MAD_spec * s4_meta.mad_s4 +
            meta.scaling_MAD_white * s4_meta.mad_s4_binned[0])
        log.writelog("Initial fitness value: " +
                     f"{history_fitness_score["initial_run"]}\n")

    for p in s3opt_meta.params_to_optimize:

        # Setup Meta objects
        meta = deepcopy(s3opt_meta)
        meta.opt_param_name = p
        s3_meta, s4_meta = initialize_meta(meta, eventlabel, ecf_path=None)

        # Extract bounds for parameter(s) to optimize
        if "bounds_" + p in meta.__dict__.keys():
            bounds = meta.__dict__["bounds_" + p]
        else:
            log.writelog(f"Parameter {p} not recognized. Skipping...")
            continue
        # Update Meta parameters with best values from previous iterations
        for key, value in best.items():
            if key in s3_meta.__dict__.keys():
                s3_meta.__dict__[key] = value
            if key in s4_meta.__dict__.keys():
                s4_meta.__dict__[key] = value

        # Perform parametric sweep
        if p == "spec_hw__bg_hw":
            # Optimize both spec_hw and bg_hw simultaneously
            # Require that spec_hw < bg_hw
            best_param_value, best_fitness_value = optimizers.sweep_list_lt(
                eventlabel,
                bounds,
                meta,
                s3_meta,
                s4_meta,
                log,
            )
        elif "__" in p:
            # Optimize two independent parameters simultaneously
            best_param_value, best_fitness_value = optimizers.sweep_list_double(
                eventlabel,
                bounds,
                meta,
                s3_meta,
                s4_meta,
                log,
            )
        else:
            # Optimize single parameter
            best_param_value, best_fitness_value = optimizers.sweep_list_single(
                eventlabel,
                bounds,
                meta,
                s3_meta,
                s4_meta,
                log,
            )

        # Save Results in "best" Dictionary
        param_names = p.split("__")
        if (type(best_param_value) is not list) and \
            (type(best_param_value) is not np.ndarray):
                best_param_value = [best_param_value]
        for i, param in enumerate(param_names):
            best[param] = best_param_value[i]

        # Print Results of Parametric Sweep
        log.writelog(f"Optimized parameter: {p}")
        log.writelog(f"Best parameter value(s): {best_param_value}")
        log.writelog(f"Best fitness value: {best_fitness_value}\n")

        history_fitness_score[p] = best_fitness_value

    log.closelog()