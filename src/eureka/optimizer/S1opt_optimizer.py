
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import pickle, shutil, fileinput
# import warnings
# warnings.filterwarnings("ignore")

import eureka.S1_detector_processing.s1_process as s1
import eureka.S2_calibrations.s2_calibrate as s2
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4
from eureka.S1_detector_processing.s1_meta import S1MetaClass
from eureka.S2_calibrations.s2_meta import S2MetaClass
from eureka.S3_data_reduction.s3_meta import S3MetaClass
from eureka.S3_data_reduction import plots_s3
from eureka.S4_generate_lightcurves.s4_meta import S4MetaClass
from .S1opt_meta import S1optMetaClass
from . import optimizers
from ..lib import logedit


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
    s1_meta : eureka.lib.readECF.MetaClass
        The Stage 1 metadata object.
    s2_meta : eureka.lib.readECF.MetaClass
        The Stage 2 metadata object.
    s3_meta : eureka.lib.readECF.MetaClass
        The Stage 3 metadata object.
    s4_meta : eureka.lib.readECF.MetaClass
        The Stage 4 metadata object.
    """
    # Setup Meta objects
    s1_meta = S1MetaClass(folder=ecf_path, eventlabel=eventlabel)
    s2_meta = S2MetaClass(folder=ecf_path, eventlabel=eventlabel)
    s3_meta = S3MetaClass(folder=ecf_path, eventlabel=eventlabel)
    s4_meta = S4MetaClass(folder=ecf_path, eventlabel=eventlabel)

    # Overwrite certain Meta values
    s1_meta.inputdir = meta.s0_inputdir
    s1_meta.outputdir_raw = os.path.join(meta.outputdir_raw, 'Stage1')
    s2_meta.outputdir_raw = os.path.join(meta.outputdir_raw, 'Stage2')
    s3_meta.outputdir_raw = os.path.join(meta.outputdir_raw, 'Stage3')
    s4_meta.outputdir_raw = os.path.join(meta.outputdir_raw, 'Stage4')
    s1_meta.isplots_S1 = meta.isplots_S1opt
    s2_meta.isplots_S2 = meta.isplots_S1opt
    s3_meta.isplots_S3 = meta.isplots_S1opt
    s4_meta.isplots_S4 = meta.isplots_S1opt
    s1_meta.verbose = meta.verbose
    s2_meta.verbose = meta.verbose
    s3_meta.verbose = meta.verbose
    s4_meta.verbose = meta.verbose
    s3_meta.record_ypos = False
    s4_meta.nspecchan = 1
    s4_meta.compute_ld = False

    return s1_meta, s2_meta, s3_meta, s4_meta


def optimize(eventlabel, ecf_path=None, initial_run=False):
    """
    Eureka! optimization wrapper for Stages 3 and 4.

    Parameters
    ----------
    eventlabel : str
        The unique identifier for these data.
    ecf_path : str; optional
        The absolute or relative path to where ecfs are stored. Defaults to
        None which resolves to './'.
    initial_run : boolean; optional
        Set to True to perform an initial run with default ECF parameters
        before starting the optimization. Defaults to False.

    Returns
    -------
    history_fitness_score : dict
        The fitness score after optimizing each parameter.
    """
    # Load optimizer parameters from ECF
    s1opt_meta = S1optMetaClass(folder=ecf_path, eventlabel=eventlabel)

    # Setup directories and log file
    s1opt_meta.s0_inputdir = os.path.join(s1opt_meta.topdir, s1opt_meta.inputdir)
    s1opt_meta.s1_logname = s1opt_meta.outputdir + f'S3opt_{eventlabel}.log'
    if not os.path.exists(s1opt_meta.outputdir):
        os.mkdir(s1opt_meta.outputdir)
    if not os.path.exists(os.path.join(s1opt_meta.outputdir, "figs")):
        os.makedirs(os.path.join(s1opt_meta.outputdir, "figs"))
    log = logedit.Logedit(s1opt_meta.s1_logname)

    # Copy ECF to output directory
    log.writelog('Copying S1opt control file', mute=(not s1opt_meta.verbose))
    s1opt_meta.copy_ecf()

    # Create dictionaries to keep track of optimization metrics
    best = {}
    history_fitness_score = {}

    if initial_run:
        # Setup Meta objects
        meta = deepcopy(s1opt_meta)
        s1_meta, s2_meta, s3_meta, s4_meta = initialize_meta(meta, eventlabel,
                                                             ecf_path=None)

        s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta)
        s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta)
        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
        s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
                                           s3_meta=s3_meta)

        # Record initial fitness score
        history_fitness_score["initial_run"] = (
            meta.scaling_MAD_spec * s4_meta.mad_s4 +
            meta.scaling_MAD_white * s4_meta.mad_s4_binned[0])
        log.writelog("Initial fitness value: " +
                     f"{history_fitness_score["initial_run"]}\n")

    for p in s1opt_meta.params_to_optimize:
        # Setup Meta objects
        meta = deepcopy(s1opt_meta)
        meta.opt_param_name = p
        s1_meta, s2_meta, s3_meta, s4_meta = initialize_meta(meta, eventlabel,
                                                             ecf_path=None)

        # Extract bounds for parameter(s) to optimize
        if "bounds_" + p in meta.__dict__.keys():
            bounds = meta.__dict__["bounds_" + p]
        else:
            log.writelog(f"Parameter {p} not recognized. Skipping...")
            continue

        # Update Meta parameters with best values from previous iterations
        for key, value in best.items():
            if key in s1_meta.__dict__.keys():
                s1_meta.params[key] = value

        # Perform parametric sweep
        if p == "spec_hw__bg_hw":
            # Optimize both spec_hw and bg_hw simultaneously
            # Require that spec_hw < bg_hw
            best_param_value, best_fitness_value = optimizers.sweep_list_lt(
                bounds, meta, log,
                s1_meta=s1_meta, s2_meta=s2_meta,
                s3_meta=s3_meta, s4_meta=s4_meta)
        elif "__" in p:
            # Optimize two independent parameters simultaneously
            best_param_value, best_fitness_value = optimizers.sweep_list_double(
                bounds, meta, log,
                s1_meta=s1_meta, s2_meta=s2_meta,
                s3_meta=s3_meta, s4_meta=s4_meta)
        else:
            # Optimize single parameter
            best_param_value, best_fitness_value = optimizers.sweep_list_single(
                bounds, meta, log,
                s1_meta=s1_meta, s2_meta=s2_meta,
                s3_meta=s3_meta, s4_meta=s4_meta)

        # Check that optimization was successful
        if best_param_value is not None:
            # Save results in "best" dictionary
            param_names = p.split("__")
            if (type(best_param_value) is not list) and \
                (type(best_param_value) is not np.ndarray):
                    best_param_value = [best_param_value]
            for i, param in enumerate(param_names):
                best[param] = best_param_value[i]

            # Print results of parametric sweep
            log.writelog(f"Optimized parameter: {p}")
            log.writelog(f"Best parameter value(s): {best_param_value}")
            log.writelog(f"Best fitness value: {best_fitness_value}\n")

            history_fitness_score[p] = best_fitness_value

    if meta.isplots_S1opt >= 1:
        plots_s3.fitness_scores(s1opt_meta, history_fitness_score)

    # Save the best dictionary to a pickle file
    with open(os.path.join(s1opt_meta.outputdir, "best_params.pkl"), "wb") as f:
        pickle.dump(best, f)

    # Define and create optimized ECF file path
    opt_path = os.path.join(s1opt_meta.outputdir, "opt_ECFs")
    if not os.path.exists(opt_path):
        os.mkdir(opt_path)

    # Update S1 ECF file with optimized parameters
    for key, value in best.items():
        if key in s1_meta.__dict__.keys():
            s1_meta.params[key] = value

    # Write optimized ECF file
    s1_meta.write(opt_path)

    log.closelog()

    return s1opt_meta, history_fitness_score