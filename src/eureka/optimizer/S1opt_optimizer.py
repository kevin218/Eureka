
import os
import numpy as np
from copy import deepcopy
import time as time_pkg
import pickle
import shutil

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
from ..lib import logedit, util


def wrapper(eventlabel, ecf_path=None, initial_run=True, final_run=True):
    """
    Eureka! optimization wrapper for Stage 1.

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
    final_run : boolean; optional
        Set to True to perform a final run with optimized ECF parameters.
        Defaults to True.

    Returns
    -------
    s1opt_meta : eureka.lib.readECF.MetaClass
        An S1opt metadata object
    history : dict
        The fitness score after optimizing each parameter.
    best : dict
        The best parameter values found during the optimization.
    """
    # Load optimizer parameters from ECF
    s1opt_meta = S1optMetaClass(folder=ecf_path, eventlabel=eventlabel)

    # Setup directories and log file
    s1opt_meta.datetime = time_pkg.strftime('%Y-%m-%d')
    run = util.makedirectory(s1opt_meta, 'S1opt')
    s1opt_meta.outputdir = util.pathdirectory(s1opt_meta, 'S1opt', run)

    s1opt_meta.s0_inputdir = os.path.join(s1opt_meta.topdir,
                                          s1opt_meta.inputdir)
    s1opt_meta.s1_logname = s1opt_meta.outputdir + f'S1opt_{eventlabel}.log'
    log = logedit.Logedit(s1opt_meta.s1_logname)
    # Update raw dir by removing topdir from ouputdir
    s1opt_meta.outputdir_raw = s1opt_meta.outputdir[len(s1opt_meta.topdir):]

    # Copy ECF to output directory
    log.writelog('Copying S1opt control file', mute=(not s1opt_meta.verbose))
    s1opt_meta.copy_ecf()

    # Create dictionaries to keep track of optimization metrics
    best = {}
    history = {}

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
        history["initial_run"] = (
            meta.scaling_MAD_spec * s4_meta.mad_s4 +
            meta.scaling_MAD_white * s4_meta.mad_s4_binned[0])
        log.writelog(f"Initial fitness value: {history["initial_run"]}")
        log.writelog(f"Initial white MAD: {s4_meta.mad_s4_binned[0]}")
        log.writelog(f"Initial spec MAD: {s4_meta.mad_s4}\n")

    for p in s1opt_meta.params_to_optimize:
        s1opt_meta, log, history, best = optimize(s1opt_meta, log, history,
                                                  best, p, eventlabel,
                                                  ecf_path, 1)

    # Save the best dictionary to a pickle file
    with open(os.path.join(s1opt_meta.outputdir, "best_params.pkl"), "wb") as f:
        pickle.dump(best, f)

    # Define and create optimized ECF file path
    opt_path = os.path.join(s1opt_meta.outputdir, "opt_ECFs")
    if not os.path.exists(opt_path):
        os.mkdir(opt_path)

    # Update S1 ECF file with optimized parameters
    s1_meta, s2_meta, s3_meta, s4_meta = initialize_meta(s1opt_meta, eventlabel,
                                                         ecf_path=None)
    for key, value in best.items():
        s1_meta.params[key] = value
        setattr(s1_meta, key, value)

    # Write optimized ECF files
    s1_meta.write(opt_path)

    if final_run:
        s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta)
        s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta)
        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
        s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
                                           s3_meta=s3_meta)
        # Record initial fitness score
        history["final_run"] = (
            s1opt_meta.scaling_MAD_spec * s4_meta.mad_s4 +
            s1opt_meta.scaling_MAD_white * s4_meta.mad_s4_binned[0])
        log.writelog(f"Final fitness value: {history["final_run"]}")
        log.writelog(f"Final white MAD: {s4_meta.mad_s4_binned[0]}")
        log.writelog(f"Final spec MAD: {s4_meta.mad_s4}\n")

    if s1opt_meta.isplots_S1opt >= 1:
        plots_s3.fitness_scores(s1opt_meta, history)

    log.closelog()

    # Delete intermediate files if requested
    if s1opt_meta.delete_final:
        if os.path.exists(s1_meta.outputdir_raw):
            shutil.rmtree(s1_meta.outputdir_raw)

    return s1opt_meta, history, best


def optimize(s1opt_meta, log, history, best, p, eventlabel, ecf_path, stage):
    """Optimize a single parameter using the specified optimizer.

    Parameters
    ----------
    s1opt_meta : eureka.lib.readECF.MetaClass
        An S1opt metadata object
    log : eureka.lib.logedit.Logedit
        The log object for writing to the log file.
    history : dict
        The fitness score after optimizing each parameter.
    best : dict
        The best parameter values found so far.
    p : str
        The parameter to optimize.
    eventlabel : str
        The unique identifier for these data.
    ecf_path : str; optional
        The absolute or relative path to where ecfs are stored.
    stage : int
        The stage number (1 for S1 optimization).

    Returns
    -------
    s1opt_meta : eureka.lib.readECF.MetaClass
        An S1opt metadata object
    log : eureka.lib.logedit.Logedit
        The log object for writing to the log file.
    history : dict
        The fitness score after optimizing each parameter.
    best : dict
        The best parameter values found so far.
    """
    # Setup Meta objects
    meta = deepcopy(s1opt_meta)
    meta.opt_param_name = p
    s1_meta, s2_meta, s3_meta, s4_meta = initialize_meta(meta, eventlabel,
                                                         ecf_path=None)

    # Extract bounds for parameter(s) to optimize
    if hasattr(meta, "bounds_" + p):
        bounds = getattr(meta, "bounds_" + p)
        log.writelog(f"Optimizing parameter {p} over bounds: {bounds}")
        log.writelog("Initial parameter value: " +
                     f"{getattr(s1_meta, p, None)}")
    elif "__" in p:
        # Extract default bounds for two parameters
        param_names = p.split("__")
        bounds = []
        init_vals = []
        for param in param_names:
            if hasattr(meta, "bounds_" + param):
                bounds.append(getattr(meta, "bounds_" + param))
                init_vals.append(getattr(s1_meta, param, None))
            else:
                log.writelog(f"Could not create bounds for parameter {p}. " +
                             "Please manually specify bounds in ECF. " +
                             "Skipping...")
                return s1opt_meta, log, history, best
        log.writelog(f"Optimizing parameters {p} over bounds: {bounds}")
        log.writelog(f"Initial parameter values: {init_vals}")
    else:
        log.writelog(f"No default bounds exist for parameter {p}. " +
                     "Please manually specify bounds in ECF. Skipping...")
        return s1opt_meta, log, history, best

    # Update Meta parameters with best values from previous iterations
    for key, value in best.items():
        s1_meta.params[key] = value
        setattr(s1_meta, key, value)

    # Perform parametric sweep
    if p == "spec_hw__bg_hw":
        # Optimize both spec_hw and bg_hw simultaneously
        # Require that spec_hw < bg_hw
        best_param_value, best_fitness_value = optimizers.sweep_list_lt(
            bounds, meta, log, stage, s1_meta=s1_meta, s2_meta=s2_meta,
            s3_meta=s3_meta, s4_meta=s4_meta)
    elif "__" in p:
        # Optimize two independent parameters simultaneously
        best_param_value, best_fitness_value = optimizers.sweep_list_double(
            bounds, meta, log, stage, s1_meta=s1_meta, s2_meta=s2_meta,
            s3_meta=s3_meta, s4_meta=s4_meta)
    else:
        # Optimize single parameter
        best_param_value, best_fitness_value = optimizers.sweep_list_single(
            bounds, meta, log, stage, s1_meta=s1_meta, s2_meta=s2_meta,
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
        log.writelog(f"Best parameter value(s): {best_param_value}")
        log.writelog(f"Best fitness value: {best_fitness_value}\n")

        history[p] = best_fitness_value

    return s1opt_meta, log, history, best


def initialize_meta(meta, eventlabel, ecf_path=None):
    """Initialize MetaClass objects for the optimization run.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        An S1opt metadata object
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
    # Setup Stage 1 Meta object and overwrite certain Meta values
    s1_meta = S1MetaClass(folder=ecf_path, eventlabel=eventlabel)
    s1_meta.inputdir = meta.s0_inputdir
    s1_meta.inputdir_raw = s1_meta.inputdir[len(meta.topdir):]
    s1_meta.outputdir_raw = os.path.join(meta.outputdir_raw, 'Stage1')
    s1_meta.isplots_S1 = meta.isplots_S1opt
    s1_meta.verbose = meta.verbose

    # Setup Stage 2 Meta object and overwrite certain Meta values
    s2_meta = S2MetaClass(folder=ecf_path, eventlabel=eventlabel)
    # s2_meta.suffix = 'rateints'
    s2_meta.inputdir = os.path.join(meta.outputdir, 'Stage1')
    s2_meta.inputdir_raw = s2_meta.inputdir[len(meta.topdir):]
    s2_meta.outputdir_raw = os.path.join(meta.outputdir_raw, 'Stage2')
    s2_meta.isplots_S2 = meta.isplots_S1opt
    s2_meta.verbose = meta.verbose

    # Setup Stage 3 Meta object and overwrite certain Meta values
    s3_meta = S3MetaClass(folder=ecf_path, eventlabel=eventlabel)
    # s3_meta.suffix = 'calints'
    s3_meta.inputdir = os.path.join(meta.outputdir, 'Stage2')
    s3_meta.inputdir_raw = s3_meta.inputdir[len(meta.topdir):]
    s3_meta.outputdir_raw = os.path.join(meta.outputdir_raw, 'Stage3')
    s3_meta.isopt_S1 = meta.isopt_S1
    s3_meta.isopt_S3 = meta.isopt_S3
    s3_meta.isplots_S3 = meta.isplots_S1opt
    s3_meta.verbose = meta.verbose
    s3_meta.record_ypos = False

    # Setup Stage 4 Meta object and overwrite certain Meta values
    s4_meta = S4MetaClass(**s3_meta.__dict__)
    s4_meta.inputdir = os.path.join(meta.outputdir, 'Stage3')
    s4_meta.inputdir_raw = s4_meta.inputdir[len(meta.topdir):]
    s4_meta.outputdir_raw = os.path.join(meta.outputdir_raw, 'Stage4')
    s4_meta.isplots_S4 = meta.isplots_S1opt
    s4_meta.verbose = meta.verbose
    s4_meta.nspecchan = 1
    s4_meta.compute_ld = False

    return s1_meta, s2_meta, s3_meta, s4_meta

