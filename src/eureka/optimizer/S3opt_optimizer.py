
import os
import numpy as np
from copy import deepcopy
import time as time_pkg
import pickle
import shutil

import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4
from eureka.S3_data_reduction.s3_meta import S3MetaClass
from eureka.S3_data_reduction import plots_s3
from eureka.S4_generate_lightcurves.s4_meta import S4MetaClass
from .S3opt_meta import S3optMetaClass
from . import optimizers
from ..lib import logedit, util


def wrapper(eventlabel, ecf_path=None, initial_run=True, final_run=True):
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
        before starting the optimization. Defaults to True.
    final_run : boolean; optional
        Set to True to perform a final run with optimized ECF parameters.
        Defaults to True.

    Returns
    -------
    s3opt_meta : eureka.lib.readECF.MetaClass
        An S3opt metadata object
    history : dict
        The fitness score after optimizing each parameter.
    best : dict
        The best parameter values found during the optimization.
    """
    # Load optimizer parameters from ECF
    s3opt_meta = S3optMetaClass(folder=ecf_path, eventlabel=eventlabel)

    # Setup directories and log file
    s3opt_meta.datetime = time_pkg.strftime('%Y-%m-%d')
    run = util.makedirectory(s3opt_meta, 'S3opt')
    s3opt_meta.outputdir = util.pathdirectory(s3opt_meta, 'S3opt', run)
    s3opt_meta.s2_inputdir = os.path.join(s3opt_meta.topdir,
                                          s3opt_meta.inputdir)
    s3opt_meta.s3_logname = s3opt_meta.outputdir + f'S3opt_{eventlabel}.log'
    log = logedit.Logedit(s3opt_meta.s3_logname)
    # Update raw dir by removing topdir from ouputdir
    s3opt_meta.outputdir_raw = s3opt_meta.outputdir[len(s3opt_meta.topdir):]

    # Copy ECF to output directory
    log.writelog('Copying S3opt control file', mute=(not s3opt_meta.verbose))
    s3opt_meta.copy_ecf()

    # Create dictionaries to keep track of optimization metrics
    best = {}
    history = {}

    if initial_run:
        # Setup Meta objects
        meta = deepcopy(s3opt_meta)
        s3_meta, s4_meta = initialize_meta(meta, eventlabel, ecf_path=ecf_path)

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

    # Run optimization loop for Stage 3 parameters
    for p in s3opt_meta.params_to_optimize_s3:
        s3opt_meta, log, history, best = optimize(s3opt_meta, log, history,
                                                  best, p, eventlabel,
                                                  ecf_path, 3)

    # Run optimization loop for Stage 4 parameters
    for i, p in enumerate(s3opt_meta.params_to_optimize_s4):
        s3opt_meta, log, history, best = optimize(s3opt_meta, log, history,
                                                  best, p, eventlabel,
                                                  ecf_path, 4)

    # Save the best dictionary to a pickle file
    with open(os.path.join(s3opt_meta.outputdir, "best_params.pkl"), "wb") as f:
        pickle.dump(best, f)

    # Define and create optimized ECF file path
    opt_path = os.path.join(s3opt_meta.outputdir, "opt_ECFs")
    if not os.path.exists(opt_path):
        os.mkdir(opt_path)

    # Update S3 and S4 ECF files with optimized parameters
    s3_meta, s4_meta = initialize_meta(s3opt_meta, eventlabel,
                                       ecf_path=ecf_path)
    for key, value in best.items():
        if key in s3_meta.__dict__.keys():
            s3_meta.params[key] = value
            s3_meta.__dict__[key] = value
        if key in s4_meta.__dict__.keys():
            s4_meta.params[key] = value
            s4_meta.__dict__[key] = value

    # Write optimized ECF files
    s3_meta.write(opt_path)
    s4_meta.write(opt_path)

    if final_run:
        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta)
        s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
                                           s3_meta=s3_meta)
        # Record initial fitness score
        history["final_run"] = (
            s3opt_meta.scaling_MAD_spec * s4_meta.mad_s4 +
            s3opt_meta.scaling_MAD_white * s4_meta.mad_s4_binned[0])
        log.writelog(f"Final fitness value: {history["final_run"]}")
        log.writelog(f"Final white MAD: {s4_meta.mad_s4_binned[0]}")
        log.writelog(f"Final spec MAD: {s4_meta.mad_s4}\n")

    if s3opt_meta.isplots_S3opt >= 1:
        plots_s3.fitness_scores(s3opt_meta, history)

    log.closelog()

    # Delete intermediate files if requested
    if s3opt_meta.delete_final:
        if os.path.exists(s3_meta.outputdir_raw):
            shutil.rmtree(s3_meta.outputdir_raw)
        if os.path.exists(s4_meta.outputdir_raw):
            shutil.rmtree(s4_meta.outputdir_raw)

    return s3opt_meta, history, best


def optimize(s3opt_meta, log, history, best, p, eventlabel, ecf_path, stage):
    """Optimize a single parameter via parametric sweep.

    Parameters
    ----------
    s3opt_meta : eureka.lib.readECF.MetaClass
        An S3opt metadata object
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
        The stage number indicating which stage's parameters to
        optimize.

    Returns
    -------
    s3opt_meta : eureka.lib.readECF.MetaClass
        An S3opt metadata object
    log : eureka.lib.logedit.Logedit
        The log object for writing to the log file.
    history : dict
        The fitness score after optimizing each parameter.
    best : dict
        The best parameter values found so far.
    """
    # Setup Meta objects
    meta = deepcopy(s3opt_meta)
    meta.opt_param_name = p
    s3_meta, s4_meta = initialize_meta(meta, eventlabel, ecf_path=ecf_path)

    # Extract bounds for parameter(s) to optimize
    if "bounds_" + p in meta.__dict__.keys():
        bounds = meta.__dict__["bounds_" + p]
        log.writelog(f"Optimizing parameter {p} over bounds: {bounds}")
        log.writelog("Initial parameter value: " +
                     f"{getattr(s3_meta, p, getattr(s4_meta, p, None))}")
    elif "__" in p:
        # Extract default bounds for two parameters
        param_names = p.split("__")
        bounds = []
        init_vals = []
        for param in param_names:
            if "bounds_" + param in meta.__dict__.keys():
                bounds.append(meta.__dict__["bounds_" + param])
                init_vals.append(getattr(s3_meta, param,
                                         getattr(s4_meta, param, None)))
            else:
                log.writelog(f"Could not create bounds for parameter {p}. " +
                             "Please manually specify bounds in ECF. " +
                             "Skipping...")
                return s3opt_meta, log, history, best
        log.writelog(f"Optimizing parameters {p} over bounds: {bounds}")
        log.writelog(f"Initial parameter values: {init_vals}")
    else:
        log.writelog(f"No default bounds exist for parameter {p}. " +
                     "Please manually specify bounds in ECF. Skipping...")
        return s3opt_meta, log, history, best

    # Update Meta parameters with best values from previous iterations
    for key, value in best.items():
        if s3_meta is not None and hasattr(s3_meta, key):
            s3_meta.params[key] = value
            s3_meta.__dict__[key] = value
        if hasattr(s4_meta, key):
            s4_meta.params[key] = value
            s4_meta.__dict__[key] = value

    # Perform parametric sweep
    if p == "spec_hw__bg_hw":
        # Optimize both spec_hw and bg_hw simultaneously
        # Require that spec_hw < bg_hw
        best_param_value, best_fitness_value = optimizers.sweep_list_lt(
            bounds, meta, log, stage, s3_meta=s3_meta, s4_meta=s4_meta)
    elif "__" in p:
        # Optimize two independent parameters simultaneously
        best_param_value, best_fitness_value = optimizers.sweep_list_double(
            bounds, meta, log, stage, s3_meta=s3_meta, s4_meta=s4_meta)
    else:
        # Optimize single parameter
        best_param_value, best_fitness_value = optimizers.sweep_list_single(
            bounds, meta, log, stage, s3_meta=s3_meta, s4_meta=s4_meta)

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

    return s3opt_meta, log, history, best


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
    # Setup Stage 3 Meta object and overwrite certain Meta values
    s3_meta = S3MetaClass(folder=ecf_path, eventlabel=eventlabel)
    s3_meta.inputdir = meta.s2_inputdir
    s3_meta.outputdir_raw = os.path.join(meta.outputdir_raw, 'Stage3')
    s3_meta.isopt_S1 = meta.isopt_S1
    s3_meta.isopt_S3 = meta.isopt_S3
    s3_meta.isplots_S3 = meta.isplots_S3opt
    s3_meta.verbose = meta.verbose
    s3_meta.record_ypos = False
    # Create list of file segments
    s3_meta = util.readfiles(s3_meta)
    # First apply any instrument-specific defaults
    if meta.photometry:
        if meta.inst == 'miri':
            meta.set_MIRI_Photometry_defaults()
        elif meta.inst == 'nircam':
            meta.set_NIRCam_Photometry_defaults()
    else:
        if meta.inst == 'miri':
            meta.set_MIRI_defaults()
        elif meta.inst == 'nircam':
            meta.set_NIRCam_defaults()
        elif meta.inst == 'nirspec':
            meta.set_NIRSpec_defaults()
        elif meta.inst == 'niriss':
            meta.set_NIRISS_defaults()
        elif meta.inst == 'wfc3':
            meta.set_WFC3_defaults()
    # Then apply instrument-agnostic defaults
    s3_meta.set_defaults()

    # Setup Stage 4 Meta object and overwrite certain Meta values
    s4_meta = S4MetaClass(**s3_meta.__dict__)
    s4_meta.inputdir = os.path.join(meta.outputdir, 'Stage3')
    s4_meta.inputdir_raw = s4_meta.inputdir[len(meta.topdir):]
    s4_meta.outputdir_raw = os.path.join(meta.outputdir_raw, 'Stage4')
    s4_meta.isplots_S4 = meta.isplots_S3opt
    s4_meta.verbose = meta.verbose
    s4_meta.nspecchan = 1
    s4_meta.compute_ld = False
    # Apply defaults
    s4_meta.set_defaults()

    return s3_meta, s4_meta
