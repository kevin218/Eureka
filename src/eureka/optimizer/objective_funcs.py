import errno
import eureka.S1_detector_processing.s1_process as s1
import eureka.S2_calibrations.s2_calibrate as s2
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4
import shutil
import time


_RMTREE_ATTEMPTS = 3
_RMTREE_RETRY_DELAY = 1.0
_RMTREE_RETRYABLE_ERRNOS = {errno.ENOTEMPTY, errno.EBUSY}


def _calculate_fitness(meta, s4_meta):
    """Calculate the optimizer fitness score from Stage 4 metadata.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The optimizer metadata object. Must include ``scaling_MAD_spec`` and
        ``scaling_MAD_white``, which set the relative weights applied to the
        spectral and white-light MAD values.
    s4_meta : eureka.lib.readECF.MetaClass
        The Stage 4 metadata object returned by ``s4.genlc``. Must include
        ``mad_s4`` and ``mad_s4_binned``, where ``mad_s4_binned[0]`` is the
        white-light MAD value.

    Returns
    -------
    float
        The weighted fitness score. Lower values correspond to a better
        optimizer result.
    """
    return (
        meta.scaling_MAD_spec * s4_meta.mad_s4 +
        meta.scaling_MAD_white * s4_meta.mad_s4_binned[0])


def _remove_intermediate_outputs(meta, run_stage, s1_meta=None, s2_meta=None,
                                 s3_meta=None, s4_meta=None, log=None):
    """Delete intermediate optimizer outputs without failing the sweep.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The optimizer metadata object. If ``meta.delete_intermediate`` is
        False, no directories are removed.
    run_stage : list of bool
        Boolean flags indicating which stages were run during the current
        objective-function evaluation. Entries 1 through 4 correspond to
        Stages 1 through 4, respectively.
    s1_meta, s2_meta, s3_meta, s4_meta : eureka.lib.readECF.MetaClass; optional
        Metadata objects for each stage. For each stage that ran, the
        corresponding ``outputdir`` will be removed if present.
    log : eureka.lib.logedit.Logedit; optional
        The optimizer log. Cleanup warnings are written here when provided;
        otherwise they are printed to stdout.

    Notes
    -----
    Cleanup is intentionally best-effort. Missing directories are ignored, and
    transient filesystem errors are retried by ``_remove_output_directory``.
    Persistent cleanup errors are logged as warnings so that a completed
    parameter evaluation is not discarded only because its intermediate output
    directory could not be removed.
    """
    if not meta.delete_intermediate:
        return

    for should_remove, stage_meta in zip(run_stage[1:],
                                         [s1_meta, s2_meta, s3_meta, s4_meta]):
        if not should_remove or stage_meta is None:
            continue

        outputdir = getattr(stage_meta, 'outputdir', None)
        if outputdir is None:
            continue

        _remove_output_directory(outputdir, log=log)


def _remove_output_directory(outputdir, log=None):
    """Delete an output directory with retries for transient failures.

    Parameters
    ----------
    outputdir : str
        The directory to remove.
    log : eureka.lib.logedit.Logedit; optional
        The optimizer log. Cleanup warnings are written here when provided;
        otherwise they are printed to stdout.

    Returns
    -------
    bool
        True if the directory was removed or was already absent. False if the
        directory could not be removed after all retry attempts.

    Notes
    -----
    Some shared filesystems can briefly report ``ENOTEMPTY`` or ``EBUSY`` after
    files have been closed or removed. Those errors are retried after a short
    delay. Other ``OSError`` exceptions are treated as persistent and logged
    immediately.
    """
    for attempt in range(_RMTREE_ATTEMPTS):
        try:
            shutil.rmtree(outputdir)
            return True
        except FileNotFoundError:
            return True
        except OSError as err:
            should_retry = (
                err.errno in _RMTREE_RETRYABLE_ERRNOS and
                attempt < _RMTREE_ATTEMPTS - 1
            )
            if should_retry:
                time.sleep(_RMTREE_RETRY_DELAY)
                continue

            message = (f"WARNING: Could not delete output directory "
                       f"{outputdir}: {err}")
            if log is None:
                print(message)
            else:
                log.writelog(message)
            return False

    return False


def single(val, meta, stage, run_S3=True, **kwargs):
    """Single variable objective function.

    Parameters
    ----------
    val : float
        The variable value to be evaluated.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    stage : int
        The stage number indicating which stage's parameters to
        optimize.
    run_S3 : boolean; optional
        If True, run Stage 3. Skip Stage 3 if optimizing Stage 4.
    **kwargs : dict
        Additional keyword arguments. Can include s1_meta, s2_meta, s3_meta,
        and s4_meta to pass in existing metadata objects for each stage.

    Returns
    -------
    fitness_value : float
        The computed objective value, which is a measure of the fitness of the
        given variable value. Lower values are preferred, suggesting a better
        configuration.
    """
    run_stage = [False, False, False, False, False]
    s1_meta = None
    s2_meta = None
    s3_meta = None
    s4_meta = None
    if 's1_meta' in kwargs and kwargs['s1_meta'] is not None:
        s1_meta = kwargs['s1_meta']
        run_stage[1] = True
    if 's2_meta' in kwargs and kwargs['s2_meta'] is not None:
        s2_meta = kwargs['s2_meta']
        run_stage[2] = True
    if 's3_meta' in kwargs and kwargs['s3_meta'] is not None and run_S3:
        s3_meta = kwargs['s3_meta']
        run_stage[3] = True
    if 's4_meta' in kwargs and kwargs['s4_meta'] is not None:
        s4_meta = kwargs['s4_meta']
        run_stage[4] = True

    if meta.opt_param_name == 'bg_thresh':
        val = [val, val]

    # Set value of the variable to be optimized
    setattr(s1_meta, meta.opt_param_name, val) if stage == 1 else None
    setattr(s2_meta, meta.opt_param_name, val) if stage == 2 else None
    setattr(s3_meta, meta.opt_param_name, val) if stage == 3 else None
    setattr(s4_meta, meta.opt_param_name, val) if stage == 4 else None

    if run_stage[1]:
        s1_meta = s1.rampfitJWST(meta.eventlabel, input_meta=s1_meta)
    if run_stage[2]:
        s2_meta = s2.calibrateJWST(meta.eventlabel, input_meta=s2_meta,
                                   s1_meta=s1_meta)
    if run_stage[3]:
        s3_spec, s3_meta = s3.reduce(meta.eventlabel, input_meta=s3_meta,
                                     s2_meta=s2_meta)
    if run_stage[4]:
        s4_spec, s4_lc, s4_meta = s4.genlc(meta.eventlabel, input_meta=s4_meta,
                                           s3_meta=s3_meta)

    fitness_value = _calculate_fitness(meta, s4_meta)
    _remove_intermediate_outputs(meta, run_stage, s1_meta, s2_meta, s3_meta,
                                 s4_meta, log=kwargs.get('log'))

    return fitness_value


def double(val, meta, stage, run_S3=True, **kwargs):
    """Double variable objective function. Also works for more than two
    variables.

    Parameters
    ----------
    val : float
        The variable value to be evaluated.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    stage : int
        The stage number indicating which stage's parameters to
        optimize.
    run_S3 : boolean; optional
        If True, run Stage 3.
    **kwargs : dict
        Additional keyword arguments. Can include s1_meta, s2_meta, s3_meta,
        and s4_meta to pass in existing metadata objects for each stage.

    Returns
    -------
    fitness_value : float
        The computed objective value, which is a measure of the fitness of the
        given variable value. Lower values are preferred, suggesting a better
        configuration.
    """
    run_stage = [False, False, False, False, False]
    s1_meta = None
    s2_meta = None
    s3_meta = None
    s4_meta = None
    if 's1_meta' in kwargs and kwargs['s1_meta'] is not None:
        s1_meta = kwargs['s1_meta']
        run_stage[1] = True
    if 's2_meta' in kwargs and kwargs['s2_meta'] is not None:
        s2_meta = kwargs['s2_meta']
        run_stage[2] = True
    if 's3_meta' in kwargs and kwargs['s3_meta'] is not None and run_S3:
        s3_meta = kwargs['s3_meta']
        run_stage[3] = True
    if 's4_meta' in kwargs and kwargs['s4_meta'] is not None:
        s4_meta = kwargs['s4_meta']
        run_stage[4] = True

    # Set values of the two (or more) variables to be optimized
    param_names = meta.opt_param_name.split('__')
    assert len(param_names) == len(val), \
        f"Expected {len(param_names)} parameters for optimization, " + \
        f"got {len(val)}."
    for p, v in zip(param_names, val):
        setattr(s1_meta, p, v) if stage == 1 else None
        setattr(s2_meta, p, v) if stage == 2 else None
        setattr(s3_meta, p, v) if stage == 3 else None
        setattr(s4_meta, p, v) if stage == 4 else None

    if run_stage[1]:
        s1_meta = s1.rampfitJWST(meta.eventlabel, input_meta=s1_meta)
    if run_stage[2]:
        s2_meta = s2.calibrateJWST(meta.eventlabel, input_meta=s2_meta,
                                   s1_meta=s1_meta)
    if run_stage[3]:
        s3_spec, s3_meta = s3.reduce(meta.eventlabel, input_meta=s3_meta,
                                     s2_meta=s2_meta)
    if run_stage[4]:
        s4_spec, s4_lc, s4_meta = s4.genlc(meta.eventlabel, input_meta=s4_meta,
                                           s3_meta=s3_meta)

    fitness_value = _calculate_fitness(meta, s4_meta)
    _remove_intermediate_outputs(meta, run_stage, s1_meta, s2_meta, s3_meta,
                                 s4_meta, log=kwargs.get('log'))

    return fitness_value
