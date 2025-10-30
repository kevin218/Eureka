import numpy as np
import eureka.S1_detector_processing.s1_process as s1
import eureka.S2_calibrations.s2_calibrate as s2
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4
import shutil


def single(val, eventlabel, meta, **kwargs):
    """Single variable objective function.

    Parameters
    ----------
    val : float
        The variable value to be evaluated.
    eventlabel : str
        The unique identifier for these data.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
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
    run_stage = [False, False, False, False]
    s1_meta = None
    s2_meta = None
    s3_meta = None
    s4_meta = None
    if 's1_meta' in kwargs:
        s1_meta = kwargs['s1_meta']
        run_stage[0] = True
    if 's2_meta' in kwargs:
        s2_meta = kwargs['s2_meta']
        run_stage[1] = True
    if 's3_meta' in kwargs:
        s3_meta = kwargs['s3_meta']
        run_stage[2] = True
    if 's4_meta' in kwargs:
        s4_meta = kwargs['s4_meta']
        run_stage[3] = True

    # Set value of the variable to be optimized
    if hasattr(s1_meta, meta.opt_param_name):
        setattr(s1_meta, meta.opt_param_name, val)
    if hasattr(s2_meta, meta.opt_param_name):
        setattr(s2_meta, meta.opt_param_name, val)
    if hasattr(s3_meta, meta.opt_param_name):
        setattr(s3_meta, meta.opt_param_name, val)
    if hasattr(s4_meta, meta.opt_param_name):
        setattr(s4_meta, meta.opt_param_name, val)

    if run_stage[0]:
        s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta)
    if run_stage[1]:
        s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta,
                                   s1_meta=s1_meta)
    if run_stage[2]:
        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta,
                                     s2_meta=s2_meta)
    if run_stage[3]:
        s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
                                           s3_meta=s3_meta)

    if meta.delete_intermediate:
        if run_stage[0]:
            shutil.rmtree(s1_meta.outputdir)
        if run_stage[1]:
            shutil.rmtree(s2_meta.outputdir)
        if run_stage[2]:
            shutil.rmtree(s3_meta.outputdir)
        if run_stage[3]:
            shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        meta.scaling_MAD_spec * s4_meta.mad_s4 +
        meta.scaling_MAD_white * s4_meta.mad_s4_binned[0])

    return fitness_value


def double(val, eventlabel, meta, **kwargs):
    """Double variable objective function. Also works for more than two
    variables.

    Parameters
    ----------
    val : float
        The variable value to be evaluated.
    eventlabel : str
        The unique identifier for these data.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
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
    run_stage = [False, False, False, False]
    s1_meta = None
    s2_meta = None
    s3_meta = None
    s4_meta = None
    if 's1_meta' in kwargs:
        s1_meta = kwargs['s1_meta']
        run_stage[0] = True
    if 's2_meta' in kwargs:
        s2_meta = kwargs['s2_meta']
        run_stage[1] = True
    if 's3_meta' in kwargs:
        s3_meta = kwargs['s3_meta']
        run_stage[2] = True
    if 's4_meta' in kwargs:
        s4_meta = kwargs['s4_meta']
        run_stage[3] = True

    # Set values of the two (or more) variables to be optimized
    param_names = meta.opt_param_name.split('__')
    assert len(param_names) == len(val), \
        f"Expected {len(param_names)} parameters for optimization, " + \
        f"got {len(val)}."
    for p, v in zip(param_names, val):
        if hasattr(s1_meta, p):
            setattr(s1_meta, p, v)
        if hasattr(s2_meta, p):
            setattr(s2_meta, p, v)
        if hasattr(s3_meta, p):
            setattr(s3_meta, p, v)
        if hasattr(s4_meta, p):
            setattr(s4_meta, p, v)

    if run_stage[0]:
        s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta)
    if run_stage[1]:
        s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta,
                                   s1_meta=s1_meta)
    if run_stage[2]:
        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta,
                                     s2_meta=s2_meta)
    if run_stage[3]:
        s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta,
                                           s3_meta=s3_meta)

    if meta.delete_intermediate:
        if run_stage[0]:
            shutil.rmtree(s1_meta.outputdir)
        if run_stage[1]:
            shutil.rmtree(s2_meta.outputdir)
        if run_stage[2]:
            shutil.rmtree(s3_meta.outputdir)
        if run_stage[3]:
            shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        meta.scaling_MAD_spec * s4_meta.mad_s4 +
        meta.scaling_MAD_white * s4_meta.mad_s4_binned[0])

    return fitness_value