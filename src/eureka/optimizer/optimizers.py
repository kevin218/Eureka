import numpy as np
import eureka.optimizer.objective_funcs as of


def sweep_range_single(bounds_var, meta, log, **kwargs):
    """
    Optimize single parameter using parametric sweep.  This function returns
    the best parameter and resulting fitness score after sweeping through
    every value within the specified bounds .

    Parameters:
    ----------
    bounds_var : tuple (min, max)
        The bounds for the variable being optimized. The function will search
        every integer value within these bounds.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    **kwargs : dict
        Additional keyword arguments. Can include s1_meta, s2_meta, s3_meta,
        and s4_meta to pass in existing metadata objects for each stage.

    Returns:
    -------
    best_param_value : float
         The optimal parameter value found for the objective function.
    best_fitness_value : float
        The best (lowest) fitness score found.

    Notes:
    -----
    If there is an error in calculating the fitness for a particular parameter
    set, it will print an error message and skip that set. This ensures
    robustness in the face of potentially problematic parameter sets.
    """
    best_fitness_value = np.inf
    best_param_value = None

    # Iterate over each value in the provided bounds
    for val in range(bounds_var[0], bounds_var[1] + 1):
        try:
            fitness_value = of.single(val, meta, **kwargs)
            if fitness_value < best_fitness_value:
                # Update best fitness and parameters if current is better
                best_fitness_value = fitness_value
                best_param_value = val
        except Exception as e:
            # Catch any errors during fitness calculation
            log.writelog("Could not calculate fitness score for " +
                         f"{meta.opt_param_name} = {val}.")
            log.writelog(f"Error: {e}")
            continue

    return best_param_value, best_fitness_value


def sweep_list_single(bounds_var, meta, log, **kwargs):
    """
    Optimize single parameter using parametric sweep.  This function returns
    the best parameter and resulting fitness score after sweeping through
    every value within the specified bounds .

    Parameters:
    ----------
    bounds_var : range object
        The sequence of numbers to be evaluated for the variable being
        optimized.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    **kwargs : dict
        Additional keyword arguments. Can include s1_meta, s2_meta, s3_meta,
        and s4_meta to pass in existing metadata objects for each stage.

    Returns:
    -------
    best_param_value : float
         The optimal parameter value found for the objective function.
    best_fitness_value : float
        The best (lowest) fitness score found.

    Notes:
    -----
    If there is an error in calculating the fitness for a particular parameter
    set, it will print an error message and skip that set. This ensures
    robustness in the face of potentially problematic parameter sets.
    """
    best_fitness_value = np.inf
    best_param_value = None

    # Iterate over each value in the provided bounds
    for i, val in enumerate(bounds_var):
        # Need to rerun Stage 3 first time around on Stage 4 optimization
        # Otherwise, save time by not rerunning Stage 3 again
        run_s3 = False if (i > 0 and stage == 4) else True
        try:
            fitness_value = of.single(val, meta, run_s3, **kwargs)
            if fitness_value < best_fitness_value:
                # Update best fitness and parameters if current is better
                best_fitness_value = fitness_value
                best_param_value = val
        except Exception as e:
            # Catch any errors during fitness calculation
            log.writelog("Could not calculate fitness score for " +
                         f"{meta.opt_param_name} = {val}.")
            log.writelog(f"Error: {e}")
            continue

    return best_param_value, best_fitness_value


def sweep_list_double(bounds_var, meta, log, stage, **kwargs):
    """
    Optimize two independent variables using parametric sweep.  This function
    returns the best parameter and resulting fitness score after sweeping
    through every value within the specified bounds .

    Parameters:
    ----------
    bounds_var : list of range objects
        The sequence of numbers to be evaluated for the variable being
        optimized.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    stage : int
        The stage number indicating which stage's parameters to
        optimize.
    **kwargs : dict
        Additional keyword arguments. Can include s1_meta, s2_meta, s3_meta,
        and s4_meta to pass in existing metadata objects for each stage.

    Returns:
    -------
    best_param_value : float
         The optimal parameter value found for the objective function.
    best_fitness_value : float
        The best (lowest) fitness score found.

    Notes:
    -----
    If there is an error in calculating the fitness for a particular parameter
    set, it will print an error message and skip that set. This ensures
    robustness in the face of potentially problematic parameter sets.
    """
    best_fitness_value = np.inf
    best_param_value = None

    # Iterate over each value in the provided bounds
    for i, var1 in enumerate(bounds_var[0]):
        for j, var2 in enumerate(bounds_var[1]):
            # Need to rerun Stage 3 first time around on Stage 4 optimization
            # Otherwise, save time by not rerunning Stage 3 again
            run_s3 = False if (stage == 4 and (i > 0 or j > 0)) else True
            try:
                val = np.array([var1, var2])
                fitness_value = of.double(val, meta, run_s3, **kwargs)
                if fitness_value < best_fitness_value:
                    # Update best fitness and parameters if current is better
                    best_fitness_value = fitness_value
                    best_param_value = val
            except Exception as e:
                # Catch any errors during fitness calculation
                param_names = meta.opt_param_name.split("__")
                log.writelog("Could not calculate fitness score for " +
                             f"{param_names[0]} = {var1} & " +
                             f"{param_names[1]} = {var2}.")
                log.writelog(f"Error: {e}")
                continue

    return best_param_value, best_fitness_value


def sweep_list_lt(bounds_var, meta, log, **kwargs):
    """
    Parametric sweep for two interdependent variables where var1 < var2.

    Optimize two interdependent variables (where var1 < var2) using parametric
    sweep.  This function returns the best parameter and resulting fitness score
    after sweeping through every value within the specified bounds .

    Parameters:
    ----------
    bounds_var : list of range objects
        The sequence of numbers to be evaluated for the variable being
        optimized.
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    log : logedit.Logedit
        The current log.
    **kwargs : dict
        Additional keyword arguments. Can include s1_meta, s2_meta, s3_meta,
        and s4_meta to pass in existing metadata objects for each stage.

    Returns:
    -------
    best_param_value : float
         The optimal parameter value found for the objective function.
    best_fitness_value : float
        The best (lowest) fitness score found.

    Notes:
    -----
    If there is an error in calculating the fitness for a particular parameter
    set, it will print an error message and skip that set. This ensures
    robustness in the face of potentially problematic parameter sets.
    """
    best_fitness_value = np.inf
    best_param_value = None

    # Iterate over each value in the provided bounds
    for var1 in bounds_var[0]:
        for var2 in bounds_var[1]:
            try:
                if var1 < var2:
                    val = np.array([var1, var2])
                    fitness_value = of.double(val, meta, **kwargs)
                    if fitness_value < best_fitness_value:
                        # Update best fitness and parameters if current is
                        # better
                        best_fitness_value = fitness_value
                        best_param_value = val
            except Exception as e:
                # Catch any errors during fitness calculation
                param_names = meta.opt_param_name.split("__")
                log.writelog("Could not calculate fitness score for " +
                             f"{param_names[0]} = {var1} & " +
                             f"{param_names[1]} = {var2}.")
                log.writelog(f"Error: {e}")
                continue

    return best_param_value, best_fitness_value