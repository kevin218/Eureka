import numpy as np
import matplotlib.pyplot as plt
import eureka.optimizer.objective_funcs as of


def sweep_range_single(
    eventlabel,
    bounds_var,
    meta,
    # s3_meta,
    # s4_meta,
    log,
    **kwargs,
):
    """
    Stage 3 Parametric Sweep

    Description:
    -----------
    Conducts an exhaustive search over the range of a single variable to
    minimize the objective function. It searches every value within
    the specified bounds to find the one that produces the lowest
    result from the provided objective function.

    Parameters:
    ----------
    objective_function : callable
        A function that returns a fitness score given an input. The lower the
        fitness score, the better the solution.

    bounds_var : tuple (min, max)
        The bounds for the variable being optimized. The function will search
        every integer value within these bounds.

    Returns:
    -------
    best_params : ndarray
        The optimal parameters (as an array) found for the objective function.

    best_fitness : float
        The best (lowest) fitness score found.

    Notes:
    -----
    If there is an error in calculating the fitness for a particular parameter
    set, it will print an error message and skip that set. This ensures
    robustness in the face of potentially problematic parameter sets.
    """

    best_fitness = np.inf
    best_params = None

    for val in range(bounds_var[0], bounds_var[1] + 1):
        try:
            fitness_value = of.single(
                val,
                eventlabel,
                meta,
                # s3_meta,
                # s4_meta,
                **kwargs,
            )

            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_params = val

        except Exception as e:
            log.writelog("Could not calculate fitness score for " +
                         f"{meta.opt_param_name} = {val}.")
            log.writelog(f"Error: {e}")
            continue

    return best_params, best_fitness


def sweep_list_single(
    eventlabel,
    bounds_var,
    meta,
    # s3_meta,
    # s4_meta,
    log,
    **kwargs,
):
    """
    Stage 3 Parametric Sweep

    Description:
    -----------
    Conducts an exhaustive search over the range of a single variable to
    minimize the objective function. It searches every value within
    the specified bounds to find the one that produces the lowest
    result from the provided objective function.

    Parameters:
    ----------
    objective_function : callable
        A function that returns a fitness score given an input. The lower the
        fitness score, the better the solution.

    bounds_var : tuple (min, max)
        The bounds for the variable being optimized. The function will search
        every integer value within these bounds.

    Returns:
    -------
    best_params : ndarray
        The optimal parameters (as an array) found for the objective function.

    best_fitness : float
        The best (lowest) fitness score found.

    Notes:
    -----
    If there is an error in calculating the fitness for a particular parameter
    set, it will print an error message and skip that set. This ensures
    robustness in the face of potentially problematic parameter sets.
    """

    best_fitness = np.inf
    best_params = None

    for val in bounds_var:
        try:
            fitness_value = of.single(
                val,
                eventlabel,
                meta,
                # s3_meta,
                # s4_meta,
                **kwargs,
            )

            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_params = val

        except Exception as e:
            log.writelog("Could not calculate fitness score for " +
                         f"{meta.opt_param_name} = {val}.")
            log.writelog(f"Error: {e}")
            continue

    return best_params, best_fitness


def sweep_list_double(
    eventlabel,
    bounds_var,
    meta,
    log,
    **kwargs,
):
    """
    Parametric sweep for two independent variables.

    Description:
    -----------
    Conducts an exhaustive search over the range of two independent
    variables to minimize the objective function. It searches every
    combination of values within the specified bounds for both variables
    to find the set that produces the lowest result from the provided
    objective function.

    Parameters:
    ----------

    Returns:
    -------
    best_params : ndarray
        The optimal parameters (as an array) found for the objective function.

    best_fitness : float
        The best (lowest) fitness score found.

    Notes:
    -----
    If there is an error in calculating the fitness for a particular parameter
    set, it will print an error message including the problematic parameters
    and skip that set. This ensures robustness in the face of potentially
    problematic parameter combinations.
    """

    best_fitness = np.inf
    best_params = None

    for var1 in bounds_var[0]:
        for var2 in bounds_var[1]:
            try:
                val = np.array([var1, var2])
                fitness_value = of.double(
                    val,
                    eventlabel,
                    meta,
                    **kwargs,
                )

                if fitness_value < best_fitness:
                    best_fitness = fitness_value
                    best_params = val

            except Exception as e:
                param_names = meta.opt_param_name.split("__")
                log.writelog("Could not calculate fitness score for " +
                             f"{param_names[0]} = {var1} & " +
                             f"{param_names[1]} = {var2}.")
                log.writelog(f"Error: {e}")
                continue

    return best_params, best_fitness


def sweep_list_lt(
    eventlabel,
    bounds_var,
    meta,
    # s3_meta,
    # s4_meta,
    log,
    **kwargs,
):
    """
    Parametric sweep for two interdependent variables where var1 < var2.

    Description:
    -----------
    Conducts an exhaustive search over the range of two interdependent
    variables to minimize the objective function. It searches every
    combination of values within the specified bounds for both variables
    to find the set that produces the lowest result from the provided
    objective function.

    Parameters:
    ----------

    Returns:
    -------
    best_params : ndarray
        The optimal parameters (as an array) found for the objective function.

    best_fitness : float
        The best (lowest) fitness score found.

    Notes:
    -----
    If there is an error in calculating the fitness for a particular parameter
    set, it will print an error message including the problematic parameters
    and skip that set. This ensures robustness in the face of potentially
    problematic parameter combinations.
    """

    best_fitness = np.inf
    best_params = None

    for var1 in bounds_var[0]:
        for var2 in bounds_var[1]:
            try:
                if var1 < var2:
                    val = np.array([var1, var2])
                    fitness_value = of.double(
                        val,
                        eventlabel,
                        meta,
                        # s3_meta,
                        # s4_meta,
                        **kwargs,
                    )

                    if fitness_value < best_fitness:
                        best_fitness = fitness_value
                        best_params = val

            except Exception as e:
                param_names = meta.opt_param_name.split("__")
                log.writelog("Could not calculate fitness score for " +
                             f"{param_names[0]} = {var1} & " +
                             f"{param_names[1]} = {var2}.")
                log.writelog(f"Error: {e}")
                continue

    return best_params, best_fitness


def plot_fitness_scores(best_fitness_values):
    """
    Visualizes the progress of best fitness scores across parameter sweeps.

    Parameters:
    ----------
    best_fitness_values : list of float
        A list of fitness scores corresponding to the best individual of each
        generation. The lower the fitness score, the better the individual.

    Outputs:
    -------
    A plot that illustrates the trend of best fitness scores across
    generations.

    Notes:
    -----
    - The function assumes that a lower fitness score is better.
    - The x-axis represents the generation number (starting from 1), and the
      y-axis represents the best fitness score.
    - The function has a commented-out label (`plt.ylabel`) that provides an
      example of using LaTeX syntax in plot labels. Users can uncomment and
      adjust this line to customize the y-axis label, especially if the
      fitness score represents a reduced chi-squared value.
    """
    plt.figure(3500, figsize=(8, 6))
    plt.clf()
    plt.title("Best Fitness Score vs. Generation")
    plt.plot(range(1, len(best_fitness_values) + 1), best_fitness_values)
    plt.xticks(range(1, len(best_fitness_values) + 1))  # Set ticks as integers
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Score")  # Use LaTeX syntax for underscript
    # plt.ylabel("Best Fitness Score ($\chi^2_{\mathrm{red}}$)")  # Use LaTeX
    plt.show()
