import numpy as np
import matplotlib.pyplot as plt

import eureka.optimizer.objective_funcs as of

"""
Eureka! Optimization Tools: Parametric Sweep & Genetic Algorithm (GA)
----------------------------------------------------------------

Description:
This script provides a suite of optimization tools, combining both parametric
sweep methods and a genetic algorithm to find the optimal parameters for a
given objective function.

Functions:

1. Parametric Optimization:
    - parametric_sweep(objective_function, bounds_var):
        Conducts an exhaustive search over the range of a single variable to
        minimize the objective function.

        Inputs:
        - objective_function: A callable that returns a fitness score.
        - bounds_var: A tuple (min, max) representing the range of the
          variable.

        Outputs:
        - best_params: Numpy array of optimal parameters.
        - best_fitness: Best fitness score found.

    - parametric_sweep_double(objective_function, bounds_var1, bounds_var2):
        Similar to parametric_sweep but considers two interdependent variables.

        Inputs:
        - bounds_var1, bounds_var2: Tuples (min, max) representing the ranges
          of the variables.

    - parametric_sweep_odd(objective_function, bounds_var):
        Optimizes over the range of odd values within the specified bounds.

2. Genetic Algorithm (GA):
    - genetic_algorithm(population_size, generations, min_bounds, max_bounds,
      initialPop, mutation_rate, fitness, target_fitness=None):

        Optimizes the objective function using a genetic algorithm, evolving
          a population over specified generations.

        Inputs:
        - population_size: Number of individuals in the population (individual
          = a set of ECF inputs).
        - generations: Number of iterations to evolve the population.
        - min_bounds, max_bounds: Tuples representing the minimum and maximum
          bounds for the parameters.
        - initialPop: A pre-selected individual (individual = a set of ECF
          inputs).
        - mutation_rate: Rate at which mutations occur. Mutations are applied
          to offspring to introduce randomness.
        - fitness: A callable function that evaluates the fitness of the
          population.
        - target_fitness (optional): A threshold below which the algorithm
          stops.

        Outputs:
        - best_individuals: A list of the best individuals from each
          generation.
        - best_fitness_values: A list of the best fitness scores from each
          generation.

Utilities:
    - crossover, initial_population, selection, mutation,
      select_best_individual, plot_fitness_scores, read_inputs:
        Supporting functions for the optimizers

Usage:
To use this script, define your objective function and call either the
 parametric optimization functions or the genetic algorithm function with
 the required parameters.
The GA also visualizes the optimization progress with a live plot.

Author: Reza Ashtari
Date: 08/22/2023
"""


# Parametric sweep function
# def parametric_sweep_S1(
#     objective_function,
#     bounds_var,
#     eventlabel,
#     s1_meta_GA,
#     s2_meta_GA,
#     s3_meta,
#     s4_meta,
#     scaling_MAD_white,
#     scaling_MAD_spec,
# ):
#     """
#     Eureka! Optimization Tools: Parametric Sweep

#     Description:
#     -----------
#     Conducts an exhaustive search over the range of a single variable to
#     minimize the objective function. It searches every value within
#     the specified bounds to find the one that produces the lowest
#     result from the provided objective function.

#     Parameters:
#     ----------
#     objective_function : callable
#         A function that returns a fitness score given an input. The lower the
#         fitness score, the better the solution.

#     bounds_var : tuple (min, max)
#         The bounds for the variable being optimized. The function will search
#         every integer value within these bounds.

#     Returns:
#     -------
#     best_params : ndarray
#         The optimal parameters (as an array) found for the objective function.

#     best_fitness : float
#         The best (lowest) fitness score found.

#     Notes:
#     -----
#     If there is an error in calculating the fitness for a particular parameter
#     set, it will print an error message and skip that set. This ensures
#     robustness in the face of potentially problematic parameter sets.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """

#     best_fitness = np.inf
#     best_params = None

#     for var1 in range(bounds_var[0], bounds_var[1] + 1):

#         try:
#             x = np.array([var1])
#             fitness_value = objective_function(
#                 x,
#                 eventlabel,
#                 s1_meta_GA,
#                 s2_meta_GA,
#                 s3_meta,
#                 s4_meta,
#                 scaling_MAD_white,
#                 scaling_MAD_spec,
#             )

#             if fitness_value < best_fitness:
#                 best_fitness = fitness_value
#                 best_params = x

#         except Exception as e:
#             print(f"Could not calculate fitness score. Error: {e}")
#             continue

#     return best_params, best_fitness


def sweep_range_single(
    eventlabel,
    bounds_var,
    meta,
    s3_meta,
    s4_meta,
    log,
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
                s3_meta,
                s4_meta,
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
    s3_meta,
    s4_meta,
    log,
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
                s3_meta,
                s4_meta,
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
    s3_meta,
    s4_meta,
    log,
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
                    s3_meta,
                    s4_meta,
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
    s3_meta,
    s4_meta,
    log,
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
                        s3_meta,
                        s4_meta,
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




# FINDME: update code below
def plot_fitness_scores(best_fitness_values):
    """
    Eureka! Optimization Tools: Visualization Mechanism for
    Genetic Algorithm (GA) Fitness Progression

    Description:
    -----------
    Visualizes the progress of the best fitness scores across generations in a
    genetic algorithm optimization. This provides insights into the evolution
    of the optimization process over time and aids in understanding the
    convergence of the GA.

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

    Author: Reza Ashtari
    Date: 08/22/2023
    """
    plt.cla()
    plt.plot(range(1, len(best_fitness_values) + 1), best_fitness_values)
    plt.xticks(range(1, len(best_fitness_values) + 1))  # Set ticks as integers
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Score")  # Use LaTeX syntax for underscript
    # plt.ylabel("Best Fitness Score ($\chi^2_{\mathrm{red}}$)")  # Use LaTeX
    plt.title("Best Fitness Score vs. Generation")
    plt.show()


def read_inputs(filename):
    """
    Eureka! Optimization Tools: Input Parsing Mechanism for Configurations

    Description:
    -----------
    This function reads an input file that contains various parameters
    specified in a key=value format. The function intelligently handles
    different data types, like integers, floats, booleans, lists, and
    strings. Moreover, it can efficiently ignore comments and process lines
    even if they contain commented sections. This functionality ensures that
    the configurations specified in the file are parsed correctly and can be
    utilized in further computational processes.

    Parameters:
    ----------
    filename : str
        The path to the input file that needs to be read. The file should
        contain parameters in the format of `key=value` with each parameter
        on a separate line.

    Outputs:
    -------
    parameters : dict
        A dictionary containing all the key-value pairs parsed from the input
        file. The dictionary values could be of type int, float, bool, list,
        or str, depending on the content of the input file.

    Notes:
    -----
    - Lines starting with a '#' or containing a '#' are considered as
      containing comments. Everything after the '#' is ignored during the
      processing of that line.
    - The function can handle quoted strings, lists specified within square
      brackets, integers, floats, and boolean values (true/false) specified as
      values.
    - If the line doesn't follow the expected `key=value` format, the function
      will print a warning indicating the problematic line.
    - The function has a mechanism to trim spaces, ensuring that unnecessary
      spaces do not affect the parsed values.
    - For list values, the function uses the `eval()` function to parse the
      string representation of the list. This approach assumes that the list
      in the file is formatted correctly.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    parameters = {}
    with open(filename, "r") as file:
        for line in file:
            # Strip off any comment from the line
            line = line.split("#", 1)[0].strip()

            # Ignore lines that are now empty
            if not line:
                continue

            # Check if the line doesn't contain '='
            if "=" not in line:
                print(f"Problematic line: {line}")
                continue

            key, value = line.split("=", 1)  # Split only once at the first '='
            key = key.strip()
            value = value.strip()

            # Remove double quotes or single quotes around strings
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]

            # Check for and convert data types
            # If value is a list, convert it from string to list
            if value.startswith("[") and value.endswith("]"):
                value = eval(value)
            # If value is a boolean string, convert to actual boolean
            elif value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            # If value is an integer
            elif value.isdigit():
                value = int(value)
            # If value is a float
            elif (
                "." in value
                and all(
                    char.isdigit() or char == "."
                    for char in value
                )
            ):
                value = float(value)

            # Add key-value pair to the dictionary
            parameters[key] = value

    return parameters

# # Parametric sweep function
# def parametric_sweep_p7thresh_S3(
#     objective_function,
#     bounds_var,
#     eventlabel,
#     last_s2_meta_outputdir,
#     s3_meta,
#     s4_meta,
#     scaling_MAD_white,
#     scaling_MAD_spec,
# ):
#     """
#     Eureka! Optimization Tools: Parametric Sweep

#     Description:
#     -----------
#     Conducts an exhaustive search over the range of a single variable to
#     minimize the objective function. It searches every value within
#     the specified bounds to find the one that produces the lowest
#     result from the provided objective function.

#     Parameters:
#     ----------
#     objective_function : callable
#         A function that returns a fitness score given an input. The lower the
#         fitness score, the better the solution.

#     bounds_var : tuple (min, max)
#         The bounds for the variable being optimized. The function will search
#         every value within these bounds in increments of 5.

#     Returns:
#     -------
#     best_params : ndarray
#         The optimal parameters (as an array) found for the objective function.

#     best_fitness : float
#         The best (lowest) fitness score found.

#     Notes:
#     -----
#     If there is an error in calculating the fitness for a particular parameter
#     set, it will print an error message and skip that set. This ensures
#     robustness in the face of potentially problematic parameter sets.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """

#     best_fitness = np.inf
#     best_params = None

#     for var1 in range(bounds_var[0], bounds_var[1] + 1, 5):  # Increment by 5
#         try:
#             x = np.array([var1])
#             fitness_value = objective_function(
#                 x,
#                 eventlabel,
#                 last_s2_meta_outputdir,
#                 s3_meta,
#                 s4_meta,
#                 scaling_MAD_white,
#                 scaling_MAD_spec,
#             )

#             if fitness_value < best_fitness:
#                 best_fitness = fitness_value
#                 best_params = x

#         except Exception as e:
#             print(f"Could not calculate fitness score. Error: {e}")
#             continue

#     return best_params, best_fitness


# # Parametric sweep function
# def parametric_sweep_S4(
#     objective_function,
#     bounds_var,
#     eventlabel,
#     last_s3_meta_outputdir,
#     s3_meta,
#     s4_meta,
#     scaling_MAD_white,
#     scaling_MAD_spec,
# ):
#     """
#     Eureka! Optimization Tools: Parametric Sweep

#     Description:
#     -----------
#     Conducts an exhaustive search over the range of a single variable to
#     minimize the objective function. It searches every value within the
#     specified bounds to find the one that produces the lowest result from
#     the provided objective function.

#     Parameters:
#     ----------
#     objective_function : callable
#         A function that returns a fitness score given an input. The lower the
#         fitness score, the better the solution.

#     bounds_var : tuple (min, max)
#         The bounds for the variable being optimized. The function will search
#         every integer value within these bounds.

#     Returns:
#     -------
#     best_params : ndarray
#         The optimal parameters (as an array) found for the objective function.

#     best_fitness : float
#         The best (lowest) fitness score found.

#     Notes:
#     -----
#     If there is an error in calculating the fitness for a particular parameter
#     set, it will print an error message and skip that set. This ensures
#     robustness in the face of potentially problematic parameter sets.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """

#     best_fitness = np.inf
#     best_params = None

#     for var1 in range(bounds_var[0], bounds_var[1] + 1):

#         try:
#             x = np.array([var1])
#             fitness_value = objective_function(
#                 x,
#                 eventlabel,
#                 last_s3_meta_outputdir,
#                 s3_meta,
#                 s4_meta,
#                 scaling_MAD_white,
#                 scaling_MAD_spec,
#             )

#             if fitness_value < best_fitness:
#                 best_fitness = fitness_value
#                 best_params = x

#         except Exception as e:
#             print(f"Could not calculate fitness score. Error: {e}")
#             # fitness value = NaN
#             continue

#     return best_params, best_fitness


# # Parametric sweep function for xwindow crop
# def parametric_sweep_xwindow_crop(
#     objective_function,
#     bounds_var,
#     eventlabel,
#     ev,
#     pixel_wave_min,
#     pixel_wave_max,
#     s3_meta,
#     s4_meta,
# ):
#     """
#     Eureka! Optimization Tools: Parametric Sweep

#     Description:
#     -----------
#     Conducts an exhaustive search over the range of a single variable to
#     minimize the objective function. It searches every value within the
#     specified bounds to find the one that produces the lowest result from
#     the provided objective function.

#     Parameters:
#     ----------
#     objective_function : callable
#         A function that returns a fitness score given an input. The lower the
#         fitness score, the better the solution.

#     bounds_var : tuple (min, max)
#         The bounds for the variable being optimized. The function will search
#         every integer value within these bounds.

#     Returns:
#     -------
#     best_params : ndarray
#         The optimal parameters (as an array) found for the objective function.

#     best_fitness : float
#         The best (lowest) fitness score found.

#     Notes:
#     -----
#     If there is an error in calculating the fitness for a particular parameter
#     set, it will print an error message and skip that set. This ensures
#     robustness in the face of potentially problematic parameter sets.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """

#     best_fitness = np.inf
#     best_params = None

#     for var1 in range(bounds_var[0], bounds_var[1] + 1):

#         try:
#             x = np.array([var1])
#             fitness_value = objective_function(
#                 x,
#                 eventlabel,
#                 ev,
#                 pixel_wave_min,
#                 pixel_wave_max,
#                 s3_meta,
#                 s4_meta,
#             )

#             if fitness_value < best_fitness:
#                 best_fitness = fitness_value
#                 best_params = x

#         except Exception as e:
#             print(f"Could not calculate fitness score. Error: {e}")
#             # fitness value = NaN
#             continue

#     return best_params, best_fitness


# # Parametric sweep function for ywindow crop
# def parametric_sweep_ywindow_crop(
#     objective_function, bounds_var, eventlabel, ev, s3_meta, s4_meta
# ):
#     """
#     Eureka! Optimization Tools: Parametric Sweep

#     Description:
#     -----------
#     Conducts an exhaustive search over the range of a single variable to
#     minimize the objective function. It searches every value within the
#     specified bounds to find the one that produces the lowest result from
#     the provided objective function.

#     Parameters:
#     ----------
#     objective_function : callable
#         A function that returns a fitness score given an input. The lower the
#         fitness score, the better the solution.

#     bounds_var : tuple (min, max)
#         The bounds for the variable being optimized. The function will search
#         every integer value within these bounds.

#     Returns:
#     -------
#     best_params : ndarray
#         The optimal parameters (as an array) found for the objective function.

#     best_fitness : float
#         The best (lowest) fitness score found.

#     Notes:
#     -----
#     If there is an error in calculating the fitness for a particular parameter
#     set, it will print an error message and skip that set. This ensures
#     robustness in the face of potentially problematic parameter sets.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """

#     best_fitness = np.inf
#     best_params = None

#     for var1 in range(bounds_var[0], bounds_var[1] + 1):

#         try:
#             x = np.array([var1])
#             fitness_value = objective_function(
#                 x, eventlabel, ev, s3_meta, s4_meta
#             )

#             if fitness_value < best_fitness:
#                 best_fitness = fitness_value
#                 best_params = x

#         except Exception as e:
#             print(f"Could not calculate fitness score. Error: {e}")
#             # fitness value = NaN
#             continue

#     return best_params, best_fitness


# # Parametric sweep function for odd numbers within bounds
# def parametric_sweep_odd_s1(
#     objective_function,
#     bounds_var,
#     eventlabel,
#     s1_meta_GA,
#     s2_meta_GA,
#     s3_meta,
#     s4_meta,
#     scaling_MAD_white,
#     scaling_MAD_spec,
# ):
#     """
#     Eureka! Optimization Tools: Parametric Sweep for Odd Numbers

#     Description:
#     -----------
#     Conducts an exhaustive search over the range of odd numbers within the
#     specified bounds to minimize the objective function. It searches every
#     odd integer value within the bounds to find the value that produces the
#     lowest result from the provided objective function.

#     Parameters:
#     ----------
#     objective_function : callable
#         A function that returns a fitness score given an input. The lower the
#         fitness score, the better the solution.

#     bounds_var : tuple (min, max)
#         The bounds for the variable being optimized. The function will search
#         every odd integer value within these bounds.

#     Returns:
#     -------
#     best_params : ndarray
#         The optimal parameter (as an array) found for the objective function.

#     best_fitness : float
#         The best (lowest) fitness score found.

#     Notes:
#     -----
#     - Even numbers within the specified bounds are skipped.
#     - If there is an error in calculating the fitness for a particular
#     parameter, it will print an error message and skip that parameter. This
#     ensures robustness in the face of potentially problematic values.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """

#     best_fitness = np.inf
#     best_params = None

#     for var1 in range(bounds_var[0], bounds_var[1] + 1):
#         if var1 % 2 == 0:  # If the number is even, skip the rest of this loop
#             continue

#         try:
#             x = np.array([var1])
#             fitness_value = objective_function(
#                 x,
#                 eventlabel,
#                 s1_meta_GA,
#                 s2_meta_GA,
#                 s3_meta,
#                 s4_meta,
#                 scaling_MAD_white,
#                 scaling_MAD_spec,
#             )

#             if fitness_value < best_fitness:
#                 best_fitness = fitness_value
#                 best_params = x

#         except Exception as e:
#             print(f"Could not calculate fitness score. Error: {e}")
#             continue

#     return best_params, best_fitness


# # Parametric sweep function for odd numbers within bounds
# def parametric_sweep_odd(
#     objective_function,
#     bounds_var,
#     eventlabel,
#     last_s2_meta_outputdir,
#     s3_meta,
#     s4_meta,
#     scaling_MAD_white,
#     scaling_MAD_spec,
# ):
#     """
#     Eureka! Optimization Tools: Parametric Sweep for Odd Numbers

#     Description:
#     -----------
#     Conducts an exhaustive search over the range of odd numbers within the
#     specified bounds to minimize the objective function. It searches every
#     odd integer value within the bounds to find the value that produces the
#     lowest result from the provided objective function.

#     Parameters:
#     ----------
#     objective_function : callable
#         A function that returns a fitness score given an input. The lower the
#         fitness score, the better the solution.

#     bounds_var : tuple (min, max)
#         The bounds for the variable being optimized. The function will search
#         every odd integer value within these bounds.

#     Returns:
#     -------
#     best_params : ndarray
#         The optimal parameter (as an array) found for the objective function.

#     best_fitness : float
#         The best (lowest) fitness score found.

#     Notes:
#     -----
#     - Even numbers within the specified bounds are skipped.
#     - If there is an error in calculating the fitness for a particular
#     parameter, it will print an error message and skip that parameter. This
#     ensures robustness in the face of potentially problematic values.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """

#     best_fitness = np.inf
#     best_params = None

#     for var1 in range(bounds_var[0], bounds_var[1] + 1):
#         if var1 % 2 == 0:  # If the number is even, skip the rest of this loop
#             continue

#         try:
#             x = np.array([var1])
#             fitness_value = objective_function(
#                 x,
#                 eventlabel,
#                 last_s2_meta_outputdir,
#                 s3_meta,
#                 s4_meta,
#                 scaling_MAD_white,
#                 scaling_MAD_spec,
#             )

#             if fitness_value < best_fitness:
#                 best_fitness = fitness_value
#                 best_params = x

#         except Exception as e:
#             print(f"Could not calculate fitness score. Error: {e}")
#             continue

#     return best_params, best_fitness


# # Parametric Sweep Function for dqmask Evaluation
# def parametric_sweep_dqmask(
#     objective_function,
#     bounds_dqmask,
#     eventlabel,
#     last_s2_meta_outputdir,
#     s3_meta,
#     s4_meta,
#     scaling_MAD_white,
#     scaling_MAD_spec,
# ):

#     best_fitness = np.inf
#     best_params = None
#     valid_fitness_scores = []

#     for dqmask_value in bounds_dqmask:
#         try:
#             print(f"Testing dqmask={dqmask_value}")

#             # Pass dqmask_value as the optimization parameter to
#             # objective_function
#             x = np.array([dqmask_value])
#             fitness_value = objective_function(
#                 x,
#                 eventlabel,
#                 last_s2_meta_outputdir,
#                 s3_meta,
#                 s4_meta,
#                 scaling_MAD_white,
#                 scaling_MAD_spec,
#             )

#             print(f"Fitness value for dqmask={dqmask_value}: {fitness_value}")

#             # Append successful fitness values for reference
#             valid_fitness_scores.append((dqmask_value, fitness_value))

#             # Update best fitness if current is lower
#             if fitness_value < best_fitness:
#                 best_fitness = fitness_value
#                 best_params = dqmask_value

#         except Exception as e:
#             print("Could not calculate fitness score for dqmask.")
#             print(f"Error: {e}")
#             continue

#     # If valid fitness scores were found,
#     # return the best; otherwise, return None
#     if valid_fitness_scores:
#         print("Valid fitness scores found:", valid_fitness_scores)
#         return best_params, best_fitness
#     else:
#         print("No valid fitness score was found.")
#         return None, None


# # Parametric Sweep Function for bg_method Evaluation with Dynamic List Handling
# def parametric_sweep_bg_method_s1(
#     objective_function,
#     methods_bg,
#     eventlabel,
#     s1_meta_GA,
#     s2_meta_GA,
#     s3_meta,
#     s4_meta,
#     scaling_MAD_white,
#     scaling_MAD_spec,
# ):
#     """
#     Eureka! Optimization Tools: Parametric Sweep for bg_method Evaluation

#     Description:
#     -----------
#     Conducts an exhaustive search over a list of possible string values
#     for `bg_method` to minimize the objective function.
#     It evaluates each option in `methods_bg` to find the one that
#     produces the lowest fitness score.

#     Parameters:
#     ----------
#     objective_function : callable
#         A function that returns a fitness score given an input. The lower the
#         fitness score, the better the solution.

#     methods_bg : list of str
#         A list containing possible values for the `bg_method` variable.
#         This can include any number or type of strings.

#     Returns:
#     -------
#     best_params : str
#         The optimal `bg_method` value found for the objective function.

#     best_fitness : float
#         The best (lowest) fitness score found.

#     Notes:
#     -----
#     If there is an error in calculating the fitness for a particular
#     `bg_method` value, it will print an error message and skip that value.
#     This ensures robustness in the face of potentially problematic
#     configurations.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """

#     best_fitness = np.inf
#     best_params = None

#     # Dynamically handle any number of `bg_method` strings in `methods_bg`
#     for i in range(len(methods_bg)):
#         bg_method_value = methods_bg[i]

#         try:
#             # Pass bg_method_value as the optimization parameter to
#             # objective_function
#             x = np.array([bg_method_value])
#             fitness_value = objective_function(
#                 x,
#                 eventlabel,
#                 s1_meta_GA,
#                 s2_meta_GA,
#                 s3_meta,
#                 s4_meta,
#                 scaling_MAD_white,
#                 scaling_MAD_spec,
#             )

#             if fitness_value < best_fitness:
#                 best_fitness = fitness_value
#                 best_params = bg_method_value

#         except Exception as e:
#             print("Could not calculate fitness score for bg_method.")
#             print(f"Error: {e}")
#             continue

#     return best_params, best_fitness


# # Parametric Sweep Function for bg_method Evaluation with Dynamic List Handling
# def parametric_sweep_bg_method(
#     objective_function,
#     methods_bg,
#     eventlabel,
#     last_s2_meta_outputdir,
#     s3_meta,
#     s4_meta,
#     scaling_MAD_white,
#     scaling_MAD_spec,
# ):
#     """
#     Eureka! Optimization Tools: Parametric Sweep for bg_method Evaluation

#     Description:
#     -----------
#     Conducts an exhaustive search over a list of possible string values for
#     `bg_method` to minimize the objective function. It evaluates each option in
#     `methods_bg` to find the one that produces the lowest fitness score.

#     Parameters:
#     ----------
#     objective_function : callable
#         A function that returns a fitness score given an input. The lower the
#         fitness score, the better the solution.

#     methods_bg : list of str
#         A list containing possible values for the `bg_method` variable.
#         This can include any number or type of strings.

#     Returns:
#     -------
#     best_params : str
#         The optimal `bg_method` value found for the objective function.

#     best_fitness : float
#         The best (lowest) fitness score found.

#     Notes:
#     -----
#     If there is an error in calculating the fitness for a particular
#     `bg_method` value, it will print an error message and skip that value.
#     This ensures robustness in the face of potentially problematic
#     configurations.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """

#     best_fitness = np.inf
#     best_params = None

#     # Dynamically handle any number of `bg_method` strings in `methods_bg`
#     for i in range(len(methods_bg)):
#         bg_method_value = methods_bg[i]

#         try:
#             # Pass bg_method_value as the optimization parameter to
#             # objective_function
#             x = np.array([bg_method_value])
#             fitness_value = objective_function(
#                 x,
#                 eventlabel,
#                 last_s2_meta_outputdir,
#                 s3_meta,
#                 s4_meta,
#                 scaling_MAD_white,
#                 scaling_MAD_spec,
#             )

#             if fitness_value < best_fitness:
#                 best_fitness = fitness_value
#                 best_params = bg_method_value

#         except Exception as e:
#             print("Could not calculate fitness score for bg_method.")
#             print(f"Error: {e}")
#             continue

#     return best_params, best_fitness



