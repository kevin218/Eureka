import numpy as np
import matplotlib.pyplot as plt

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


# Parametric Optimization Code #

# Parametric sweep function
def parametric_sweep_S1(
    objective_function,
    bounds_var,
    eventlabel,
    s1_meta_GA,
    s2_meta_GA,
    s3_meta_GA,
    s4_meta_GA,
    scaling_MAD_white,
    scaling_MAD_spec,
):
    """
    Eureka! Optimization Tools: Parametric Sweep

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

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    best_fitness = float("inf")
    best_params = None

    for var1 in range(bounds_var[0], bounds_var[1] + 1):

        try:
            x = np.array([var1])
            fitness_value = objective_function(
                x,
                eventlabel,
                s1_meta_GA,
                s2_meta_GA,
                s3_meta_GA,
                s4_meta_GA,
                scaling_MAD_white,
                scaling_MAD_spec,
            )

            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_params = x

        except Exception as e:
            print(f"Could not calculate fitness score. Error: {e}")
            continue

    return best_params, best_fitness


# Parametric sweep function
def parametric_sweep_S3(
    objective_function,
    bounds_var,
    eventlabel,
    last_s2_meta_outputdir,
    s3_meta_GA,
    s4_meta_GA,
    scaling_MAD_white,
    scaling_MAD_spec,
):
    """
    Eureka! Optimization Tools: Parametric Sweep

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

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    best_fitness = float("inf")
    best_params = None

    for var1 in range(bounds_var[0], bounds_var[1] + 1):

        try:
            x = np.array([var1])
            fitness_value = objective_function(
                x,
                eventlabel,
                last_s2_meta_outputdir,
                s3_meta_GA,
                s4_meta_GA,
                scaling_MAD_white,
                scaling_MAD_spec,
            )

            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_params = x

        except Exception as e:
            print(f"Could not calculate fitness score. Error: {e}")
            continue

    return best_params, best_fitness


# Parametric sweep function
def parametric_sweep_p7thresh_S3(
    objective_function,
    bounds_var,
    eventlabel,
    last_s2_meta_outputdir,
    s3_meta_GA,
    s4_meta_GA,
    scaling_MAD_white,
    scaling_MAD_spec,
):
    """
    Eureka! Optimization Tools: Parametric Sweep

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
        every value within these bounds in increments of 5.

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

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    best_fitness = float("inf")
    best_params = None

    for var1 in range(bounds_var[0], bounds_var[1] + 1, 5):  # Increment by 5
        try:
            x = np.array([var1])
            fitness_value = objective_function(
                x,
                eventlabel,
                last_s2_meta_outputdir,
                s3_meta_GA,
                s4_meta_GA,
                scaling_MAD_white,
                scaling_MAD_spec,
            )

            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_params = x

        except Exception as e:
            print(f"Could not calculate fitness score. Error: {e}")
            continue

    return best_params, best_fitness


# Parametric sweep function
def parametric_sweep_S4(
    objective_function,
    bounds_var,
    eventlabel,
    last_s3_meta_outputdir,
    s3_meta,
    s4_meta_GA,
    scaling_MAD_white,
    scaling_MAD_spec,
):
    """
    Eureka! Optimization Tools: Parametric Sweep

    Description:
    -----------
    Conducts an exhaustive search over the range of a single variable to
    minimize the objective function. It searches every value within the
    specified bounds to find the one that produces the lowest result from
    the provided objective function.

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

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    best_fitness = float("inf")
    best_params = None

    for var1 in range(bounds_var[0], bounds_var[1] + 1):

        try:
            x = np.array([var1])
            fitness_value = objective_function(
                x,
                eventlabel,
                last_s3_meta_outputdir,
                s3_meta,
                s4_meta_GA,
                scaling_MAD_white,
                scaling_MAD_spec,
            )

            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_params = x

        except Exception as e:
            print(f"Could not calculate fitness score. Error: {e}")
            # fitness value = NaN
            continue

    return best_params, best_fitness


# Parametric sweep function for xwindow crop
def parametric_sweep_xwindow_crop(
    objective_function,
    bounds_var,
    eventlabel,
    ev,
    pixel_wave_min,
    pixel_wave_max,
    s3_meta_GA,
    s4_meta_GA,
):
    """
    Eureka! Optimization Tools: Parametric Sweep

    Description:
    -----------
    Conducts an exhaustive search over the range of a single variable to
    minimize the objective function. It searches every value within the
    specified bounds to find the one that produces the lowest result from
    the provided objective function.

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

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    best_fitness = float("inf")
    best_params = None

    for var1 in range(bounds_var[0], bounds_var[1] + 1):

        try:
            x = np.array([var1])
            fitness_value = objective_function(
                x,
                eventlabel,
                ev,
                pixel_wave_min,
                pixel_wave_max,
                s3_meta_GA,
                s4_meta_GA,
            )

            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_params = x

        except Exception as e:
            print(f"Could not calculate fitness score. Error: {e}")
            # fitness value = NaN
            continue

    return best_params, best_fitness


# Parametric sweep function for ywindow crop
def parametric_sweep_ywindow_crop(
    objective_function, bounds_var, eventlabel, ev, s3_meta_GA, s4_meta_GA
):
    """
    Eureka! Optimization Tools: Parametric Sweep

    Description:
    -----------
    Conducts an exhaustive search over the range of a single variable to
    minimize the objective function. It searches every value within the
    specified bounds to find the one that produces the lowest result from
    the provided objective function.

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

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    best_fitness = float("inf")
    best_params = None

    for var1 in range(bounds_var[0], bounds_var[1] + 1):

        try:
            x = np.array([var1])
            fitness_value = objective_function(
                x, eventlabel, ev, s3_meta_GA, s4_meta_GA
            )

            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_params = x

        except Exception as e:
            print(f"Could not calculate fitness score. Error: {e}")
            # fitness value = NaN
            continue

    return best_params, best_fitness


# Parametric sweep function for two interdependent variables
def parametric_sweep_double(
    objective_function,
    bounds_var1,
    bounds_var2,
    eventlabel,
    last_s2_meta_outputdir,
    s3_meta_GA,
    s4_meta_GA,
    scaling_MAD_white,
    scaling_MAD_spec,
):
    """
    Eureka! Optimization Tools: Parametric Sweep for Two Interdependent
    Variables

    Description:
    -----------
    Conducts an exhaustive search over the range of two interdependent
    variables to minimize the objective function. It searches every
    combination of values within the specified bounds for both variables
    to find the set that produces the lowest result from the provided
    objective function.

    Parameters:
    ----------
    objective_function : callable
        A function that returns a fitness score given an input. The lower
        the fitness score, the better the solution.

    bounds_var1 : tuple (min, max)
        The bounds for the first variable being optimized. The function will
        search every integer value within these bounds.

    bounds_var2 : tuple (min, max)
        The bounds for the second variable being optimized. The function will
        search every integer value within these bounds.

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

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    best_fitness = float("inf")
    best_params = None

    for var1 in range(bounds_var1[0], bounds_var1[1] + 1):
        for var2 in range(bounds_var2[0], bounds_var2[1] + 1):

            try:
                x = np.array([var1, var2])
                fitness_value = objective_function(
                    x,
                    eventlabel,
                    last_s2_meta_outputdir,
                    s3_meta_GA,
                    s4_meta_GA,
                    scaling_MAD_white,
                    scaling_MAD_spec,
                )

                if fitness_value < best_fitness:
                    best_fitness = fitness_value
                    best_params = x

            except Exception as e:
                print(f"bg_hw = {var1} & spec_hw = {var2}. Error: {e}")
                # fitness value = NaN
                continue

    return best_params, best_fitness


# Parametric sweep function for odd numbers within bounds
def parametric_sweep_odd_s1(
    objective_function,
    bounds_var,
    eventlabel,
    s1_meta_GA,
    s2_meta_GA,
    s3_meta_GA,
    s4_meta_GA,
    scaling_MAD_white,
    scaling_MAD_spec,
):
    """
    Eureka! Optimization Tools: Parametric Sweep for Odd Numbers

    Description:
    -----------
    Conducts an exhaustive search over the range of odd numbers within the
    specified bounds to minimize the objective function. It searches every
    odd integer value within the bounds to find the value that produces the
    lowest result from the provided objective function.

    Parameters:
    ----------
    objective_function : callable
        A function that returns a fitness score given an input. The lower the
        fitness score, the better the solution.

    bounds_var : tuple (min, max)
        The bounds for the variable being optimized. The function will search
        every odd integer value within these bounds.

    Returns:
    -------
    best_params : ndarray
        The optimal parameter (as an array) found for the objective function.

    best_fitness : float
        The best (lowest) fitness score found.

    Notes:
    -----
    - Even numbers within the specified bounds are skipped.
    - If there is an error in calculating the fitness for a particular
    parameter, it will print an error message and skip that parameter. This
    ensures robustness in the face of potentially problematic values.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    best_fitness = float("inf")
    best_params = None

    for var1 in range(bounds_var[0], bounds_var[1] + 1):
        if var1 % 2 == 0:  # If the number is even, skip the rest of this loop
            continue

        try:
            x = np.array([var1])
            fitness_value = objective_function(
                x,
                eventlabel,
                s1_meta_GA,
                s2_meta_GA,
                s3_meta_GA,
                s4_meta_GA,
                scaling_MAD_white,
                scaling_MAD_spec,
            )

            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_params = x

        except Exception as e:
            print(f"Could not calculate fitness score. Error: {e}")
            continue

    return best_params, best_fitness


# Parametric sweep function for odd numbers within bounds
def parametric_sweep_odd(
    objective_function,
    bounds_var,
    eventlabel,
    last_s2_meta_outputdir,
    s3_meta_GA,
    s4_meta_GA,
    scaling_MAD_white,
    scaling_MAD_spec,
):
    """
    Eureka! Optimization Tools: Parametric Sweep for Odd Numbers

    Description:
    -----------
    Conducts an exhaustive search over the range of odd numbers within the
    specified bounds to minimize the objective function. It searches every
    odd integer value within the bounds to find the value that produces the
    lowest result from the provided objective function.

    Parameters:
    ----------
    objective_function : callable
        A function that returns a fitness score given an input. The lower the
        fitness score, the better the solution.

    bounds_var : tuple (min, max)
        The bounds for the variable being optimized. The function will search
        every odd integer value within these bounds.

    Returns:
    -------
    best_params : ndarray
        The optimal parameter (as an array) found for the objective function.

    best_fitness : float
        The best (lowest) fitness score found.

    Notes:
    -----
    - Even numbers within the specified bounds are skipped.
    - If there is an error in calculating the fitness for a particular
    parameter, it will print an error message and skip that parameter. This
    ensures robustness in the face of potentially problematic values.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    best_fitness = float("inf")
    best_params = None

    for var1 in range(bounds_var[0], bounds_var[1] + 1):
        if var1 % 2 == 0:  # If the number is even, skip the rest of this loop
            continue

        try:
            x = np.array([var1])
            fitness_value = objective_function(
                x,
                eventlabel,
                last_s2_meta_outputdir,
                s3_meta_GA,
                s4_meta_GA,
                scaling_MAD_white,
                scaling_MAD_spec,
            )

            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_params = x

        except Exception as e:
            print(f"Could not calculate fitness score. Error: {e}")
            continue

    return best_params, best_fitness


# Parametric Sweep Function for dqmask Evaluation
def parametric_sweep_dqmask(
    objective_function,
    bounds_dqmask,
    eventlabel,
    last_s2_meta_outputdir,
    s3_meta_GA,
    s4_meta_GA,
    scaling_MAD_white,
    scaling_MAD_spec,
):

    best_fitness = float("inf")
    best_params = None
    valid_fitness_scores = [] 

    for dqmask_value in bounds_dqmask:
        try:
            print(f"Testing dqmask={dqmask_value}")

            # Pass dqmask_value as the optimization parameter to 
            # objective_function
            x = np.array([dqmask_value])
            fitness_value = objective_function(
                x,
                eventlabel,
                last_s2_meta_outputdir,
                s3_meta_GA,
                s4_meta_GA,
                scaling_MAD_white,
                scaling_MAD_spec,
            )

            print(f"Fitness value for dqmask={dqmask_value}: {fitness_value}")

            # Append successful fitness values for reference
            valid_fitness_scores.append((dqmask_value, fitness_value))

            # Update best fitness if current is lower
            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_params = dqmask_value

        except Exception as e:
            print(
                f"Could not calculate fitness score for dqmask={dqmask_value}. Error: {e}"
            )
            continue

    # If valid fitness scores were found, 
    # return the best; otherwise, return None
    if valid_fitness_scores:
        print("Valid fitness scores found:", valid_fitness_scores)
        return best_params, best_fitness
    else:
        print("No valid fitness score was found.")
        return None, None


# Parametric Sweep Function for bg_method Evaluation with Dynamic List Handling
def parametric_sweep_bg_method_s1(
    objective_function,
    methods_bg,
    eventlabel,
    s1_meta_GA,
    s2_meta_GA,
    s3_meta_GA,
    s4_meta_GA,
    scaling_MAD_white,
    scaling_MAD_spec,
):
    """
    Eureka! Optimization Tools: Parametric Sweep for bg_method Evaluation

    Description:
    -----------
    Conducts an exhaustive search over a list of possible string values 
    for `bg_method` to minimize the objective function. 
    It evaluates each option in `methods_bg` to find the one that 
    produces the lowest fitness score.

    Parameters:
    ----------
    objective_function : callable
        A function that returns a fitness score given an input. The lower the
        fitness score, the better the solution.

    methods_bg : list of str
        A list containing possible values for the `bg_method` variable. 
        This can include any number or type of strings.

    Returns:
    -------
    best_params : str
        The optimal `bg_method` value found for the objective function.

    best_fitness : float
        The best (lowest) fitness score found.

    Notes:
    -----
    If there is an error in calculating the fitness for a particular 
    `bg_method` value, it will print an error message and skip that value. 
    This ensures robustness in the face of potentially problematic 
    configurations.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    best_fitness = float("inf")
    best_params = None

    # Dynamically handle any number of `bg_method` strings in `methods_bg`
    for i in range(len(methods_bg)):
        bg_method_value = methods_bg[i]

        try:
            # Pass bg_method_value as the optimization parameter to 
            # objective_function
            x = np.array([bg_method_value])
            fitness_value = objective_function(
                x,
                eventlabel,
                s1_meta_GA,
                s2_meta_GA,
                s3_meta_GA,
                s4_meta_GA,
                scaling_MAD_white,
                scaling_MAD_spec,
            )

            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_params = bg_method_value

        except Exception as e:
            print(
                f"Could not calculate fitness score for bg_method={bg_method_value}. Error: {e}"
            )
            continue

    return best_params, best_fitness


# Parametric Sweep Function for bg_method Evaluation with Dynamic List Handling
def parametric_sweep_bg_method(
    objective_function,
    methods_bg,
    eventlabel,
    last_s2_meta_outputdir,
    s3_meta_GA,
    s4_meta_GA,
    scaling_MAD_white,
    scaling_MAD_spec,
):
    """
    Eureka! Optimization Tools: Parametric Sweep for bg_method Evaluation

    Description:
    -----------
    Conducts an exhaustive search over a list of possible string values for 
    `bg_method` to minimize the objective function. It evaluates each option in 
    `methods_bg` to find the one that produces the lowest fitness score.

    Parameters:
    ----------
    objective_function : callable
        A function that returns a fitness score given an input. The lower the
        fitness score, the better the solution.

    methods_bg : list of str
        A list containing possible values for the `bg_method` variable. 
        This can include any number or type of strings.

    Returns:
    -------
    best_params : str
        The optimal `bg_method` value found for the objective function.

    best_fitness : float
        The best (lowest) fitness score found.

    Notes:
    -----
    If there is an error in calculating the fitness for a particular 
    `bg_method` value, it will print an error message and skip that value. 
    This ensures robustness in the face of potentially problematic 
    configurations.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    best_fitness = float("inf")
    best_params = None

    # Dynamically handle any number of `bg_method` strings in `methods_bg`
    for i in range(len(methods_bg)):
        bg_method_value = methods_bg[i]

        try:
            # Pass bg_method_value as the optimization parameter to 
            # objective_function
            x = np.array([bg_method_value])
            fitness_value = objective_function(
                x,
                eventlabel,
                last_s2_meta_outputdir,
                s3_meta_GA,
                s4_meta_GA,
                scaling_MAD_white,
                scaling_MAD_spec,
            )

            if fitness_value < best_fitness:
                best_fitness = fitness_value
                best_params = bg_method_value

        except Exception as e:
            print(
                f"Could not calculate fitness score for bg_method={bg_method_value}. Error: {e}"
            )
            continue

    return best_params, best_fitness


# # Genetic Algorithm Optimization Code #
# def genetic_algorithm(population_size, generations, min_bounds, max_bounds,
#                       initialPop, mutation_rate, fitness, eventlabel,
#                       s3_meta_GA, s4_meta_GA, scaling_MAD_white, 
#                       scaling_MAD_spec, scaling_chi2red,
#                       target_fitness=None):
#     """
#     Eureka! Optimization Tools: Genetic Algorithm (GA)

#     Description:
#     -----------
#     Optimizes the objective function using a genetic algorithm, evolving a
#     population over specified generations. The genetic algorithm is a search
#     heuristic inspired by the process of natural selection. It is used to
#     find approximate solutions to optimization and search problems.

#     Parameters:
#     ----------
#     population_size : int
#         Number of individuals in the population (individual = a set of
#         parameters).

#     generations : int
#         Number of iterations to evolve the population.

#     min_bounds : ndarray
#         A list or array of minimum bounds for each parameter.

#     max_bounds : ndarray
#         A list or array of maximum bounds for each parameter.

#     initialPop : ndarray
#         A pre-selected individual (individual = a set of parameters) that is
#         included in the initial population.

#     mutation_rate : float
#         Rate at which mutations occur. Mutations introduce variability in the
#         offspring, ensuring a diverse gene pool.

#     fitness : callable
#         A function that evaluates the fitness of the population and returns a
#         tuple containing fitness values and the updated population.

#     target_fitness : float, optional
#         A threshold below which the algorithm stops. If not provided, the
#         algorithm runs for the full number of generations.

#     Returns:
#     -------
#     best_individuals : list
#         A list of the best individuals (sets of parameters) from each
#         generation.

#     best_fitness_values : list
#         A list of the best fitness scores from each generation.

#     Notes:
#     -----
#     - The algorithm uses crossover, mutation, and selection operations to
#       produce new generations.
#     - The algorithm visualizes the optimization progress with a live plot
#       showing the best fitness score vs. generation.
#     - If the target fitness is met or exceeded, the algorithm stops early
#       and prints a message indicating this.
#     - The function checks and ensures that the population size is an even
#       number.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """
#     if population_size % 2 != 0:
#         raise ValueError("Population size must be an even number.")
#     population = initial_population(population_size, min_bounds, max_bounds)
#     population[0] = initialPop  # Replace first individual with pre-selected
#     # individual

#     print(population)
#     best_individuals = []
#     best_fitness_values = []

#     def plot_fitness_scores_live(best_fitness_values):
#         plt.cla()
#         plt.plot(range(1, len(best_fitness_values) + 1), best_fitness_values)
#         plt.xticks(range(1, len(best_fitness_values) + 1))  # Set x-axis 
#         ticks
#         # as integers

#         plt.xlabel("Generation")
#         plt.ylabel("Best Fitness Score")
#         plt.title("Best Fitness Score vs. Generation")
#         plt.pause(0.01)

#     plt.figure()
#     plt.ion()
#     plt.show()

#     for generation in range(generations):
#         fitness_values, population = fitness(population)

#         # Check if population size has decreased
#         if population.shape[0] < population_size:
#             additional_population = initial_population(population_size
#                                                        - population.shape[0],
#                                                        min_bounds, 
#                                                        max_bounds)
#             population = np.vstack((population, additional_population))
#             additional_fitness = fitness(additional_population)[0]
#             fitness_values = np.concatenate((fitness_values,
#                                              additional_fitness))

#         parents, _ = selection(population,
#                                fitness_values,
#                                population_size // 2)
#         offspring = crossover(parents,
#                               (population_size // 2, population.shape[1]))
#         offspring = mutation(offspring, min_bounds, max_bounds, 
#                              mutation_rate)
#         population = np.vstack((parents, offspring))

#         # Recalculate fitness values after updating the population
#         fitness_values, population = fitness(population)

#         # best_individual = population[np.argmin(fitness_values)]
#         # best_individuals.append(best_individual)
#         # best_fitness_values.append(fitness_values
#                                      [np.argmin(fitness_values)])

#         best_fitness = np.min(fitness_values)  # Minimum fitness value
#         best_individual = population[np.argmin(fitness_values)]  # Individual
#         # with minimum fitness

#         best_fitness_values.append(best_fitness)
#         best_individuals.append(best_individual)

#         print(f"Generation {generation + 1}: Best fitness = {best_fitness}")

#         plot_fitness_scores_live(best_fitness_values)

#         if target_fitness is not None:
#             if best_fitness_values[-1] <= target_fitness:
#                 print(f"Target fitness value of {target_fitness}"
#                       f" has been met.")

#                 # Check and store the individual with the lowest score among
#                 # the best individuals
#                 min_fitness_index = np.argmin(best_fitness_values)
#                 min_fitness_individual = best_individuals[min_fitness_index]
#                 print(f"Lowest fit_val: "
#                       f"{best_fitness_values[min_fitness_index]}")
#                 print(f"Individual w/ lowest fitness: "
#                       f"{min_fitness_individual}")

#                 break

#     plt.ioff()

#     return best_individuals, best_fitness_values


# def crossover(parents, offspring_size):
#     """
#     Eureka! Optimization Tools: Crossover Operation for 
#     Genetic Algorithm (GA)

#     Description:
#     -----------
#     Performs the crossover operation on pairs of parents to produce 
#     offspring.
#     The crossover operation is a crucial step in genetic algorithms, where 
#     two
#     parent chromosomes share a portion of their genes to produce a new
#     offspring.

#     Parameters:
#     ----------
#     parents : ndarray
#         A two-dimensional array where each row represents a parent. The 
#         number
#         of columns corresponds to the number of genes in each parent.

#     offspring_size : tuple (int, int)
#         A tuple representing the desired shape (number of offsprings, number 
#         of
#         genes) for the resulting offspring population.

#     Returns:
#     -------
#     offspring : ndarray
#         A two-dimensional array where each row represents an offspring. The
#         number of columns corresponds to the number of genes in each 
#         offspring.

#     Notes:
#     -----
#     - The crossover point determines where genes are split between the two
#       parents when creating an offspring. The function currently uses a
#       single-point crossover, where genes before the crossover point are 
#       taken
#       from one parent, and genes after are taken from the other parent.
#     - This function ensures that the genes of the offspring are integers by
#       explicitly converting them before returning.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """

#     offspring = np.empty(offspring_size)
#     crossover_point = offspring_size[1] // 2
#     for i in range(offspring_size[0]):
#         parent1_idx = i % parents.shape[0]
#         parent2_idx = (i + 1) % parents.shape[0]
#         offspring[i, :crossover_point] = 
#         parents[parent1_idx, :crossover_point]
#         offspring[i, crossover_point:] = 
#         parents[parent2_idx, crossover_point:]

#         offspring[i] = offspring[i].astype(int)  # Convert to integers

#     return offspring


# def initial_population(size, min_bounds, max_bounds):
#     """
#     Eureka! Optimization Tools: Initial Population Generator for
#     Genetic Algorithm (GA)

#     Description:
#     -----------
#     Generates an initial population for the genetic algorithm. The 
#     individuals in this population are randomly generated based 
#     on specified bounds for each gene.

#     Parameters:
#     ----------
#     size : int
#         The number of individuals desired in the initial population.

#     min_bounds : ndarray or list
#         A one-dimensional array or list specifying the minimum bound for each
#         gene.

#     max_bounds : ndarray or list
#         A one-dimensional array or list specifying the maximum bound for each
#         gene.

#     Returns:
#     -------
#     population : ndarray
#         A two-dimensional array where each row represents an individual and
#         columns correspond to genes of the individual. The values are 
#         randomly generated based on the given bounds.

#     Notes:
#     -----
#     - The number of genes for each individual is determined by the length of
#     the min_bounds or max_bounds list/array.
#     - The function ensures that the values of the genes are integers by using
#     the np.random.randint function.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """
#     return np.random.randint(min_bounds, max_bounds + 1,
#                              (size, len(min_bounds)))


# def selection(population, fitness_values, num_parents):
#     """
#     Eureka! Optimization Tools: Selection Mechanism for 
#     Genetic Algorithm (GA)

#     Description:
#     -----------
#     Selects the top-performing individuals from a population based on their
#     fitness values. This function uses a deterministic selection mechanism,
#     specifically by sorting individuals according to their fitness values.

#     Parameters:
#     ----------
#     population : ndarray
#         A two-dimensional array where each row represents an individual and
#         columns correspond to genes of the individual.

#     fitness_values : ndarray or list
#         A one-dimensional array or list that holds the fitness values of each
#         individual in the population. The order of values should match the
#         order of individuals in the population array.

#     num_parents : int
#         The number of top-performing individuals to select from the 
#         population.

#     Returns:
#     -------
#     selected_population : ndarray
#         A two-dimensional array of the top-performing individuals selected
#         based on their fitness values.

#     selected_indices : ndarray
#         A one-dimensional array of indices that correspond to the selected
#         individuals in the original population array.

#     Notes:
#     -----
#     - The function selects the individuals with the lowest fitness values,
#       assuming that a lower fitness value is better.
#     - The returned `selected_indices` can be used for tracking the original
#       positions of the selected individuals in the population.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """
#     selected_indices = np.argsort(fitness_values)[:num_parents]
#     return population[selected_indices], selected_indices


# def mutation(offspring, min_bounds, max_bounds, mutation_rate):
#     """
#     Eureka! Optimization Tools: Mutation Mechanism for Genetic Algorithm (GA)

#     Description:
#     -----------
#     Introduces variability into the offspring by modifying genes of
#     individuals based on a specified mutation rate. Mutations play a pivotal
#     role in the genetic algorithm by introducing new genetic structures in
#     the population and ensuring diversity.

#     Parameters:
#     ----------
#     offspring : ndarray
#         A two-dimensional array where each row represents an offspring
#         individual and columns correspond to genes of the individual.

#     min_bounds : ndarray or list
#         A one-dimensional array or list that specifies the minimum bounds for
#         each gene. Its length should be equal to the number of genes
#         (i.e., `offspring.shape[1]`).

#     max_bounds : ndarray or list
#         A one-dimensional array or list that specifies the maximum bounds for
#         each gene. Its length should be equal to the number of genes (i.e.,
#         `offspring.shape[1]`).

#     mutation_rate : float
#         The probability that a particular gene will undergo mutation. It
#         should be between 0 (no mutation) and 1 (mutation for every gene).

#     Returns:
#     -------
#     mutated_offspring : ndarray
#         A two-dimensional array of the offspring after the mutation process.

#     Notes:
#     -----
#     - The mutation value is drawn uniformly between -1 and 1. This value is
#       then added to the original gene value.
#     - After mutation, gene values are clipped to stay within the specified
#       bounds (`min_bounds` and `max_bounds`).
#     - All gene values are then converted to integers.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """
#     for i in range(offspring.shape[0]):
#         for j in range(offspring.shape[1]):
#             if np.random.rand() < mutation_rate:
#                 mutation_value = np.random.uniform(-1, 1)
#                 offspring[i, j] += mutation_value
#                 offspring[i, j] = np.clip(offspring[i, j], min_bounds[j],
#                                           max_bounds[j])
#                 offspring[i, j] = int(offspring[i, j])  # Convert to integer
#     return offspring


# def select_best_individual(best_individuals, best_fitness_values):
#     """
#     Eureka! Optimization Tools: Selection Mechanism for Best Individual in
#     Genetic Algorithm (GA)

#     Description:
#     -----------
#     Selects the best individual (with the lowest fitness score) from a list 
#     of
#     best individuals gathered over multiple generations. It aids in
#     identifying the optimal solution post-GA optimization.

#     Parameters:
#     ----------
#     best_individuals : list of ndarray
#         A list of individuals where each individual is a one-dimensional 
#         array
#         representing the genetic structure. Each individual in this list
#         corresponds to the best individual of a particular generation.

#     best_fitness_values : list of float
#         A list of fitness scores corresponding to each individual in
#         `best_individuals`. The lower the fitness score, the better the
#         individual.

#     Returns:
#     -------
#     best_individual : ndarray
#         The individual with the lowest fitness score from `best_individuals`.

#     Notes:
#     -----
#     - The function assumes that a lower fitness score is better.
#     - Both `best_individuals` and `best_fitness_values` should have the same
#     length, as they map one-to-one.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """
#     best_index = np.argmin(best_fitness_values)
#     best_individual = best_individuals[best_index]
#     return best_individual


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
            elif "." in value and all(char.isdigit() or char == "." for char in value):
                value = float(value)

            # Add key-value pair to the dictionary
            parameters[key] = value

    return parameters
