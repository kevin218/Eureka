import eureka.S1_detector_processing.s1_process as s1
import eureka.S2_calibrations.s2_calibrate as s2
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4
import shutil


# Objective Function
def jump_rejection_threshold_s1(x, eventlabel, s1_meta_GA, s2_meta_GA, 
                                s3_meta_GA, s4_meta_GA, 
                                scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Thresholding
    Differences in Light Curve Analysis

    Description:
    -----------
    This function is structured to optimize the thresholding differences
    across multiple stages (specifically Stages 3, 4, and 5). By adjusting
    the `diffthresh` parameter, based on the optimization variable `x`, the
    function aims to minimize an objective that combines the chi-squared value,
    MAD of spectroscopic light curves, and MAD of binned white light curves.

    The function integrates the effect of `diffthresh` across the stages,
    processes the light curves, and subsequently computes the fitness value.

    Parameters:
    ----------
    x : list
        List of optimization variables. Here, it specifically adjusts the
        difference threshold (`diffthresh`) for the light curve stages.

    eventlabel : str
        A label or identifier for the event being analyzed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. These get
        modified within the function based on the optimization variable `x`.

    params : dict or object
        A collection of parameters, likely used in Stage 5's fitting process.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors that are used to weight the respective components in
        the objective function. They ensure that each component's contribution
        to the fitness value is appropriately adjusted.

    Outputs:
    -------
    fitness_value : float
        The computed objective value, which is a measure of the fitness of the
        given set of optimization variables `x`. Lower values are preferred,
        suggesting a better thresholding configuration.

    Notes:
    -----
    - The function updates the `diffthresh` parameter in the metadata of
      Stages 3, 4, and 5 using the optimization variable `x[0]`.
    - After adjusting the `diffthresh`, the function processes the light
      curves in each stage and computes the required metrics.
    - The objective is a weighted sum of the chi-squared value from Stage 5,
      MAD of spectroscopic light curves from Stage 4, and MAD of binned white
      light curves from Stage 4.
    - The function also performs cleanup by removing directories related to
      each stage after processing.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 1
    s1_meta_GA.jump_rejection_threshold = x[0]

    s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
    s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s1_meta.outputdir)
    shutil.rmtree(s2_meta.outputdir)
    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def bg_deg_s1(x, eventlabel, s1_meta_GA, s2_meta_GA, s3_meta_GA, s4_meta_GA, 
              scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Thresholding
    Differences in Light Curve Analysis

    Description:
    -----------
    This function is structured to optimize the thresholding differences
    across multiple stages (specifically Stages 3, 4, and 5). By adjusting
    the `diffthresh` parameter, based on the optimization variable `x`, the
    function aims to minimize an objective that combines the chi-squared value,
    MAD of spectroscopic light curves, and MAD of binned white light curves.

    The function integrates the effect of `diffthresh` across the stages,
    processes the light curves, and subsequently computes the fitness value.

    Parameters:
    ----------
    x : list
        List of optimization variables. Here, it specifically adjusts the
        difference threshold (`diffthresh`) for the light curve stages.

    eventlabel : str
        A label or identifier for the event being analyzed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. These get
        modified within the function based on the optimization variable `x`.

    params : dict or object
        A collection of parameters, likely used in Stage 5's fitting process.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors that are used to weight the respective components in
        the objective function. They ensure that each component's contribution
        to the fitness value is appropriately adjusted.

    Outputs:
    -------
    fitness_value : float
        The computed objective value, which is a measure of the fitness of the
        given set of optimization variables `x`. Lower values are preferred,
        suggesting a better thresholding configuration.

    Notes:
    -----
    - The function updates the `diffthresh` parameter in the metadata of
      Stages 3, 4, and 5 using the optimization variable `x[0]`.
    - After adjusting the `diffthresh`, the function processes the light
      curves in each stage and computes the required metrics.
    - The objective is a weighted sum of the chi-squared value from Stage 5,
      MAD of spectroscopic light curves from Stage 4, and MAD of binned white
      light curves from Stage 4.
    - The function also performs cleanup by removing directories related to
      each stage after processing.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 1
    s1_meta_GA.bg_deg = x[0]

    s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
    s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s1_meta.outputdir)
    shutil.rmtree(s2_meta.outputdir)
    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def bg_method_s1(x, eventlabel, s1_meta_GA, s2_meta_GA, s3_meta_GA, s4_meta_GA, 
                 scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Background Method 
    Selection in Light Curve Analysis

    Description:
    -----------
    This function optimizes the background subtraction method (bg_method) 
    across
    multiple stages (specifically Stages 3, 4, and 5). By adjusting the 
    bg_method
    parameter based on the optimization variable x, the function aims to 
    minimize
    an objective that combines the chi-squared value, MAD of spectroscopic 
    light
    curves, and MAD of binned white light curves.

    The function integrates the effect of bg_method across the stages,
    processes the light curves, and subsequently computes the fitness value.

    Parameters:
    ----------
    x : list
        List of optimization variables. Here, it specifically sets the 
        background
        subtraction method (bg_method) for the light curve stages, given as a 
        string.

    eventlabel : str
        A label or identifier for the event being analyzed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. These get
        modified within the function based on the optimization variable x.

    params : dict or object
        A collection of parameters, likely used in Stage 5's fitting process.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors that are used to weight the respective components in
        the objective function. They ensure that each component's contribution
        to the fitness value is appropriately adjusted.

    Outputs:
    -------
    fitness_value : float
        The computed objective value, which is a measure of the fitness of the
        given set of optimization variables x. Lower values are preferred,
        suggesting a better background method configuration.

    Notes:
    -----
    - The function updates the bg_method parameter in the metadata of
      Stages 3, 4, and 5 using the optimization variable x[0].
    - After adjusting bg_method, the function processes the light
      curves in each stage and computes the required metrics.
    - The objective is a weighted sum of the chi-squared value from Stage 5,
      MAD of spectroscopic light curves from Stage 4, and MAD of binned white
      light curves from Stage 4.
    - The function also performs cleanup by removing directories related to
      each stage after processing.

    Author: Reza Ashtari
    Date: 08/22/2023
    """
    def remove_apostrophes(s):
        return s.replace("'", "")

    # Set bg_method in metadata for each stage based on the input string
    bg_method_value = remove_apostrophes(x[0])  # Assumes x[0] is string
    # bg_method_value = x[0]  # Assuming x[0] is a string, e.g. 'std'
    s1_meta_GA.bg_method = bg_method_value

    s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
    s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s1_meta.outputdir)
    shutil.rmtree(s2_meta.outputdir)
    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    # Calculate the fitness value based on the specified scaling factors
    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white * (sum(s4_meta.mad_s4_binned) / 
                             len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def p3thresh_s1(x, eventlabel, s1_meta_GA, s2_meta_GA, s3_meta_GA, s4_meta_GA, 
                scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Thresholding
    Differences in Light Curve Analysis

    Description:
    -----------
    This function is structured to optimize the thresholding differences
    across multiple stages (specifically Stages 3, 4, and 5). By adjusting
    the `diffthresh` parameter, based on the optimization variable `x`, the
    function aims to minimize an objective that combines the chi-squared value,
    MAD of spectroscopic light curves, and MAD of binned white light curves.

    The function integrates the effect of `diffthresh` across the stages,
    processes the light curves, and subsequently computes the fitness value.

    Parameters:
    ----------
    x : list
        List of optimization variables. Here, it specifically adjusts the
        difference threshold (`diffthresh`) for the light curve stages.

    eventlabel : str
        A label or identifier for the event being analyzed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. These get
        modified within the function based on the optimization variable `x`.

    params : dict or object
        A collection of parameters, likely used in Stage 5's fitting process.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors that are used to weight the respective components in
        the objective function. They ensure that each component's contribution
        to the fitness value is appropriately adjusted.

    Outputs:
    -------
    fitness_value : float
        The computed objective value, which is a measure of the fitness of the
        given set of optimization variables `x`. Lower values are preferred,
        suggesting a better thresholding configuration.

    Notes:
    -----
    - The function updates the `diffthresh` parameter in the metadata of
      Stages 3, 4, and 5 using the optimization variable `x[0]`.
    - After adjusting the `diffthresh`, the function processes the light
      curves in each stage and computes the required metrics.
    - The objective is a weighted sum of the chi-squared value from Stage 5,
      MAD of spectroscopic light curves from Stage 4, and MAD of binned white
      light curves from Stage 4.
    - The function also performs cleanup by removing directories related to
      each stage after processing.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 1
    s1_meta_GA.p3thresh = x[0]

    s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
    s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s1_meta.outputdir)
    shutil.rmtree(s2_meta.outputdir)
    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def window_len_s1(x, eventlabel, s1_meta_GA, s2_meta_GA, s3_meta_GA, 
                  s4_meta_GA, scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Window Length Parameter
    Optimization in Light Curve Analysis

    Description:
    -----------
    This function is geared towards the optimization of the `window_len`
    parameter across Stages 3, 4, and 5 in the light curve analysis process.
    The `window_len` parameter, represented by the optimization variable
    `x[0]`, gets assigned in the metadata of each respective stage. Subsequent
    to this configuration, the light curves are generated and processed. An
    objective or "fitness" value is then computed based on a combination of the
    chi-squared value, MAD of spectroscopic light curves, and MAD of binned
    white light curves.

    Parameters:
    ----------
    x : list
        A list of optimization variables where `x[0]` specifies the value to
        be set for the `window_len` parameter.

    eventlabel : str
        A distinct label or identifier for the specific event being processed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects pertaining to Stages 3, 4, and 5, respectively. Within
        the function, these objects are modified based on the value of `x[0]`.

    params : dict or object
        A collection of parameters, likely utilized in the fitting procedure
        of Stage 5.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling coefficients designated to balance the contributions of
        various components in the objective function. These ensure that the
        impact of each component on the fitness value is harmonized.

    Outputs:
    -------
    fitness_value : float
        A computed objective score, indicative of the efficacy of the proposed
        `window_len` value in `x[0]`. A more minimal score is indicative of a
        better or more optimal configuration for the `window_len` parameter.

    Notes:
    -----
    - The function assigns the `window_len` parameter for the metadata of
      Stages 3, 4, and 5 based on `x[0]`.
    - The objective score embodies a weighted aggregate of the chi-squared
    value from Stage 5, MAD of spectroscopic light curves from Stage 4, and
      MAD of the binned white light curves also from Stage 4.
    - For housekeeping, directories associated with each of the stages are
      deleted post-processing.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 1
    s1_meta_GA.window_len = x[0]

    s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
    s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s1_meta.outputdir)
    shutil.rmtree(s2_meta.outputdir)
    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def expand_mask_s1(x, eventlabel, s1_meta_GA, s2_meta_GA, s3_meta_GA, 
                   s4_meta_GA, scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Thresholding
    Differences in Light Curve Analysis

    Description:
    -----------
    This function is structured to optimize the thresholding differences
    across multiple stages (specifically Stages 3, 4, and 5). By adjusting
    the `diffthresh` parameter, based on the optimization variable `x`, the
    function aims to minimize an objective that combines the chi-squared value,
    MAD of spectroscopic light curves, and MAD of binned white light curves.

    The function integrates the effect of `diffthresh` across the stages,
    processes the light curves, and subsequently computes the fitness value.

    Parameters:
    ----------
    x : list
        List of optimization variables. Here, it specifically adjusts the
        difference threshold (`diffthresh`) for the light curve stages.

    eventlabel : str
        A label or identifier for the event being analyzed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. These get
        modified within the function based on the optimization variable `x`.

    params : dict or object
        A collection of parameters, likely used in Stage 5's fitting process.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors that are used to weight the respective components in
        the objective function. They ensure that each component's contribution
        to the fitness value is appropriately adjusted.

    Outputs:
    -------
    fitness_value : float
        The computed objective value, which is a measure of the fitness of the
        given set of optimization variables `x`. Lower values are preferred,
        suggesting a better thresholding configuration.

    Notes:
    -----
    - The function updates the `diffthresh` parameter in the metadata of
      Stages 3, 4, and 5 using the optimization variable `x[0]`.
    - After adjusting the `diffthresh`, the function processes the light
      curves in each stage and computes the required metrics.
    - The objective is a weighted sum of the chi-squared value from Stage 5,
      MAD of spectroscopic light curves from Stage 4, and MAD of binned white
      light curves from Stage 4.
    - The function also performs cleanup by removing directories related to
      each stage after processing.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 1
    s1_meta_GA.expand_mask = x[0]

    s1_meta = s1.rampfitJWST(eventlabel, input_meta=s1_meta_GA)
    s2_meta = s2.calibrateJWST(eventlabel, input_meta=s2_meta_GA)
    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s1_meta.outputdir)
    shutil.rmtree(s2_meta.outputdir)
    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def xwindow_crop(x, eventlabel, ev, pixel_wave_min, pixel_wave_max, s3_meta_GA,
                 s4_meta_GA):
    """
    Eureka! Optimization Tools: Objective Function for X-Window Cropping in
    Light Curve Analysis

    Description:
    -----------
    This function defines an objective to optimize the box extraction of
    light curves, particularly pertaining to Eureka! Stages 3 and 4. By
    adjusting the `xwindow` based on the given optimization parameters `x`,
    the function tries to minimize the Mean Absolute Deviation (MAD) of the
    White Light Curve.

    The objective is a composite measure, influenced by MAD values from both
    Stage 3 and Stage 4 light curves. Proper bounds are enforced on the
    `xwindow` to ensure the cropping does not exceed the allowable limits.

    Parameters:
    ----------
    x : list
        List of optimization variables. In this function, it specifically
        modifies the xwindow boundaries for cropping.

    eventlabel : str
        A label or identifier for the event being analyzed.

    ev : object
        An object that contains information about the event, especially
        the initial `xwindow` values.

    pixel_wave_min : float
        The minimum allowable boundary for cropping on the x-axis.

    pixel_wave_max : float
        The maximum allowable boundary for cropping on the x-axis.

    s3_meta_GA : object
        Metadata for Stage 3, which gets modified within this function.

    s4_meta_GA : object
        Metadata for Stage 4, not modified directly within this function but
        used for further processing.

    Outputs:
    -------
    fitness_value : float
        The computed objective value which represents the fitness of the
        given set of optimization variables `x`. Lower values are better,
        indicating a better cropping configuration.

    Notes:
    -----
    - The function is designed to return an infinite fitness value (i.e., a
      very bad solution) in case of any errors, such as if the `xwindow`
      boundaries are violated.

      The objective formula is given by a fitness value INDEPENDENT from the
      other objective functions used = `0.1*s3_meta.mad_s3 + s4_meta.mad_s4 +
      sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned)`, which weighs
      contributions from Stage 3 and Stage 4 light curves.

    - There are commented-out lines that appear to remove directories once
      processing is done. These lines might be for cleanup purposes, but they
      are currently inactive.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    try:

        # Define Variables to be optimized
        # Stage 3
        s3_meta_GA.xwindow = [ev.xwindow[0] + x[0], ev.xwindow[1] - x[0]]

        # Check conditions for xwindow values

        # if s3_meta_GA.xwindow[0] > pixel_wave_min-3 or \
        #    s3_meta_GA.xwindow[1] < pixel_wave_max+3:

        if s3_meta_GA.xwindow[0] > pixel_wave_min or \
           s3_meta_GA.xwindow[1] < pixel_wave_max:

            raise ValueError("xwindow boundaries violated")

        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
        s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                           s3_meta=s3_meta)
        # shutil.rmtree(s3_meta.outputdir)
        # shutil.rmtree(s4_meta.outputdir)

        fitness_value = (
            0.05 * s3_meta.mad_s3 +
            0.5 * s4_meta.mad_s4 +
            1.0 * sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned)
        )

        return fitness_value

    except:

        fitness_value = float('inf')

        return fitness_value

    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)


# Objective Function
def ywindow_crop(x, eventlabel, ev, s3_meta_GA, s4_meta_GA):
    """
    Eureka! Optimization Tools: Objective Function for X-Window Cropping in
    Light Curve Analysis

    Description:
    -----------
    This function defines an objective to optimize the box extraction of
    light curves, particularly pertaining to Eureka! Stages 3 and 4. By
    adjusting the `ywindow` based on the given optimization parameters `y`,
    the function tries to minimize the Mean Absolute Deviation (MAD) of the
    White Light Curve.

    The objective is a composite measure, influenced by MAD values from both
    Stage 3 and Stage 4 light curves. Proper bounds are enforced on the
    `ywindow` to ensure the cropping does not exceed the allowable limits.

    Parameters:
    ----------
    x : list
        List of optimization variables. In this function, it specifically
        modifies the ywindow boundaries for cropping.

    eventlabel : str
        A label or identifier for the event being analyzed.

    ev : object
        An object that contains information about the event, especially
        the initial `ywindow` values.

    s3_meta_GA : object
        Metadata for Stage 3, which gets modified within this function.

    s4_meta_GA : object
        Metadata for Stage 4, not modified directly within this function but
        used for further processing.

    Outputs:
    -------
    fitness_value : float
        The computed objective value which represents the fitness of the
        given set of optimization variables `x`. Lower values are better,
        indicating a better cropping configuration.

    Notes:
    -----
    - The function is designed to return an infinite fitness value (i.e., a
      very bad solution) in case of any errors, such as if the `ywindow`
      boundaries are violated.

      The objective formula is given by a fitness value INDEPENDENT from the
      other objective functions used = `0.1*s3_meta.mad_s3 + s4_meta.mad_s4 +
      sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned)`, which weighs
      contributions from Stage 3 and Stage 4 light curves.

    - There are commented-out lines that appear to remove directories once
      processing is done. These lines might be for cleanup purposes, but they
      are currently inactive.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    try:

        # Define Variables to be optimized
        # Stage 3
        s3_meta_GA.ywindow = [ev.ywindow[0] + x[0], ev.ywindow[1] - x[0]]

        # Check conditions for ywindow values --> Maybe Later, Gator

        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
        s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                           s3_meta=s3_meta)
        # shutil.rmtree(s3_meta.outputdir)
        # shutil.rmtree(s4_meta.outputdir)

        fitness_value = (
            0.05 * s3_meta.mad_s3 +
            0.5 * s4_meta.mad_s4 +
            1.0 * sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned)
        )

        return fitness_value

    except:

        fitness_value = float('inf')

        return fitness_value

    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)


# # Objective Function
# def diffthresh(x, eventlabel, s3_meta_GA, s4_meta_GA,
#                scaling_MAD_white, scaling_MAD_spec):
#     """
#     Eureka! Optimization Tools: Objective Function for Thresholding
#     Differences in Light Curve Analysis

#     Description:
#     -----------
#     This function is structured to optimize the thresholding differences
#     across multiple stages (specifically Stages 3, 4, and 5). By adjusting
#     the `diffthresh` parameter, based on the optimization variable `x`, the
#     function aims to minimize an objective that combines the chi-squared 
#     value, MAD of spectroscopic light curves, and MAD of binned white light 
#     curves.

#     The function integrates the effect of `diffthresh` across the stages,
#     processes the light curves, and subsequently computes the fitness value.

#     Parameters:
#     ----------
#     x : list
#         List of optimization variables. Here, it specifically adjusts the
#         difference threshold (`diffthresh`) for the light curve stages.

#     eventlabel : str
#         A label or identifier for the event being analyzed.

#     s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
#         Metadata objects for Stages 3, 4, and 5, respectively. These get
#         modified within the function based on the optimization variable `x`.

#     params : dict or object
#         A collection of parameters, likely used in Stage 5's fitting process.

#     scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
#         Scaling factors that are used to weight the respective components in
#         the objective function. They ensure that each component's 
#         contribution to the fitness value is appropriately adjusted.

#     Outputs:
#     -------
#     fitness_value : float
#         The computed objective value, which is a measure of the fitness of 
#         the given set of optimization variables `x`. Lower values are 
#         preferred, suggesting a better thresholding configuration.

#     Notes:
#     -----
#     - The function updates the `diffthresh` parameter in the metadata of
#       Stages 3, 4, and 5 using the optimization variable `x[0]`.
#     - After adjusting the `diffthresh`, the function processes the light
#       curves in each stage and computes the required metrics.
#     - The objective is a weighted sum of the chi-squared value from Stage 5,
#       MAD of spectroscopic light curves from Stage 4, and MAD of binned white
#       light curves from Stage 4.
#     - The function also performs cleanup by removing directories related to
#       each stage after processing.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """

#     # Define Variables to be optimized
#     # Stage 3
#     s3_meta_GA.diffthresh = x[0]
#     # Stage 4
#     s4_meta_GA.diffthresh = x[0]

#     s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
#                                        s3_meta=s3_meta)

#     shutil.rmtree(s3_meta.outputdir)
#     shutil.rmtree(s4_meta.outputdir)

#     fitness_value = (
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white *
#         (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
#     )

#     # fitness_value = (scaling_chi2red * s5_meta.chi2red) + \
#     #                 (scaling_MAD_spec * s4_meta.mad_s4) + \
#     #                 (scaling_MAD_white * \
#     #                 (sum(s4_meta.mad_s4_binned) / \
#     #                 len(s4_meta.mad_s4_binned)))

#     return fitness_value


# Objective Function
def dqmask(x, eventlabel, last_s2_meta_outputdir, s3_meta_GA, s4_meta_GA, 
           scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Thresholding
    Differences in Light Curve Analysis using dqmask Logic.

    Description:
    -----------
    This function optimizes the dqmask flag across multiple stages 
    (Stages 3, 4, and 5).
    By adjusting the dqmask parameter based on the boolean value of `x[0]`, the
    function aims to minimize an objective that combines the chi-squared value,
    MAD of spectroscopic light curves, and MAD of binned white light curves.

    Parameters:
    ----------
    x : list
        List of optimization variables. 
        Here, it specifically adjusts the dqmask
        (data quality mask) flag for the light curve stages, 
        interpreted as a boolean.

    eventlabel : str
        A label or identifier for the event being analyzed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. 
        These get modified within
        the function based on the optimization variable `x`.

    params : dict or object
        A collection of parameters, likely used in Stage 5's fitting process.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors that are used to weight the respective components in
        the objective function. They ensure that each component's contribution
        to the fitness value is appropriately adjusted.

    Outputs:
    -------
    fitness_value : float
        The computed objective value, which is a measure of the fitness of the
        given set of optimization variables `x`. Lower values are preferred,
        suggesting a better dqmask configuration.

    Notes:
    -----
    - The function updates the dqmask parameter in the metadata of Stages 3, 4, 
      and 5 using the
      boolean value of `x[0]`.
    - After adjusting dqmask, the function processes the 
      light curves in each stage
      and computes the required metrics.
    - The objective is a weighted sum of the chi-squared value from Stage 5,
      MAD of spectroscopic light curves from Stage 4, and MAD of binned white
      light curves from Stage 4.
    - The function also performs cleanup by removing directories related to
      each stage after processing.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Convert the optimization variable to boolean for dqmask
    dqmask_value = bool(x[0])

    # Update dqmask in metadata for each stage
    s3_meta_GA.dqmask = dqmask_value
    s4_meta_GA.dqmask = dqmask_value

    s3_meta_GA.inputdir = last_s2_meta_outputdir

    # Perform the light curve processing for each stage
    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA, 
                                       s3_meta=s3_meta)

    # Clean up the directories for each stage
    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    # Calculate the fitness value based on the specified scaling factors
    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white * 
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def expand(x, eventlabel, last_s2_meta_outputdir, s3_meta_GA, s4_meta_GA, 
           scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Thresholding
    Differences in Light Curve Analysis

    Description:
    -----------
    This function is structured to optimize the thresholding differences
    across multiple stages (specifically Stages 3, 4, and 5). By adjusting
    the `diffthresh` parameter, based on the optimization variable `x`, the
    function aims to minimize an objective that combines the chi-squared value,
    MAD of spectroscopic light curves, and MAD of binned white light curves.

    The function integrates the effect of `diffthresh` across the stages,
    processes the light curves, and subsequently computes the fitness value.

    Parameters:
    ----------
    x : list
        List of optimization variables. Here, it specifically adjusts the
        difference threshold (`diffthresh`) for the light curve stages.

    eventlabel : str
        A label or identifier for the event being analyzed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. These get
        modified within the function based on the optimization variable `x`.

    params : dict or object
        A collection of parameters, likely used in Stage 5's fitting process.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors that are used to weight the respective components in
        the objective function. They ensure that each component's contribution
        to the fitness value is appropriately adjusted.

    Outputs:
    -------
    fitness_value : float
        The computed objective value, which is a measure of the fitness of the
        given set of optimization variables `x`. Lower values are preferred,
        suggesting a better thresholding configuration.

    Notes:
    -----
    - The function updates the `diffthresh` parameter in the metadata of
      Stages 3, 4, and 5 using the optimization variable `x[0]`.
    - After adjusting the `diffthresh`, the function processes the light
      curves in each stage and computes the required metrics.
    - The objective is a weighted sum of the chi-squared value from Stage 5,
      MAD of spectroscopic light curves from Stage 4, and MAD of binned white
      light curves from Stage 4.
    - The function also performs cleanup by removing directories related to
      each stage after processing.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 3
    s3_meta_GA.expand = x[0]
    # Stage 4
    s4_meta_GA.expand = x[0]

    s3_meta_GA.inputdir = last_s2_meta_outputdir

    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def bg_thresh(x, eventlabel, last_s2_meta_outputdir, s3_meta_GA, s4_meta_GA, 
              scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Thresholding
    Differences in Light Curve Analysis

    Description:
    -----------
    This function is structured to optimize the thresholding differences
    across multiple stages (specifically Stages 3, 4, and 5). By adjusting
    the `diffthresh` parameter, based on the optimization variable `x`, the
    function aims to minimize an objective that combines the chi-squared value,
    MAD of spectroscopic light curves, and MAD of binned white light curves.

    The function integrates the effect of `diffthresh` across the stages,
    processes the light curves, and subsequently computes the fitness value.

    Parameters:
    ----------
    x : list
        List of optimization variables. Here, it specifically adjusts the
        difference threshold (`diffthresh`) for the light curve stages.

    eventlabel : str
        A label or identifier for the event being analyzed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. These get
        modified within the function based on the optimization variable `x`.

    params : dict or object
        A collection of parameters, likely used in Stage 5's fitting process.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors that are used to weight the respective components in
        the objective function. They ensure that each component's contribution
        to the fitness value is appropriately adjusted.

    Outputs:
    -------
    fitness_value : float
        The computed objective value, which is a measure of the fitness of the
        given set of optimization variables `x`. Lower values are preferred,
        suggesting a better thresholding configuration.

    Notes:
    -----
    - The function updates the `diffthresh` parameter in the metadata of
      Stages 3, 4, and 5 using the optimization variable `x[0]`.
    - After adjusting the `diffthresh`, the function processes the light
      curves in each stage and computes the required metrics.
    - The objective is a weighted sum of the chi-squared value from Stage 5,
      MAD of spectroscopic light curves from Stage 4, and MAD of binned white
      light curves from Stage 4.
    - The function also performs cleanup by removing directories related to
      each stage after processing.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 3
    s3_meta_GA.bgthresh = x[0]
    # Stage 4
    s4_meta_GA.bgthresh = x[0]

    s3_meta_GA.inputdir = last_s2_meta_outputdir

    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def bg_hw_spec_hw(x, eventlabel, last_s2_meta_outputdir, s3_meta_GA, 
                  s4_meta_GA, scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Spectral and Background
    Half-Width Optimization in Light Curve Analysis

    Description:
    -----------
    This function aims to optimize the spectral (`spec_hw`) and background
    (`bg_hw`) half-widths across Stages 3, 4, and 5 in the light curve
    analysis. It checks the feasibility of the optimization variables `x`,
    ensuring that the background half-width is not smaller than the spectral
    half-width. Upon passing this condition, it modifies the metadata for
    each stage, processes the light curves, and computes a fitness value based
    on chi-squared, MAD of spectroscopic light curves, and MAD of binned white
    light curves.

    If the condition is not met (i.e., `bg_hw` is smaller than `spec_hw`), it
    returns an infinite fitness value, indicating an infeasible solution.

    Parameters:
    ----------
    x : list
        List of optimization variables. `x[0]` adjusts the background
        half-width (`bg_hw`) and `x[1]` adjusts the spectral half-width
        (`spec_hw`).

    eventlabel : str
        A label or identifier for the event being analyzed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. These get
        modified within the function based on the optimization variables `x`.

    params : dict or object
        A collection of parameters, probably used in Stage 5's fitting process.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors used to weight the respective components in the
        objective function, ensuring that each component's influence on the
        fitness value is correctly balanced.

    Outputs:
    -------
    fitness_value : float
        The computed objective value, representing the fitness of the given
        set of optimization variables `x`. A lower value is desired, suggesting
        an optimal configuration of spectral and background half-widths.

    Notes:
    -----
    - The function first ensures that the background half-width (`bg_hw`) is
      not smaller than the spectral half-width (`spec_hw`). If this condition
      fails, an infinite fitness value is returned.
    - If the condition is met, the function updates the `bg_hw` and `spec_hw`
      parameters in the metadata of Stages 3, 4, and 5 using `x[0]` and `x[1]`,
      respectively.
    - The objective is a weighted sum of the chi-squared value from Stage 5,
      MAD of spectroscopic light curves from Stage 4, and MAD of binned white
      light curves from Stage 4.
    - After processing, directories related to each stage are removed for
      cleanup.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    if x[0] >= x[1]:

        # Define Variables to be optimized
        # Stage 3
        s3_meta_GA.bg_hw = x[0]
        s3_meta_GA.spec_hw = x[1]

        # Stage 4
        s4_meta_GA.bg_hw = x[0]
        s4_meta_GA.spec_hw = x[1]
        
        s3_meta_GA.inputdir = last_s2_meta_outputdir

        s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
        s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                           s3_meta=s3_meta)

        shutil.rmtree(s3_meta.outputdir)
        shutil.rmtree(s4_meta.outputdir)

        fitness_value = (
            scaling_MAD_spec * s4_meta.mad_s4 +
            scaling_MAD_white *
            (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
        )

        return fitness_value

    else:

        fitness_value = float('inf')

        return fitness_value


# Objective Function
def bg_deg(x, eventlabel, last_s2_meta_outputdir, s3_meta_GA, s4_meta_GA, 
           scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Thresholding
    Differences in Light Curve Analysis

    Description:
    -----------
    This function is structured to optimize the thresholding differences
    across multiple stages (specifically Stages 3, 4, and 5). By adjusting
    the `diffthresh` parameter, based on the optimization variable `x`, the
    function aims to minimize an objective that combines the chi-squared value,
    MAD of spectroscopic light curves, and MAD of binned white light curves.

    The function integrates the effect of `diffthresh` across the stages,
    processes the light curves, and subsequently computes the fitness value.

    Parameters:
    ----------
    x : list
        List of optimization variables. Here, it specifically adjusts the
        difference threshold (`diffthresh`) for the light curve stages.

    eventlabel : str
        A label or identifier for the event being analyzed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. These get
        modified within the function based on the optimization variable `x`.

    params : dict or object
        A collection of parameters, likely used in Stage 5's fitting process.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors that are used to weight the respective components in
        the objective function. They ensure that each component's contribution
        to the fitness value is appropriately adjusted.

    Outputs:
    -------
    fitness_value : float
        The computed objective value, which is a measure of the fitness of the
        given set of optimization variables `x`. Lower values are preferred,
        suggesting a better thresholding configuration.

    Notes:
    -----
    - The function updates the `diffthresh` parameter in the metadata of
      Stages 3, 4, and 5 using the optimization variable `x[0]`.
    - After adjusting the `diffthresh`, the function processes the light
      curves in each stage and computes the required metrics.
    - The objective is a weighted sum of the chi-squared value from Stage 5,
      MAD of spectroscopic light curves from Stage 4, and MAD of binned white
      light curves from Stage 4.
    - The function also performs cleanup by removing directories related to
      each stage after processing.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 3
    s3_meta_GA.bg_deg = x[0]
    # Stage 4
    s4_meta_GA.bg_deg = x[0]

    s3_meta_GA.inputdir = last_s2_meta_outputdir

    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def bg_method(x, eventlabel, last_s2_meta_outputdir, s3_meta_GA, s4_meta_GA, 
              scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Background Method 
    Selection in Light Curve Analysis

    Description:
    -----------
    This function optimizes the background subtraction method (bg_method) 
    across
    multiple stages (specifically Stages 3, 4, and 5). By adjusting the 
    bg_method
    parameter based on the optimization variable x, the function aims to 
    minimize
    an objective that combines the chi-squared value, MAD of spectroscopic 
    light
    curves, and MAD of binned white light curves.

    The function integrates the effect of bg_method across the stages,
    processes the light curves, and subsequently computes the fitness value.

    Parameters:
    ----------
    x : list
        List of optimization variables. Here, it specifically sets the 
        background
        subtraction method (bg_method) for the light curve stages, given as a 
        string.

    eventlabel : str
        A label or identifier for the event being analyzed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. These get
        modified within the function based on the optimization variable x.

    params : dict or object
        A collection of parameters, likely used in Stage 5's fitting process.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors that are used to weight the respective components in
        the objective function. They ensure that each component's contribution
        to the fitness value is appropriately adjusted.

    Outputs:
    -------
    fitness_value : float
        The computed objective value, which is a measure of the fitness of the
        given set of optimization variables x. Lower values are preferred,
        suggesting a better background method configuration.

    Notes:
    -----
    - The function updates the bg_method parameter in the metadata of
      Stages 3, 4, and 5 using the optimization variable x[0].
    - After adjusting bg_method, the function processes the light
      curves in each stage and computes the required metrics.
    - The objective is a weighted sum of the chi-squared value from Stage 5,
      MAD of spectroscopic light curves from Stage 4, and MAD of binned white
      light curves from Stage 4.
    - The function also performs cleanup by removing directories related to
      each stage after processing.

    Author: Reza Ashtari
    Date: 08/22/2023
    """
    def remove_apostrophes(s):
        return s.replace("'", "")

    # Set bg_method in metadata for each stage based on the input string
    bg_method_value = remove_apostrophes(x[0])  # Assuming x[0] is a string
    # bg_method_value = x[0]  # Assuming x[0] is a string, e.g. 'std'
    s3_meta_GA.bg_method = bg_method_value
    s4_meta_GA.bg_method = bg_method_value

    s3_meta_GA.inputdir = last_s2_meta_outputdir

    # Perform the light curve processing for each stage
    s3_spec, s3_meta = s3.reduce(eventlabel, 
                                 input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, 
                                       input_meta=s4_meta_GA, s3_meta=s3_meta)

    # Clean up the directories for each stage
    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    # Calculate the fitness value based on the specified scaling factors
    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white * (sum(s4_meta.mad_s4_binned) / 
                             len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def p3thresh(x, eventlabel, last_s2_meta_outputdir, s3_meta_GA, s4_meta_GA, 
             scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for p3thresh Parameter
    Optimization in Light Curve Analysis

    Description:
    -----------
    This function seeks to optimize the `p3thresh` parameter across Stages
    3, 4, and 5 in the light curve analysis. The `p3thresh` parameter,
    represented by the optimization variable `x[0]`, is modified in the
    metadata for each stage. Subsequently, the light curves are processed,
    and a fitness value is computed based on a weighted sum of the
    chi-squared value, MAD of spectroscopic light curves, and MAD of binned
    white light curves.

    Parameters:
    ----------
    x : list
        List of optimization variables. `x[0]` adjusts the `p3thresh`
        parameter.

    eventlabel : str
        A label or identifier for the event being analyzed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. These get
        modified within the function based on the optimization variable `x[0]`.

    params : dict or object
        A collection of parameters, possibly used in Stage 5's fitting process.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors used to weight the respective components in the
        objective function, ensuring that each component's influence on the
        fitness value is correctly balanced.

    Outputs:
    -------
    fitness_value : float
        The computed objective value, indicating the fitness of the given
        `p3thresh` value in `x[0]`. A lower value is preferred, suggesting
        an optimal setting of the `p3thresh` parameter.

    Notes:
    -----
    - The function sets the `p3thresh` parameter in the metadata of Stages
      3, 4, and 5 using `x[0]`.
    - The objective is a weighted sum of the chi-squared value from Stage 5,
      MAD of spectroscopic light curves from Stage 4, and MAD of binned white
      light curves from Stage 4.
    - After processing, directories associated with each stage are removed
      for cleanup.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 3
    s3_meta_GA.p3thresh = x[0]
    # Stage 4
    s4_meta_GA.p3thresh = x[0]

    s3_meta_GA.inputdir = last_s2_meta_outputdir

    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def median_thresh(x, eventlabel, last_s2_meta_outputdir, s3_meta_GA, 
                  s4_meta_GA, scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Median Threshold
    Parameter Optimization in Light Curve Analysis

    Description:
    -----------
    This function aims to optimize the `median_thresh` parameter across Stages
    3, 4, and 5 in the light curve analysis. The `median_thresh` parameter,
    represented by the optimization variable `x[0]`, is adjusted in the
    metadata or each stage. Following this, the light curves are generated and
    processed, and a fitness value is derived based on a weighted sum of the
    chi-squared value, MAD of spectroscopic light curves, and MAD of binned
    white light curves.

    Parameters:
    ----------
    x : list
        List of optimization variables. `x[0]` represents the value assigned
        to the `median_thresh` parameter.

    eventlabel : str
        A label or identifier for the event being analyzed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. These are
        adjusted within the function based on the optimization variable `x[0]`.

    params : dict or object
        A collection of parameters potentially employed in Stage 5's fitting
        procedure.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling coefficients used to proportionately weight the various
        components in the objective function. This ensures each component's
        contribution to the fitness value is appropriately balanced.

    Outputs:
    -------
    fitness_value : float
        The calculated objective value, reflecting the fitness of the provided
        `median_thresh` value in `x[0]`. A lower score is more desirable,
        suggesting a more optimal setting of the `median_thresh` parameter.

    Notes:
    -----
    - The function assigns the `median_thresh` parameter in the metadata of
      Stages 3, 4, and 5 using `x[0]`.
    - The objective comprises a weighted sum of the chi-squared value from
      Stage 5, MAD of spectroscopic light curves from Stage 4, and MAD of
      binned white light curves from Stage 4.
    - Upon processing, directories corresponding to each stage are deleted for
      cleanup.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 3
    s3_meta_GA.median_thresh = x[0]
    # Stage 4
    s4_meta_GA.median_thresh = x[0]

    s3_meta_GA.inputdir = last_s2_meta_outputdir

    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def window_len(x, eventlabel, last_s2_meta_outputdir, s3_meta_GA, s4_meta_GA,
               scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Window Length Parameter
    Optimization in Light Curve Analysis

    Description:
    -----------
    This function is geared towards the optimization of the `window_len`
    parameter across Stages 3, 4, and 5 in the light curve analysis process.
    The `window_len` parameter, represented by the optimization variable
    `x[0]`, gets assigned in the metadata of each respective stage. Subsequent
    to this configuration, the light curves are generated and processed. An
    objective or "fitness" value is then computed based on a combination of the
    chi-squared value, MAD of spectroscopic light curves, and MAD of binned
    white light curves.

    Parameters:
    ----------
    x : list
        A list of optimization variables where `x[0]` specifies the value to
        be set for the `window_len` parameter.

    eventlabel : str
        A distinct label or identifier for the specific event being processed.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects pertaining to Stages 3, 4, and 5, respectively. Within
        the function, these objects are modified based on the value of `x[0]`.

    params : dict or object
        A collection of parameters, likely utilized in the fitting procedure
        of Stage 5.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling coefficients designated to balance the contributions of
        various components in the objective function. These ensure that the
        impact of each component on the fitness value is harmonized.

    Outputs:
    -------
    fitness_value : float
        A computed objective score, indicative of the efficacy of the proposed
        `window_len` value in `x[0]`. A more minimal score is indicative of a
        better or more optimal configuration for the `window_len` parameter.

    Notes:
    -----
    - The function assigns the `window_len` parameter for the metadata of
      Stages 3, 4, and 5 based on `x[0]`.
    - The objective score embodies a weighted aggregate of the chi-squared
    value from Stage 5, MAD of spectroscopic light curves from Stage 4, and
      MAD of the binned white light curves also from Stage 4.
    - For housekeeping, directories associated with each of the stages are
      deleted post-processing.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 3
    s3_meta_GA.window_len = x[0]
    # Stage 4
    s4_meta_GA.window_len = x[0]

    s3_meta_GA.inputdir = last_s2_meta_outputdir

    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def p5thresh(x, eventlabel, last_s2_meta_outputdir, s3_meta_GA, s4_meta_GA, 
             scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for P5 Threshold Parameter
    Optimization in Light Curve Analysis

    Description:
    -----------
    This function targets the optimization of the `p5thresh` parameter across
    Stages 3, 4, and 5 in the light curve analysis process. The `p5thresh`
    parameter, represented by the optimization variable `x[0]`, is assigned
    to the metadata of each respective stage. After this configuration, the
    light curves are generated and processed. An objective or "fitness" value
    is subsequently calculated based on a combination of the chi-squared
    value, MAD of spectroscopic light curves, and MAD of binned white light
    curves.

    Parameters:
    ----------
    x : list
        A list of optimization variables where `x[0]` defines the value to be
        set for the `p5thresh` parameter.

    eventlabel : str
        A unique label or identifier for the specific event under
        investigation.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. The function
        modifies these objects according to the value of `x[0]`.

    params : dict or object
        A collection of parameters, presumably used in the fitting procedure
        of Stage 5.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors assigned to balance the contributions of different
        components in the objective function. These ensure each component's
        influence on the fitness value is proportionate.

    Outputs:
    -------
    fitness_value : float
        A computed objective score, indicating the effectiveness of the
        proposed `p5thresh` value in `x[0]`. A lower score suggests a more
        optimal configuration for the `p5thresh` parameter.

    Notes:
    -----
    - The function configures the `p5thresh` parameter in the metadata of
      Stages 3, 4, and 5 based on `x[0]`.
    - The objective score is a weighted sum of the chi-squared value from
      Stage 5, MAD of spectroscopic light curves from Stage 4, and MAD of the
      binned white light curves from Stage 4.
    - Post-processing, directories associated with each stage are removed for
      cleanliness.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 3
    s3_meta_GA.p5thresh = x[0]
    # Stage 4
    s4_meta_GA.p5thresh = x[0]

    s3_meta_GA.inputdir = last_s2_meta_outputdir

    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def p7thresh(x, eventlabel, last_s2_meta_outputdir, s3_meta_GA, s4_meta_GA,
             scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for P7 Threshold Parameter
    Optimization in Light Curve Analysis

    Description:
    -----------
    This function targets the optimization of the `p7thresh` parameter across
    Stages 3, 4, and 5 in the light curve analysis process. The `p7thresh`
    parameter, represented by the optimization variable `x[0]`, is assigned to
    the metadata of each respective stage. After this configuration, the light
    curves are generated and processed. An objective or "fitness" value is
    subsequently calculated based on a combination of the chi-squared value,
    MAD of spectroscopic light curves, and MAD of binned white light curves.

    Parameters:
    ----------
    x : list
        A list of optimization variables where `x[0]` defines the value to be
        set for the `p5thresh` parameter.

    eventlabel : str
        A unique label or identifier for the specific event under
        investigation.

    s3_meta_GA, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. The function
        modifies these objects according to the value of `x[0]`.

    params : dict or object
        A collection of parameters, presumably used in the fitting procedure
        of Stage 5.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors assigned to balance the contributions of different
        components in the objective function. These ensure each component's
        influence on the fitness value is proportionate.

    Outputs:
    -------
    fitness_value : float
        A computed objective score, indicating the effectiveness of the
        proposed `p7thresh` value in `x[0]`. A lower score suggests a more
        optimal configuration for the `p7thresh` parameter.

    Notes:
    -----
    - The function configures the `p7thresh` parameter in the metadata of
      Stages 3, 4, and 5 based on `x[0]`.
    - The objective score is a weighted sum of the chi-squared value from
      Stage 5, MAD of spectroscopic light curves from Stage 4, and MAD of
      the binned white light curves from Stage 4.
    - Post-processing, directories associated with each stage are removed
      for cleanliness.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 3
    s3_meta_GA.p7thresh = x[0]
    # Stage 4
    s4_meta_GA.p7thresh = x[0]

    s3_meta_GA.inputdir = last_s2_meta_outputdir

    s3_spec, s3_meta = s3.reduce(eventlabel, input_meta=s3_meta_GA)
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s3_meta.outputdir)
    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# # Objective Function
# def drift_range(x, eventlabel, last_s3_meta_outputdir, s3_meta, s4_meta_GA,
#                 s5_meta_GA, params, scaling_MAD_white, scaling_MAD_spec,
#                 scaling_chi2red):
#     """
#     Eureka! Optimization Tools: Objective Function for Drift Range Parameter
#     Optimization in Light Curve Analysis

#     Description:
#     -----------
#     This function focuses on optimizing the `drift_range` parameter within
#     Stages 4 and 5 of light curve analysis. The `drift_range` parameter,
#     denoted by the optimization variable `x[0]`, is set in the metadata
#     of Stages 4 and 5. After this configuration, the light curves are
#     generated and processed in Stages 4 and 5, with the output of Stage 3
#     acting as an input to Stage 4. The fitness or "objective" score is
#     subsequently determined based on a combination of the chi-squared value,
#     MAD of spectroscopic light curves, and MAD of binned white light curves.

#     Parameters:
#     ----------
#     x : list
#         A list containing optimization variables where `x[0]` represents the
#         value designated for the `drift_range` parameter.

#     eventlabel : str
#         A unique label or identifier for the event being analyzed.

#     last_s3_meta_outputdir : str
#         The directory path of the most recent output from Stage 3, used as
#         an input for Stage 4.

#     s3_meta, s4_meta_GA, s5_meta_GA : objects
#         Metadata objects for Stages 3, 4, and 5, respectively. The function
#         modifies `s4_meta_GA` and `s5_meta_GA` according to the value of
#         `x[0]`.

#     params : dict or object
#         Parameters used in the Stage 5 fitting procedure.

#     scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
#         Scaling factors to balance the contributions of different components
#         in the objective function. These ensure each component's impact on
#         the objective value is proportionate.

#     Outputs:
#     -------
#     fitness_value : float
#         A calculated score indicating the effectiveness of the `drift_range`
#         value in `x[0]`. Lower scores suggest a more optimal setting for the
#         `drift_range` parameter.

#     Notes:
#     -----
#     - The function sets the `drift_range` parameter in the metadata of Stages
#       4 and 5 based on `x[0]`.
#     - Stage 4 takes the most recent output of Stage 3 as an input, specified
#       by `last_s3_meta_outputdir`.
#     - The objective score is a weighted combination of the chi-squared value
#       from Stage 5, MAD of spectroscopic light curves from Stage 4, and MAD
#       of the binned white light curves from Stage 4.
#     - Directories created in Stages 4 and 5 are removed after processing for
#       cleanliness.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """

#     # Define Variables to be optimized
#     # Stage 4
#     s4_meta_GA.drift_range = x[0]
#     # Stage 5
#     s5_meta_GA.drift_range = x[0]

#     s4_meta_GA.inputdir = last_s3_meta_outputdir
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
#                                        s3_meta=s3_meta)
#     s5_meta = s5.fitlc(eventlabel, params, input_meta=s5_meta_GA,
#                        s4_meta=s4_meta)

#     shutil.rmtree(s4_meta.outputdir)
#     shutil.rmtree(s5_meta.outputdir)

#     fitness_value = (
#         scaling_chi2red * s5_meta.chi2red +
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white *
#         (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
#     )

#     return fitness_value


# # Objective Function
# def highpassWidth(x, eventlabel, last_s3_meta_outputdir, s3_meta, s4_meta_GA,
#                   s5_meta_GA, params, scaling_MAD_white, scaling_MAD_spec,
#                   scaling_chi2red):
#     """
#     Eureka! Optimization Tools: Objective Function for Drift Range Parameter
#     Optimization in Light Curve Analysis

#     Description:
#     -----------
#     This function focuses on optimizing the `highpassWidth` parameter within
#     Stages 4 and 5 of light curve analysis. The `highpassWidth` parameter,
#     denoted by the optimization variable `x[0]`, is set in the metadata
#     of Stages 4 and 5. After this configuration, the light curves are
#     generated and processed in Stages 4 and 5, with the output of Stage 3
#     acting as an input to Stage 4. The fitness or "objective" score is
#     subsequently determined based on a combination of the chi-squared value,
#     MAD of spectroscopic light curves, and MAD of binned white light curves.

#     Parameters:
#     ----------
#     x : list
#         A list containing optimization variables where `x[0]` represents the
#         value designated for the `highpassWidth` parameter.

#     eventlabel : str
#         A unique label or identifier for the event being analyzed.

#     last_s3_meta_outputdir : str
#         The directory path of the most recent output from Stage 3, used as
#         an input for Stage 4.

#     s3_meta, s4_meta_GA, s5_meta_GA : objects
#         Metadata objects for Stages 3, 4, and 5, respectively. The function
#         modifies `s4_meta_GA` and `s5_meta_GA` according to the value of
#         `x[0]`.

#     params : dict or object
#         Parameters used in the Stage 5 fitting procedure.

#     scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
#         Scaling factors to balance the contributions of different components
#         in the objective function. These ensure each component's impact on
#         the objective value is proportionate.

#     Outputs:
#     -------
#     fitness_value : float
#         A calculated score indicating the effectiveness of `highpassWidth`
#         value in `x[0]`. Lower scores suggest a more optimal setting for 
#         `highpassWidth` parameter.

#     Notes:
#     -----
#     - The function sets the `highpassWidth` parameter in the metadata of 
#       Stages 4 and 5 based on `x[0]`.
#     - Stage 4 takes the most recent output of Stage 3 as an input, specified
#       by `last_s3_meta_outputdir`.
#     - The objective score is a weighted combination of the chi-squared value
#       from Stage 5, MAD of spectroscopic light curves from Stage 4, and MAD
#       of the binned white light curves from Stage 4.
#     - Directories created in Stages 4 and 5 are removed after processing for
#       cleanliness.

#     Author: Reza Ashtari
#     Date: 08/22/2023
#     """

#     # Define Variables to be optimized
#     # Stage 4
#     s4_meta_GA.highpassWidth = x[0]
#     # Stage 5
#     s5_meta_GA.highpassWidth = x[0]

#     s4_meta_GA.inputdir = last_s3_meta_outputdir
#     s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
#                                        s3_meta=s3_meta)
#     s5_meta = s5.fitlc(eventlabel, params, input_meta=s5_meta_GA,
#                        s4_meta=s4_meta)

#     shutil.rmtree(s4_meta.outputdir)
#     shutil.rmtree(s5_meta.outputdir)

#     fitness_value = (
#         scaling_chi2red * s5_meta.chi2red +
#         scaling_MAD_spec * s4_meta.mad_s4 +
#         scaling_MAD_white *
#         (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
#     )

#     return fitness_value


# Objective Function
def sigma(x, eventlabel, last_s3_meta_outputdir, s3_meta, s4_meta_GA,
          scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Drift Range Parameter
    Optimization in Light Curve Analysis

    Description:
    -----------
    This function focuses on optimizing the `sigma` parameter within
    Stages 4 and 5 of light curve analysis. The `sigma` parameter,
    denoted by the optimization variable `x[0]`, is set in the metadata
    of Stages 4 and 5. After this configuration, the light curves are
    generated and processed in Stages 4 and 5, with the output of Stage 3
    acting as an input to Stage 4. The fitness or "objective" score is
    subsequently determined based on a combination of the chi-squared value,
    MAD of spectroscopic light curves, and MAD of binned white light curves.

    Parameters:
    ----------
    x : list
        A list containing optimization variables where `x[0]` represents the
        value designated for the `sigma` parameter.

    eventlabel : str
        A unique label or identifier for the event being analyzed.

    last_s3_meta_outputdir : str
        The directory path of the most recent output from Stage 3, used as
        an input for Stage 4.

    s3_meta, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. The function
        modifies `s4_meta_GA` and `s5_meta_GA` according to the value of
        `x[0]`.

    params : dict or object
        Parameters used in the Stage 5 fitting procedure.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors to balance the contributions of different components
        in the objective function. These ensure each component's impact on
        the objective value is proportionate.

    Outputs:
    -------
    fitness_value : float
        A calculated score indicating the effectiveness of the `sigma`
        value in `x[0]`. Lower scores suggest a more optimal setting for the
        `sigma` parameter.

    Notes:
    -----
    - The function sets the `sigma` parameter in the metadata of Stages
      4 and 5 based on `x[0]`.
    - Stage 4 takes the most recent output of Stage 3 as an input, specified
      by `last_s3_meta_outputdir`.
    - The objective score is a weighted combination of the chi-squared value
      from Stage 5, MAD of spectroscopic light curves from Stage 4, and MAD
      of the binned white light curves from Stage 4.
    - Directories created in Stages 4 and 5 are removed after processing for
      cleanliness.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 4
    s4_meta_GA.sigma = x[0]

    s4_meta_GA.inputdir = last_s3_meta_outputdir
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value


# Objective Function
def box_width(x, eventlabel, last_s3_meta_outputdir, s3_meta, s4_meta_GA, 
              scaling_MAD_white, scaling_MAD_spec):
    """
    Eureka! Optimization Tools: Objective Function for Drift Range Parameter
    Optimization in Light Curve Analysis

    Description:
    -----------
    This function focuses on optimizing the `box_width` parameter within
    Stages 4 and 5 of light curve analysis. The `box_width` parameter,
    denoted by the optimization variable `x[0]`, is set in the metadata
    of Stages 4 and 5. After this configuration, the light curves are
    generated and processed in Stages 4 and 5, with the output of Stage 3
    acting as an input to Stage 4. The fitness or "objective" score is
    subsequently determined based on a combination of the chi-squared value,
    MAD of spectroscopic light curves, and MAD of binned white light curves.

    Parameters:
    ----------
    x : list
        A list containing optimization variables where `x[0]` represents the
        value designated for the `box_width` parameter.

    eventlabel : str
        A unique label or identifier for the event being analyzed.

    last_s3_meta_outputdir : str
        The directory path of the most recent output from Stage 3, used as
        an input for Stage 4.

    s3_meta, s4_meta_GA, s5_meta_GA : objects
        Metadata objects for Stages 3, 4, and 5, respectively. The function
        modifies `s4_meta_GA` and `s5_meta_GA` according to the value of
        `x[0]`.

    params : dict or object
        Parameters used in the Stage 5 fitting procedure.

    scaling_MAD_white, scaling_MAD_spec, scaling_chi2red : float
        Scaling factors to balance the contributions of different components
        in the objective function. These ensure each component's impact on
        the objective value is proportionate.

    Outputs:
    -------
    fitness_value : float
        A calculated score indicating the effectiveness of the `box_width`
        value in `x[0]`. Lower scores suggest a more optimal setting for the
        `box_width` parameter.

    Notes:
    -----
    - The function sets the `box_width` parameter in the metadata of Stages
      4 and 5 based on `x[0]`.
    - Stage 4 takes the most recent output of Stage 3 as an input, specified
      by `last_s3_meta_outputdir`.
    - The objective score is a weighted combination of the chi-squared value
      from Stage 5, MAD of spectroscopic light curves from Stage 4, and MAD
      of the binned white light curves from Stage 4.
    - Directories created in Stages 4 and 5 are removed after processing for
      cleanliness.

    Author: Reza Ashtari
    Date: 08/22/2023
    """

    # Define Variables to be optimized
    # Stage 4
    s4_meta_GA.box_width = x[0]

    s4_meta_GA.inputdir = last_s3_meta_outputdir
    s4_spec, s4_lc, s4_meta = s4.genlc(eventlabel, input_meta=s4_meta_GA,
                                       s3_meta=s3_meta)

    shutil.rmtree(s4_meta.outputdir)

    fitness_value = (
        scaling_MAD_spec * s4_meta.mad_s4 +
        scaling_MAD_white *
        (sum(s4_meta.mad_s4_binned) / len(s4_meta.mad_s4_binned))
    )

    return fitness_value
