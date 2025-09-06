# from eureka.optimizer import optimizers
# from eureka.optimizer import objective_funcs
# import warnings

# warnings.filterwarnings("ignore")


# def dqmask(
#     s3_meta_GA,
#     s4_meta_GA,
#     s5_meta_GA,
#     best,
#     manual_clip_lists,
#     bounds_dqmask,
#     eventlabel,
#     params,
#     scaling_MAD_white,
#     scaling_MAD_spec,
#     scaling_chi2red,
# ):
#     """
#     Perform a parametric sweep of "dqmask".

#     Parameters:
#         s3_meta_GA (S3MetaClass): Metadata object for S3.
#         s4_meta_GA (S4MetaClass): Metadata object for S4.
#         s5_meta_GA (S5MetaClass): Metadata object for S5.
#         best (dict): Dictionary containing initial best values for parameters.
#         manual_clip_lists (list): List for manual clipping setup.
#         bounds_dqmask (tuple): Bounds for dqmask parameter.
#         eventlabel (str): Label for the event.
#         params (dict): Parameters for the optimizer.
#         scaling_MAD_white (float): Scaling factor for MAD white.
#         scaling_MAD_spec (float): Scaling factor for MAD spec.
#         scaling_chi2red (float): Scaling factor for reduced chi-squared.

#     Returns:
#         tuple: best_param (bool), updated history_fitness_score (dict),
#         s3_meta_GA (S3MetaClass), s4_meta_GA (S4MetaClass),
#         s5_meta_GA (S5MetaClass).
#     """
#     # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
#     s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
#     s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

#     # Set Up Manual Clipping
#     s5_meta_GA.manual_clip = manual_clip_lists

#     # Set Fit Method
#     s5_meta_GA.fit_method = "lsq"

#     # Perform parametric sweep
#     best_param, best_fitness_value = optimizers.parametric_sweep_dqmask(
#         objective_funcs.dqmask,
#         bounds_dqmask,
#         eventlabel,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#         params,
#         scaling_MAD_white,
#         scaling_MAD_spec,
#         scaling_chi2red,
#     )

#     # Update the 'best' dictionary
#     best["dqmask"] = bool(best_param)

#     # Print results
#     print("Best parameters: ", best_param)
#     print("Best fitness: ", best_fitness_value)

#     # Update history fitness score
#     history_fitness_score = {"dqmask": best_fitness_value}

#     return (
#         best_param,
#         history_fitness_score,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#     )


# def bg_thresh(
#     s3_meta_GA,
#     s4_meta_GA,
#     s5_meta_GA,
#     best,
#     manual_clip_lists,
#     bounds_bg_thresh,
#     eventlabel,
#     params,
#     scaling_MAD_white,
#     scaling_MAD_spec,
#     scaling_chi2red,
# ):
#     """
#     Perform a parametric sweep of "bg_thresh".

#     Parameters:
#         s3_meta_GA (S3MetaClass): Metadata object for S3.
#         s4_meta_GA (S4MetaClass): Metadata object for S4.
#         s5_meta_GA (S5MetaClass): Metadata object for S5.
#         best (dict): Dictionary containing initial best values for parameters.
#         manual_clip_lists (list): List for manual clipping setup.
#         bounds_bg_thresh (tuple): Bounds for bg_thresh parameter.
#         eventlabel (str): Label for the event.
#         params (dict): Parameters for the optimizer.
#         scaling_MAD_white (float): Scaling factor for MAD white.
#         scaling_MAD_spec (float): Scaling factor for MAD spec.
#         scaling_chi2red (float): Scaling factor for reduced chi-squared.

#     Returns:
#         tuple: best_param (list), updated history_fitness_score (dict),
#         s3_meta_GA (S3MetaClass), s4_meta_GA (S4MetaClass),
#         s5_meta_GA (S5MetaClass).
#     """
#     # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
#     s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
#     s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

#     # Set Up Manual Clipping
#     s5_meta_GA.manual_clip = manual_clip_lists

#     # Set Fit Method
#     s5_meta_GA.fit_method = "lsq"

#     # Setup Meta / Define Initial Population
#     # Stage 3
#     s3_meta_GA.dqmask = best["dqmask"]

#     # Stage 4
#     s4_meta_GA.dqmask = best["dqmask"]

#     # Stage 5
#     s5_meta_GA.dqmask = best["dqmask"]

#     # Perform parametric sweep
#     best_param, best_fitness_value = optimizers.parametric_sweep_S3(
#         objective_funcs.bg_thresh,
#         bounds_bg_thresh,
#         eventlabel,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#         params,
#         scaling_MAD_white,
#         scaling_MAD_spec,
#         scaling_chi2red,
#     )

#     # Update the 'best' dictionary
#     best["bg_thresh"] = [best_param, best_param]

#     # Print results
#     print("Best parameters: ", [best_param, best_param])
#     print("Best fitness: ", best_fitness_value)

#     # Update history fitness score
#     history_fitness_score = {"bg_thresh": best_fitness_value}

#     return (
#         best_param,
#         history_fitness_score,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#     )


# def bg_hw_spec_hw(
#     s3_meta_GA,
#     s4_meta_GA,
#     s5_meta_GA,
#     best,
#     manual_clip_lists,
#     bounds_bg_hw,
#     bounds_spec_hw,
#     eventlabel,
#     params,
#     scaling_MAD_white,
#     scaling_MAD_spec,
#     scaling_chi2red,
#     spec_hw_selection,
#     bg_hw_selection,
# ):
#     """
#     Perform a parametric sweep of "bg_hw" and "spec_hw".

#     Parameters:
#         s3_meta_GA (S3MetaClass): Metadata object for S3.
#         s4_meta_GA (S4MetaClass): Metadata object for S4.
#         s5_meta_GA (S5MetaClass): Metadata object for S5.
#         best (dict): Dictionary containing initial best values for parameters.
#         manual_clip_lists (list): List for manual clipping setup.
#         bounds_bg_hw (tuple): Bounds for bg_hw parameter.
#         bounds_spec_hw (tuple): Bounds for spec_hw parameter.
#         eventlabel (str): Label for the event.
#         params (dict): Parameters for the optimizer.
#         scaling_MAD_white (float): Scaling factor for MAD white.
#         scaling_MAD_spec (float): Scaling factor for MAD spec.
#         scaling_chi2red (float): Scaling factor for reduced chi-squared.
#         spec_hw_selection (str): Selection method for spec_hw
#         ('auto' or 'manual').
#         bg_hw_selection (str): Selection method for bg_hw ('auto' or 'manual').

#     Returns:
#         tuple: best_params (list), updated history_fitness_score (dict),
#         s3_meta_GA (S3MetaClass), s4_meta_GA (S4MetaClass),
#         s5_meta_GA (S5MetaClass).
#     """
#     # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
#     s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
#     s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

#     # Set spec_hw & bg_hw values for auto vs. manual selection methods
#     if spec_hw_selection == "auto":
#         s3_meta_GA.spec_hw = 3  # MIRI - LRS
#     if bg_hw_selection == "auto":
#         s3_meta_GA.bg_hw = s3_meta_GA.spec_hw

#     # Set Up Manual Clipping
#     s5_meta_GA.manual_clip = manual_clip_lists

#     # Set Fit Method
#     s5_meta_GA.fit_method = "lsq"

#     # Setup Meta / Define Initial Population
#     # Stage 3
#     s3_meta_GA.dqmask = best["dqmask"]
#     s3_meta_GA.bg_thresh = best["bg_thresh"]

#     # Stage 4
#     s4_meta_GA.dqmask = best["dqmask"]
#     s4_meta_GA.bg_thresh = best["bg_thresh"]

#     # Stage 5
#     s5_meta_GA.dqmask = best["dqmask"]
#     s5_meta_GA.bg_thresh = best["bg_thresh"]

#     # Perform parametric sweep
#     best_params, best_fitness_value = optimizers.parametric_sweep_double(
#         objective_funcs.bg_hw_spec_hw,
#         bounds_bg_hw,
#         bounds_spec_hw,
#         eventlabel,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#         params,
#         scaling_MAD_white,
#         scaling_MAD_spec,
#         scaling_chi2red,
#     )

#     # Save Results in "best" Dictionary
#     best["bg_hw"] = best_params[0]
#     best["spec_hw"] = best_params[1]

#     # Print Results of Parametric Sweep
#     print("Best parameters: ", best_params)
#     print("Best fitness: ", best_fitness_value)

#     # Update history fitness score
#     history_fitness_score = {"spec_hw_bg_hw": best_fitness_value}

#     return (
#         best_params,
#         history_fitness_score,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#     )


# def bg_deg(
#     s3_meta_GA,
#     s4_meta_GA,
#     s5_meta_GA,
#     best,
#     manual_clip_lists,
#     bounds_bg_deg,
#     eventlabel,
#     params,
#     scaling_MAD_white,
#     scaling_MAD_spec,
#     scaling_chi2red,
# ):
#     """
#     Perform a parametric sweep of "bg_deg".

#     Parameters:
#         s3_meta_GA (S3MetaClass): Metadata object for S3.
#         s4_meta_GA (S4MetaClass): Metadata object for S4.
#         s5_meta_GA (S5MetaClass): Metadata object for S5.
#         best (dict): Dictionary containing initial best values for parameters.
#         manual_clip_lists (list): List for manual clipping setup.
#         bounds_bg_deg (tuple): Bounds for bg_deg parameter.
#         eventlabel (str): Label for the event.
#         params (dict): Parameters for the optimizer.
#         scaling_MAD_white (float): Scaling factor for MAD white.
#         scaling_MAD_spec (float): Scaling factor for MAD spec.
#         scaling_chi2red (float): Scaling factor for reduced chi-squared.

#     Returns:
#         tuple: best_param (list), updated history_fitness_score (dict),
#         s3_meta_GA (S3MetaClass), s4_meta_GA (S4MetaClass),
#         s5_meta_GA (S5MetaClass).
#     """
#     # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
#     s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
#     s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

#     # Set Up Manual Clipping
#     s5_meta_GA.manual_clip = manual_clip_lists

#     # Set Fit Method
#     s5_meta_GA.fit_method = "lsq"

#     # Setup Meta / Define Initial Population
#     # Stage 3
#     s3_meta_GA.dqmask = best["dqmask"]
#     s3_meta_GA.bg_thresh = best["bg_thresh"]
#     s3_meta_GA.bg_hw = best["bg_hw"]
#     s3_meta_GA.spec_hw = best["spec_hw"]

#     # Stage 4
#     s4_meta_GA.dqmask = best["dqmask"]
#     s4_meta_GA.bg_thresh = best["bg_thresh"]
#     s4_meta_GA.bg_hw = best["bg_hw"]
#     s4_meta_GA.spec_hw = best["spec_hw"]

#     # Stage 5
#     s5_meta_GA.dqmask = best["dqmask"]
#     s5_meta_GA.bg_thresh = best["bg_thresh"]
#     s5_meta_GA.bg_hw = best["bg_hw"]
#     s5_meta_GA.spec_hw = best["spec_hw"]

#     # Perform parametric sweep
#     best_param, best_fitness_value = optimizers.parametric_sweep_S3(
#         objective_funcs.bg_deg,
#         bounds_bg_deg,
#         eventlabel,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#         params,
#         scaling_MAD_white,
#         scaling_MAD_spec,
#         scaling_chi2red,
#     )

#     # Update the 'best' dictionary
#     best["bg_deg"] = best_param[0]

#     # Print results
#     print("Best parameters: ", best_param[0])
#     print("Best fitness: ", best_fitness_value)

#     # Update history fitness score
#     history_fitness_score = {"bg_deg": best_fitness_value}

#     return (
#         best_param,
#         history_fitness_score,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#     )


# def bg_method(
#     s3_meta_GA,
#     s4_meta_GA,
#     s5_meta_GA,
#     best,
#     manual_clip_lists,
#     bounds_bg_method,
#     eventlabel,
#     params,
#     scaling_MAD_white,
#     scaling_MAD_spec,
#     scaling_chi2red,
# ):
#     """
#     Perform a parametric sweep of "bg_method".

#     Parameters:
#         s3_meta_GA (S3MetaClass): Metadata object for S3.
#         s4_meta_GA (S4MetaClass): Metadata object for S4.
#         s5_meta_GA (S5MetaClass): Metadata object for S5.
#         best (dict): Dictionary containing initial best values for parameters.
#         manual_clip_lists (list): List for manual clipping setup.
#         bounds_bg_method (tuple): Bounds for bg_method parameter.
#         eventlabel (str): Label for the event.
#         params (dict): Parameters for the optimizer.
#         scaling_MAD_white (float): Scaling factor for MAD white.
#         scaling_MAD_spec (float): Scaling factor for MAD spec.
#         scaling_chi2red (float): Scaling factor for reduced chi-squared.

#     Returns:
#         tuple: best_param (any), updated history_fitness_score (dict),
#         s3_meta_GA (S3MetaClass), s4_meta_GA (S4MetaClass),
#         s5_meta_GA (S5MetaClass).
#     """
#     # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
#     s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
#     s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

#     # Set Up Manual Clipping
#     s5_meta_GA.manual_clip = manual_clip_lists

#     # Set Fit Method
#     s5_meta_GA.fit_method = "lsq"

#     # Setup Meta / Define Initial Population
#     # Stage 3
#     s3_meta_GA.dqmask = best["dqmask"]
#     s3_meta_GA.bg_thresh = best["bg_thresh"]
#     s3_meta_GA.bg_hw = best["bg_hw"]
#     s3_meta_GA.spec_hw = best["spec_hw"]
#     s3_meta_GA.bg_deg = best["bg_deg"]

#     # Stage 4
#     s4_meta_GA.dqmask = best["dqmask"]
#     s4_meta_GA.bg_thresh = best["bg_thresh"]
#     s4_meta_GA.bg_hw = best["bg_hw"]
#     s4_meta_GA.spec_hw = best["spec_hw"]
#     s4_meta_GA.bg_deg = best["bg_deg"]

#     # Stage 5
#     s5_meta_GA.dqmask = best["dqmask"]
#     s5_meta_GA.bg_thresh = best["bg_thresh"]
#     s5_meta_GA.bg_hw = best["bg_hw"]
#     s5_meta_GA.spec_hw = best["spec_hw"]
#     s5_meta_GA.bg_deg = best["bg_deg"]

#     # Perform parametric sweep
#     best_param, best_fitness_value = optimizers.parametric_sweep_bg_method(
#         objective_funcs.bg_method,
#         bounds_bg_method,
#         eventlabel,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#         params,
#         scaling_MAD_white,
#         scaling_MAD_spec,
#         scaling_chi2red,
#     )

#     # Update the 'best' dictionary
#     best["bg_method"] = best_param

#     # Print results
#     print("Best parameters: ", best_param)
#     print("Best fitness: ", best_fitness_value)

#     # Update history fitness score
#     history_fitness_score = {"bg_method": best_fitness_value}

#     return (
#         best_param,
#         history_fitness_score,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#     )


# def p3thresh(
#     s3_meta_GA,
#     s4_meta_GA,
#     s5_meta_GA,
#     best,
#     manual_clip_lists,
#     bounds_p3thresh,
#     eventlabel,
#     params,
#     scaling_MAD_white,
#     scaling_MAD_spec,
#     scaling_chi2red,
# ):
#     """
#     Perform a parametric sweep of "p3thresh".

#     Parameters:
#         s3_meta_GA (S3MetaClass): Metadata object for S3.
#         s4_meta_GA (S4MetaClass): Metadata object for S4.
#         s5_meta_GA (S5MetaClass): Metadata object for S5.
#         best (dict): Dictionary containing initial best values for parameters.
#         manual_clip_lists (list): List for manual clipping setup.
#         bounds_p3thresh (tuple): Bounds for p3thresh parameter.
#         eventlabel (str): Label for the event.
#         params (dict): Parameters for the optimizer.
#         scaling_MAD_white (float): Scaling factor for MAD white.
#         scaling_MAD_spec (float): Scaling factor for MAD spec.
#         scaling_chi2red (float): Scaling factor for reduced chi-squared.

#     Returns:
#         tuple: best_param (list), updated history_fitness_score (dict),
#         s3_meta_GA (S3MetaClass), s4_meta_GA (S4MetaClass),
#         s5_meta_GA (S5MetaClass).
#     """
#     # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
#     s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
#     s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

#     # Set Up Manual Clipping
#     s5_meta_GA.manual_clip = manual_clip_lists

#     # Set Fit Method
#     s5_meta_GA.fit_method = "lsq"

#     # Setup Meta / Define Initial Population
#     # Stage 3
#     s3_meta_GA.dqmask = best["dqmask"]
#     s3_meta_GA.bg_thresh = best["bg_thresh"]
#     s3_meta_GA.bg_hw = best["bg_hw"]
#     s3_meta_GA.spec_hw = best["spec_hw"]
#     s3_meta_GA.bg_deg = best["bg_deg"]
#     s3_meta_GA.bg_method = best["bg_method"]

#     # Stage 4
#     s4_meta_GA.dqmask = best["dqmask"]
#     s4_meta_GA.bg_thresh = best["bg_thresh"]
#     s4_meta_GA.bg_hw = best["bg_hw"]
#     s4_meta_GA.spec_hw = best["spec_hw"]
#     s4_meta_GA.bg_deg = best["bg_deg"]
#     s4_meta_GA.bg_method = best["bg_method"]

#     # Stage 5
#     s5_meta_GA.dqmask = best["dqmask"]
#     s5_meta_GA.bg_thresh = best["bg_thresh"]
#     s5_meta_GA.bg_hw = best["bg_hw"]
#     s5_meta_GA.spec_hw = best["spec_hw"]
#     s5_meta_GA.bg_deg = best["bg_deg"]
#     s5_meta_GA.bg_method = best["bg_method"]

#     # Perform parametric sweep
#     best_param, best_fitness_value = optimizers.parametric_sweep_S3(
#         objective_funcs.p3thresh,
#         bounds_p3thresh,
#         eventlabel,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#         params,
#         scaling_MAD_white,
#         scaling_MAD_spec,
#         scaling_chi2red,
#     )

#     # Update the 'best' dictionary
#     best["p3thresh"] = best_param[0]

#     # Print results
#     print("Best parameters: ", best_param[0])
#     print("Best fitness: ", best_fitness_value)

#     # Update history fitness score
#     history_fitness_score = {"p3thresh": best_fitness_value}

#     return (
#         best_param,
#         history_fitness_score,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#     )


# def median_thresh(
#     s3_meta_GA,
#     s4_meta_GA,
#     s5_meta_GA,
#     best,
#     manual_clip_lists,
#     bounds_median_thresh,
#     eventlabel,
#     params,
#     scaling_MAD_white,
#     scaling_MAD_spec,
#     scaling_chi2red,
# ):
#     """
#     Perform a parametric sweep of "median_thresh".

#     Parameters:
#         s3_meta_GA (S3MetaClass): Metadata object for S3.
#         s4_meta_GA (S4MetaClass): Metadata object for S4.
#         s5_meta_GA (S5MetaClass): Metadata object for S5.
#         best (dict): Dictionary containing initial best values for parameters.
#         manual_clip_lists (list): List for manual clipping setup.
#         bounds_median_thresh (tuple): Bounds for median_thresh parameter.
#         eventlabel (str): Label for the event.
#         params (dict): Parameters for the optimizer.
#         scaling_MAD_white (float): Scaling factor for MAD white.
#         scaling_MAD_spec (float): Scaling factor for MAD spec.
#         scaling_chi2red (float): Scaling factor for reduced chi-squared.

#     Returns:
#         tuple: best_param (list), updated history_fitness_score (dict),
#         s3_meta_GA (S3MetaClass), s4_meta_GA (S4MetaClass),
#         s5_meta_GA (S5MetaClass).
#     """
#     # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
#     s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
#     s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

#     # Set Up Manual Clipping
#     s5_meta_GA.manual_clip = manual_clip_lists

#     # Set Fit Method
#     s5_meta_GA.fit_method = "lsq"

#     # Setup Meta / Define Initial Population
#     # Stage 3
#     s3_meta_GA.dqmask = best["dqmask"]
#     s3_meta_GA.bg_thresh = best["bg_thresh"]
#     s3_meta_GA.bg_hw = best["bg_hw"]
#     s3_meta_GA.spec_hw = best["spec_hw"]
#     s3_meta_GA.bg_deg = best["bg_deg"]
#     s3_meta_GA.bg_method = best["bg_method"]
#     s3_meta_GA.p3thresh = best["p3thresh"]

#     # Stage 4
#     s4_meta_GA.dqmask = best["dqmask"]
#     s4_meta_GA.bg_thresh = best["bg_thresh"]
#     s4_meta_GA.bg_hw = best["bg_hw"]
#     s4_meta_GA.spec_hw = best["spec_hw"]
#     s4_meta_GA.bg_deg = best["bg_deg"]
#     s4_meta_GA.bg_method = best["bg_method"]
#     s4_meta_GA.p3thresh = best["p3thresh"]

#     # Stage 5
#     s5_meta_GA.dqmask = best["dqmask"]
#     s5_meta_GA.bg_thresh = best["bg_thresh"]
#     s5_meta_GA.bg_hw = best["bg_hw"]
#     s5_meta_GA.spec_hw = best["spec_hw"]
#     s5_meta_GA.bg_deg = best["bg_deg"]
#     s5_meta_GA.bg_method = best["bg_method"]
#     s5_meta_GA.p3thresh = best["p3thresh"]

#     # Perform parametric sweep
#     best_param, best_fitness_value = optimizers.parametric_sweep_S3(
#         objective_funcs.median_thresh,
#         bounds_median_thresh,
#         eventlabel,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#         params,
#         scaling_MAD_white,
#         scaling_MAD_spec,
#         scaling_chi2red,
#     )

#     # Update the 'best' dictionary
#     best["median_thresh"] = best_param[0]

#     # Print results
#     print("Best parameters: ", best_param[0])
#     print("Best fitness: ", best_fitness_value)

#     # Update history fitness score
#     history_fitness_score = {"median_thresh": best_fitness_value}

#     return (
#         best_param,
#         history_fitness_score,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#     )


# def window_len(
#     s3_meta_GA,
#     s4_meta_GA,
#     s5_meta_GA,
#     best,
#     manual_clip_lists,
#     bounds_window_len,
#     eventlabel,
#     params,
#     scaling_MAD_white,
#     scaling_MAD_spec,
#     scaling_chi2red,
# ):
#     """
#     Perform a parametric sweep of "window_len".

#     Parameters:
#         s3_meta_GA (S3MetaClass): Metadata object for S3.
#         s4_meta_GA (S4MetaClass): Metadata object for S4.
#         s5_meta_GA (S5MetaClass): Metadata object for S5.
#         best (dict): Dictionary containing initial best values for parameters.
#         manual_clip_lists (list): List for manual clipping setup.
#         bounds_window_len (tuple): Bounds for window_len parameter.
#         eventlabel (str): Label for the event.
#         params (dict): Parameters for the optimizer.
#         scaling_MAD_white (float): Scaling factor for MAD white.
#         scaling_MAD_spec (float): Scaling factor for MAD spec.
#         scaling_chi2red (float): Scaling factor for reduced chi-squared.

#     Returns:
#         tuple: best_param (list), updated history_fitness_score (dict),
#         s3_meta_GA (S3MetaClass), s4_meta_GA (S4MetaClass),
#         s5_meta_GA (S5MetaClass).
#     """
#     # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
#     s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
#     s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

#     # Set Up Manual Clipping
#     s5_meta_GA.manual_clip = manual_clip_lists

#     # Set Fit Method
#     s5_meta_GA.fit_method = "lsq"

#     # Setup Meta / Define Initial Population
#     # Stage 3
#     s3_meta_GA.dqmask = best["dqmask"]
#     s3_meta_GA.bg_thresh = best["bg_thresh"]
#     s3_meta_GA.bg_hw = best["bg_hw"]
#     s3_meta_GA.spec_hw = best["spec_hw"]
#     s3_meta_GA.bg_deg = best["bg_deg"]
#     s3_meta_GA.bg_method = best["bg_method"]
#     s3_meta_GA.p3thresh = best["p3thresh"]
#     s3_meta_GA.median_thresh = best["median_thresh"]

#     # Stage 4
#     s4_meta_GA.dqmask = best["dqmask"]
#     s4_meta_GA.bg_thresh = best["bg_thresh"]
#     s4_meta_GA.bg_hw = best["bg_hw"]
#     s4_meta_GA.spec_hw = best["spec_hw"]
#     s4_meta_GA.bg_deg = best["bg_deg"]
#     s4_meta_GA.bg_method = best["bg_method"]
#     s4_meta_GA.p3thresh = best["p3thresh"]
#     s4_meta_GA.median_thresh = best["median_thresh"]

#     # Stage 5
#     s5_meta_GA.dqmask = best["dqmask"]
#     s5_meta_GA.bg_thresh = best["bg_thresh"]
#     s5_meta_GA.bg_hw = best["bg_hw"]
#     s5_meta_GA.spec_hw = best["spec_hw"]
#     s5_meta_GA.bg_deg = best["bg_deg"]
#     s5_meta_GA.bg_method = best["bg_method"]
#     s5_meta_GA.p3thresh = best["p3thresh"]
#     s5_meta_GA.median_thresh = best["median_thresh"]

#     # Perform parametric sweep for odd numbers
#     best_param, best_fitness_value = optimizers.parametric_sweep_odd(
#         objective_funcs.window_len,
#         bounds_window_len,
#         eventlabel,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#         params,
#         scaling_MAD_white,
#         scaling_MAD_spec,
#         scaling_chi2red,
#     )

#     # Update the 'best' dictionary
#     best["window_len"] = best_param[0]

#     # Print results
#     print("Best parameters: ", best_param[0])
#     print("Best fitness: ", best_fitness_value)

#     # Update history fitness score
#     history_fitness_score = {"window_len": best_fitness_value}

#     return (
#         best_param,
#         history_fitness_score,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#     )


# def p7thresh(
#     s3_meta_GA,
#     s4_meta_GA,
#     s5_meta_GA,
#     best,
#     manual_clip_lists,
#     bounds_p7thresh,
#     eventlabel,
#     params,
#     scaling_MAD_white,
#     scaling_MAD_spec,
#     scaling_chi2red,
# ):
#     """
#     Perform a parametric sweep of "p7thresh".

#     Parameters:
#         s3_meta_GA (S3MetaClass): Metadata object for S3.
#         s4_meta_GA (S4MetaClass): Metadata object for S4.
#         s5_meta_GA (S5MetaClass): Metadata object for S5.
#         best (dict): Dictionary containing initial best values for parameters.
#         manual_clip_lists (list): List for manual clipping setup.
#         bounds_p7thresh (tuple): Bounds for p7thresh parameter.
#         eventlabel (str): Label for the event.
#         params (dict): Parameters for the optimizer.
#         scaling_MAD_white (float): Scaling factor for MAD white.
#         scaling_MAD_spec (float): Scaling factor for MAD spec.
#         scaling_chi2red (float): Scaling factor for reduced chi-squared.

#     Returns:
#         tuple: best_param (list), updated history_fitness_score (dict),
#         s3_meta_GA (S3MetaClass), s4_meta_GA (S4MetaClass),
#         s5_meta_GA (S5MetaClass).
#     """
#     # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
#     s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
#     s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

#     # Set Up Manual Clipping
#     s5_meta_GA.manual_clip = manual_clip_lists

#     # Set Fit Method
#     s5_meta_GA.fit_method = "lsq"

#     # Setup Meta / Define Initial Population
#     # Stage 3
#     s3_meta_GA.dqmask = best["dqmask"]
#     s3_meta_GA.bg_thresh = best["bg_thresh"]
#     s3_meta_GA.bg_hw = best["bg_hw"]
#     s3_meta_GA.spec_hw = best["spec_hw"]
#     s3_meta_GA.bg_deg = best["bg_deg"]
#     s3_meta_GA.bg_method = best["bg_method"]
#     s3_meta_GA.p3thresh = best["p3thresh"]
#     s3_meta_GA.median_thresh = best["median_thresh"]
#     s3_meta_GA.window_len = best["window_len"]

#     # Stage 4
#     s4_meta_GA.dqmask = best["dqmask"]
#     s4_meta_GA.bg_thresh = best["bg_thresh"]
#     s4_meta_GA.bg_hw = best["bg_hw"]
#     s4_meta_GA.spec_hw = best["spec_hw"]
#     s4_meta_GA.bg_deg = best["bg_deg"]
#     s4_meta_GA.bg_method = best["bg_method"]
#     s4_meta_GA.p3thresh = best["p3thresh"]
#     s4_meta_GA.median_thresh = best["median_thresh"]
#     s4_meta_GA.window_len = best["window_len"]

#     # Stage 5
#     s5_meta_GA.dqmask = best["dqmask"]
#     s5_meta_GA.bg_thresh = best["bg_thresh"]
#     s5_meta_GA.bg_hw = best["bg_hw"]
#     s5_meta_GA.spec_hw = best["spec_hw"]
#     s5_meta_GA.bg_deg = best["bg_deg"]
#     s5_meta_GA.bg_method = best["bg_method"]
#     s5_meta_GA.p3thresh = best["p3thresh"]
#     s5_meta_GA.median_thresh = best["median_thresh"]
#     s5_meta_GA.window_len = best["window_len"]

#     # Perform parametric sweep
#     best_param, best_fitness_value = optimizers.parametric_sweep_S3(
#         objective_funcs.p7thresh,
#         bounds_p7thresh,
#         eventlabel,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#         params,
#         scaling_MAD_white,
#         scaling_MAD_spec,
#         scaling_chi2red,
#     )

#     # Update the 'best' dictionary
#     best["p7thresh"] = best_param[0]

#     # Print results
#     print("Best parameters: ", best_param[0])
#     print("Best fitness: ", best_fitness_value)

#     # Update history fitness score
#     history_fitness_score = {"p7thresh": best_fitness_value}

#     return (
#         best_param,
#         history_fitness_score,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#     )


# def expand(
#     s3_meta_GA,
#     s4_meta_GA,
#     s5_meta_GA,
#     best,
#     manual_clip_lists,
#     bounds_expand,
#     eventlabel,
#     params,
#     scaling_MAD_white,
#     scaling_MAD_spec,
#     scaling_chi2red,
# ):
#     """
#     Perform a parametric sweep of "expand".

#     Parameters:
#         s3_meta_GA (S3MetaClass): Metadata object for S3.
#         s4_meta_GA (S4MetaClass): Metadata object for S4.
#         s5_meta_GA (S5MetaClass): Metadata object for S5.
#         best (dict): Dictionary containing initial best values for parameters.
#         manual_clip_lists (list): List for manual clipping setup.
#         bounds_expand (tuple): Bounds for expand parameter.
#         eventlabel (str): Label for the event.
#         params (dict): Parameters for the optimizer.
#         scaling_MAD_white (float): Scaling factor for MAD white.
#         scaling_MAD_spec (float): Scaling factor for MAD spec.
#         scaling_chi2red (float): Scaling factor for reduced chi-squared.

#     Returns:
#         tuple: best_param (list), updated history_fitness_score (dict),
#         s3_meta_GA (S3MetaClass), s4_meta_GA (S4MetaClass),
#         s5_meta_GA (S5MetaClass).
#     """
#     # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
#     s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
#     s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

#     # Set Up Manual Clipping
#     s5_meta_GA.manual_clip = manual_clip_lists

#     # Set Fit Method
#     s5_meta_GA.fit_method = "lsq"

#     # Setup Meta / Define Initial Population
#     # Stage 3
#     s3_meta_GA.dqmask = best["dqmask"]
#     s3_meta_GA.bg_thresh = best["bg_thresh"]
#     s3_meta_GA.bg_hw = best["bg_hw"]
#     s3_meta_GA.spec_hw = best["spec_hw"]
#     s3_meta_GA.bg_deg = best["bg_deg"]
#     s3_meta_GA.bg_method = best["bg_method"]
#     s3_meta_GA.p3thresh = best["p3thresh"]
#     s3_meta_GA.median_thresh = best["median_thresh"]
#     s3_meta_GA.window_len = best["window_len"]
#     s3_meta_GA.p7thresh = best["p7thresh"]

#     # Stage 4
#     s4_meta_GA.dqmask = best["dqmask"]
#     s4_meta_GA.bg_thresh = best["bg_thresh"]
#     s4_meta_GA.bg_hw = best["bg_hw"]
#     s4_meta_GA.spec_hw = best["spec_hw"]
#     s4_meta_GA.bg_deg = best["bg_deg"]
#     s4_meta_GA.bg_method = best["bg_method"]
#     s4_meta_GA.p3thresh = best["p3thresh"]
#     s4_meta_GA.median_thresh = best["median_thresh"]
#     s4_meta_GA.window_len = best["window_len"]
#     s4_meta_GA.p7thresh = best["p7thresh"]

#     # Stage 5
#     s5_meta_GA.dqmask = best["dqmask"]
#     s5_meta_GA.bg_thresh = best["bg_thresh"]
#     s5_meta_GA.bg_hw = best["bg_hw"]
#     s5_meta_GA.spec_hw = best["spec_hw"]
#     s5_meta_GA.bg_deg = best["bg_deg"]
#     s5_meta_GA.bg_method = best["bg_method"]
#     s5_meta_GA.p3thresh = best["p3thresh"]
#     s5_meta_GA.median_thresh = best["median_thresh"]
#     s5_meta_GA.window_len = best["window_len"]
#     s5_meta_GA.p7thresh = best["p7thresh"]

#     # Perform parametric sweep
#     best_param, best_fitness_value = optimizers.parametric_sweep_S3(
#         objective_funcs.expand,
#         bounds_expand,
#         eventlabel,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#         params,
#         scaling_MAD_white,
#         scaling_MAD_spec,
#         scaling_chi2red,
#     )

#     # Update the 'best' dictionary
#     best["expand"] = best_param[0]

#     # Print results
#     print("Best parameters: ", best_param[0])
#     print("Best fitness: ", best_fitness_value)

#     # Update history fitness score
#     history_fitness_score = {"expand": best_fitness_value}

#     return (
#         best_param,
#         history_fitness_score,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#     )


# def sigma(
#     s3_meta_GA,
#     s4_meta_GA,
#     s5_meta_GA,
#     best,
#     manual_clip_lists,
#     bounds_sigma,
#     eventlabel,
#     last_s3_meta_outputdir,
#     params,
#     scaling_MAD_white,
#     scaling_MAD_spec,
#     scaling_chi2red,
# ):
#     """
#     Perform a parametric sweep of "sigma".

#     Parameters:
#         s3_meta_GA (S3MetaClass): Metadata object for S3.
#         s4_meta_GA (S4MetaClass): Metadata object for S4.
#         s5_meta_GA (S5MetaClass): Metadata object for S5.
#         best (dict): Dictionary containing initial best values for parameters.
#         manual_clip_lists (list): List for manual clipping setup.
#         bounds_sigma (tuple): Bounds for sigma parameter.
#         eventlabel (str): Label for the event.
#         last_s3_meta_outputdir (str): Output directory from the
#         last stage 3 meta.
#         params (dict): Parameters for the optimizer.
#         scaling_MAD_white (float): Scaling factor for MAD white.
#         scaling_MAD_spec (float): Scaling factor for MAD spec.
#         scaling_chi2red (float): Scaling factor for reduced chi-squared.

#     Returns:
#         tuple: best_param (list), updated history_fitness_score (dict),
#         s3_meta_GA (S3MetaClass), s4_meta_GA (S4MetaClass),
#         s5_meta_GA (S5MetaClass).
#     """
#     # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
#     s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
#     s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

#     # Set Up Manual Clipping
#     s5_meta_GA.manual_clip = manual_clip_lists

#     # Set Fit Method
#     s5_meta_GA.fit_method = "lsq"

#     # Setup Meta / Define Initial Population
#     # Stage 3
#     s3_meta_GA.dqmask = best["dqmask"]
#     s3_meta_GA.expand = best["expand"]
#     s3_meta_GA.bg_thresh = best["bg_thresh"]
#     s3_meta_GA.bg_hw = best["bg_hw"]
#     s3_meta_GA.spec_hw = best["spec_hw"]
#     s3_meta_GA.bg_deg = best["bg_deg"]
#     s3_meta_GA.bg_method = best["bg_method"]
#     s3_meta_GA.p3thresh = best["p3thresh"]
#     s3_meta_GA.median_thresh = best["median_thresh"]
#     s3_meta_GA.window_len = best["window_len"]
#     s3_meta_GA.p7thresh = best["p7thresh"]
#     s3_meta_GA.expand = best["expand"]

#     # Stage 4
#     s4_meta_GA.dqmask = best["dqmask"]
#     s4_meta_GA.expand = best["expand"]
#     s4_meta_GA.bg_thresh = best["bg_thresh"]
#     s4_meta_GA.bg_hw = best["bg_hw"]
#     s4_meta_GA.spec_hw = best["spec_hw"]
#     s4_meta_GA.bg_deg = best["bg_deg"]
#     s4_meta_GA.bg_method = best["bg_method"]
#     s4_meta_GA.p3thresh = best["p3thresh"]
#     s4_meta_GA.median_thresh = best["median_thresh"]
#     s4_meta_GA.window_len = best["window_len"]
#     s4_meta_GA.p7thresh = best["p7thresh"]
#     s4_meta_GA.expand = best["expand"]

#     # Stage 5
#     s5_meta_GA.dqmask = best["dqmask"]
#     s5_meta_GA.expand = best["expand"]
#     s5_meta_GA.bg_thresh = best["bg_thresh"]
#     s5_meta_GA.bg_hw = best["bg_hw"]
#     s5_meta_GA.spec_hw = best["spec_hw"]
#     s5_meta_GA.bg_deg = best["bg_deg"]
#     s5_meta_GA.bg_method = best["bg_method"]
#     s5_meta_GA.p3thresh = best["p3thresh"]
#     s5_meta_GA.median_thresh = best["median_thresh"]
#     s5_meta_GA.window_len = best["window_len"]
#     s5_meta_GA.p7thresh = best["p7thresh"]
#     s5_meta_GA.expand = best["expand"]

#     # Perform parametric sweep
#     best_param, best_fitness_value = optimizers.parametric_sweep_S4(
#         objective_funcs.sigma,
#         bounds_sigma,
#         eventlabel,
#         last_s3_meta_outputdir,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#         params,
#         scaling_MAD_white,
#         scaling_MAD_spec,
#         scaling_chi2red,
#     )

#     # Update the 'best' dictionary
#     best["sigma"] = best_param[0]

#     # Print results
#     print("Best parameters: ", best_param[0])
#     print("Best fitness: ", best_fitness_value)

#     # Update history fitness score
#     history_fitness_score = {"sigma": best_fitness_value}

#     return (
#         best_param,
#         history_fitness_score,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#     )


# def parametric_sweep_box_width(
#     s3_meta_GA,
#     s4_meta_GA,
#     s5_meta_GA,
#     best,
#     manual_clip_lists,
#     bounds_box_width,
#     eventlabel,
#     last_s3_meta_outputdir,
#     params,
#     scaling_MAD_white,
#     scaling_MAD_spec,
#     scaling_chi2red,
# ):
#     """
#     Perform a parametric sweep of "box_width".

#     Parameters:
#         s3_meta_GA (S3MetaClass): Metadata object for S3.
#         s4_meta_GA (S4MetaClass): Metadata object for S4.
#         s5_meta_GA (S5MetaClass): Metadata object for S5.
#         best (dict): Dictionary containing initial best values for parameters.
#         manual_clip_lists (list): List for manual clipping setup.
#         bounds_box_width (tuple): Bounds for box_width parameter.
#         eventlabel (str): Label for the event.
#         last_s3_meta_outputdir (str): Output directory from the
#         last stage 3 meta.
#         params (dict): Parameters for the optimizer.
#         scaling_MAD_white (float): Scaling factor for MAD white.
#         scaling_MAD_spec (float): Scaling factor for MAD spec.
#         scaling_chi2red (float): Scaling factor for reduced chi-squared.

#     Returns:
#         tuple: best_param (list), updated history_fitness_score (dict),
#         s3_meta_GA (S3MetaClass), s4_meta_GA (S4MetaClass),
#         s5_meta_GA (S5MetaClass).
#     """
#     # Overwrite ECF values with extracted ev.xwindow and ev.ywindow values
#     s3_meta_GA.xwindow = [best["xwindow_LB"], best["xwindow_UB"]]
#     s3_meta_GA.ywindow = [best["ywindow_LB"], best["ywindow_UB"]]

#     # Set Up Manual Clipping
#     s5_meta_GA.manual_clip = manual_clip_lists

#     # Set Fit Method
#     s5_meta_GA.fit_method = "lsq"

#     # Setup Meta / Define Initial Population
#     # Stage 3
#     s3_meta_GA.dqmask = best["dqmask"]
#     s3_meta_GA.bg_thresh = best["bg_thresh"]
#     s3_meta_GA.bg_hw = best["bg_hw"]
#     s3_meta_GA.spec_hw = best["spec_hw"]
#     s3_meta_GA.bg_deg = best["bg_deg"]
#     s3_meta_GA.bg_method = best["bg_method"]
#     s3_meta_GA.p3thresh = best["p3thresh"]
#     s3_meta_GA.median_thresh = best["median_thresh"]
#     s3_meta_GA.window_len = best["window_len"]
#     s3_meta_GA.p7thresh = best["p7thresh"]
#     s3_meta_GA.expand = best["expand"]

#     # Stage 4
#     s4_meta_GA.dqmask = best["dqmask"]
#     s4_meta_GA.bg_thresh = best["bg_thresh"]
#     s4_meta_GA.bg_hw = best["bg_hw"]
#     s4_meta_GA.spec_hw = best["spec_hw"]
#     s4_meta_GA.bg_deg = best["bg_deg"]
#     s4_meta_GA.bg_method = best["bg_method"]
#     s4_meta_GA.p3thresh = best["p3thresh"]
#     s4_meta_GA.median_thresh = best["median_thresh"]
#     s4_meta_GA.window_len = best["window_len"]
#     s4_meta_GA.p7thresh = best["p7thresh"]
#     s4_meta_GA.sigma = best["sigma"]
#     s4_meta_GA.expand = best["expand"]

#     # Stage 5
#     s5_meta_GA.dqmask = best["dqmask"]
#     s5_meta_GA.bg_thresh = best["bg_thresh"]
#     s5_meta_GA.bg_hw = best["bg_hw"]
#     s5_meta_GA.spec_hw = best["spec_hw"]
#     s5_meta_GA.bg_deg = best["bg_deg"]
#     s5_meta_GA.bg_method = best["bg_method"]
#     s5_meta_GA.p3thresh = best["p3thresh"]
#     s5_meta_GA.median_thresh = best["median_thresh"]
#     s5_meta_GA.window_len = best["window_len"]
#     s5_meta_GA.p7thresh = best["p7thresh"]
#     s5_meta_GA.sigma = best["sigma"]
#     s5_meta_GA.expand = best["expand"]

#     # Perform parametric sweep
#     best_param, best_fitness_value = optimizers.parametric_sweep_S4(
#         objective_funcs.box_width,
#         bounds_box_width,
#         eventlabel,
#         last_s3_meta_outputdir,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#         params,
#         scaling_MAD_white,
#         scaling_MAD_spec,
#         scaling_chi2red,
#     )

#     # Update the 'best' dictionary
#     best["box_width"] = best_param[0]

#     # Print results
#     print("Best parameters: ", best_param[0])
#     print("Best fitness: ", best_fitness_value)

#     # Update history fitness score
#     history_fitness_score = {"box_width": best_fitness_value}

#     return (
#         best_param,
#         history_fitness_score,
#         s3_meta_GA,
#         s4_meta_GA,
#         s5_meta_GA,
#     )
