'''markdown
# Eureka! Optimizer - Instructions


# Introduction

* The Eureka! Optimizer offers automatic data optimization for Eureka! stages 3-6.
* The optimization suite currently offers two optimization methods:
  '''
  1. Parametric Optimization - Free parameters are optimized using parametric sweeps
     '''
     Parameters are evaluated for every value within a user-designated bounds.
     The optimal value for each parameter is selected based on a user-designated fitness value.

     This process is repeated sequentially for the stage 3 ECF input variables:
     - diffthresh
     - ap_hw
     - bg_hw
     - p3thresh 
     - median_thresh
     - window_len
     - p5thresh
     - p7thresh

     and the Stage 4 variables:
     - drift_range
     - highpassWidth
     - sigma
     - box_width

     Values for ap_hw & bg_hw are interdependent, and are thus evaluated as a nested loop. 
     This tests every possible combination within the user-designated bounds of both variables.

  2. Genetic Optimization - Free parameters are optimized using a genetic algorithm (GA)
     '''
     Assortments of several parameters are evaluated, optimizing several parameters simultaneously.
     Parameters are evaluated randomnly within user-designated bounds.
     The optimal value for each parameter is selected based on a user-designated fitness value.

     To control the number of assortments to test, we define a population size.
     The most fit assortment of values or "best-fit indiviual" is then selected for a number of generations.
     This process is repeated until the a designated fitness value is reached, meeting our optimization criteria (e.g. reduced chi-quared of 1.2 for light curve fitting). 

     NOTE: The default value for the fitness value is 0, thus the total number of runs for each step of the genetic optimization scales according to population_size * generations.

     An outline of the GA-based optimization scheme is provided below.


     Step 1 - Genetic optimizer evaluates stage 3 ECF input variables:
     - ap_hw
     - bg_hw

     Step 2 - Genetic optimizer evaluates stage 3 ECF input variables:
     - p3thresh 
     - median_thresh
     - window_len
     - p5thresh
     - p7thresh

     Step 3 - Genetic optimizer evaluates stage 4 ECF input variables:
     - drift_range
     - highpassWidth
     - sigma
     - box_width

     NOTE: diffthresh is evaluated via parametric sweep.
     '''


# Features

* Feature 1: Automated execution of Eureka! Stages 3-6 for HST WFC-3 observations (Stages 3, 4, 5, 6)
* Feature 2: Parametric Optimizer for ECF inputs (Stages 3, 4)
* Feature 3: Genetic Optimizer for ECF inputs (Stages 3, 4)
* Feature 4: Box extraction sizing for bright targets (Stage 3)
* Feature 5: Automated Limb-Darkening Parameter Retrieval using ExoMAST values (Stages 4, 5)
* Feature 6: Automated EPF generation using ExoMAST values (Stage 5)


# Installation

* Please follow the installation instructions for Eureka! on https://eurekadocs.readthedocs.io/en/latest/installation.html#installation.
* To enable offline exoplanet information retrieval from the MAST exoplanet database, download all_planet_data.pkl to your 'home/Data/' directory.


# Usage

* To operate the optimizer, the standard Eureka instructions for processing HST observations must first be followed:
  '''
  1. Download the observation data
  2. Download the Horizons file
  3. Set up the .ecf electronic configuration files (ECF) for Stages 3-6

  Make sure to enter the centroidguess values in the Stage 3 ECF. Use DS9 to verify the star location in the calibration image.

  '''
* Now we may begin using the optimizer.
  '''
  4. Complete the optimizer_inputs.txt file.
     The main parts to fill out are:
     - planet_name & eventlabel in the GENERAL section
     - bounds_ap_hw & bounds_bg_hw in the DEFINE BOUNDS section. Inspect the spectra in DS9 or Eureka! to better narrow your guess for the bounds.
     - everything in the DEFINE FILEPATHS section

     The rest of the file may be run using the default inputs.

     An example is shown below.


    ## EUREKA! OPTIMIZATION -- INPUT PARAMETERS


    ### GENERAL
    planet_name = 'HD 86226 c'   # Name of exoplanet as spelled in the MAST archive
    eventlabel = 'HD86226c'

    ### TARGET BRIGHTNESS
    bright_star = False            # True will select a wider ywindow for the extraction box

    ### BOX EXTRACTION
    xwindow_selection = 'auto'    # 'manual' or 'auto'; manual will use the values in the S3 ECF
    ywindow_selection = 'auto'    # 'manual' or 'auto'; manual will use the values in the S3 ECF

    ## SPEC_HW & BG_HW ##
    spec_hw_selection = 'auto'    # 'manual' or 'auto'; manual will use the values in the S3 ECF
    bg_hw_selection = 'auto'      # 'manual' or 'auto'; manual will use the values in the S3 ECF

    ### OPTIMIZER
    optimizer = "parametric"      # "parametric" or "genetic" (used for optimization of S3-S4 inputs)
    target_fitness = 0            # Target chi2red value (0 is default, where # of runs = pop_size * generations)

    #### Fitness Scaling
    scaling_chi2red = 0.0         # Default = 0.0
    scaling_MAD_spec = 1.0        # Default = 1.0
    scaling_MAD_white = 1.0       # Default = 1.0

    #### GENETIC ALGORITHM   <!-- # Only fill out this section if optimizer == 'genetic'
    ##### Step 1 Settings        <!-- # Optimization of S3 ECF values: ap_bg & bg_hw
    population_size_Step_1 = 8    # Must be a multiple of 2
    generations_Step_1 = 7        # Number of Generations for Step 1 of GA

    ##### Step 2 Settings        <!-- # Optimization of S3 ECF values: p3thresh, median_thresh, window_len, p5thresh, p7thresh
    population_size_Step_2 = 10   # Must be a multiple of 2
    generations_Step_2 = 7        # Number of Generations for Step 2 of GA

    ##### Step 3 Settings        <!-- # Optimization of S4 ECF values: drift_range, highpassWidth, sigma, box_width
    population_size_Step_3 = 8    # Must be a multiple of 2
    generations_Step_3 = 7        # Number of Generations for Step 3 of GA

    ### SKIP STEPS 
    skip_xwindow_crop = True      # If True, xwindow cropping will be skipped. False recommended for first run.
    skip_ywindow_crop = True      # If True, xwindow cropping will be skipped. False recommended for first run.

    ### DEFINE BOUNDS
    #### Crop the box extraction window
    bounds_xwindow_crop = [0, 25]   # Range of values for optimizer to test
    bounds_ywindow_crop = [0, 40]   # Range of values for optimizer to test

    ### HST ONLY
    offset_trace = 5               # Offset xrange of WFC3 Trace (5 is default for 256x256 pixel grid)

    ### Manual Clipping
    clip_first_orbit = True        # Clip the first orbit. True or False.
    manual_clip_npoints = 3        # Clip first n data points of each frame. Value must be greater than 0.

    ### Limb Darkening
    enable_exotic_ld = True        # If True, the final run of the white light data, as well as the spectroscopic data, will be analyzed using exotic limb darkening.

    use_generate_ld = 'exotic-ld'  # Turn on use_generate_ld and enter paths for ld files


    ###  DEFINE FILEPATHS
    #### File path to save optimization results
    outputdir_optimization = "/home/DataAnalysis/HST/HD86226c/Optimized/"

    #### Path to observation data
    loc_sci = '/home/Data/HST/WFC3/HD86226c/Visit08-11/' 

    #### File path for MAST data
    exomast_file = "/home/Data/all_planet_data.pkl"

    #### Path to exotic-ld ancillary files 
    exotic_ld_direc = '/home/Data/exotic-ld_data/'

    #### File path for exotic-ld throughput file 
    exotic_ld_file = '/home/Data/exotic-ld_data/Sensitivity_files/HST_WFC3_G141_throughput.csv'

    #### File path for ld file (white) 
    ld_file = '/home/Data/exotic-ld_data/Sensitivity_files/HST_WFC3_G141_throughput.csv'

    #### File path for ld file (white) 
    ld_file_white = '/home/Data/exotic-ld_data/Sensitivity_files/HST_WFC3_G141_throughput.csv'


  5. To execute the optimizer, run the notebook, run_eureka_optimization_wfc.ipynb.
     Optimized results are placed in the outputdir_optimization folder specified in the optimizer_inputs.txt file.

     The final results are optimized based on the algorithm's fitness value, defined in the FITNESS SCALING section of the optimizer_inputs.txt file.
     For further explanation, lets refer to the default scalings:

    scaling_chi2red = 0.0         # Default = 0.0
    scaling_MAD_spec = 1.0        # Default = 1.0
    scaling_MAD_white = 1.0       # Default = 1.0

    Using these values, the optimizer will disregard the reduced chi-squared value from the light-curve fitting in Stage 5 of Eureka!, 
    and will instead prioritize lowering the MAD values for the white light curve and the spectroscopic light curves.

    If we wanted to re-run the optimizer, prioritizing the output quality of the white light curve, we could apply the following scalings instead:

    scaling_chi2red = 0.0         # Default = 0.0
    scaling_MAD_spec = 0.1        # Default = 1.0
    scaling_MAD_white = 1.0       # Default = 1.0
    
  6. If necessary, repeat Steps 4 - 5 with different scalings until the optimized output is satisfactory. 

  '''


# License
  '''
  Distributed under the MIT License.
  '''


# Contact

* Reza Ashtari - [Reza.Ashtari@jhuapl.edu](mailto:Reza.Ashtari@jhuapl.edu)
* Kevin Stevenson - [Kevin.Stevenson@jhuapl.edu](mailto:Kevin.Stevenson@jhuapl.edu)
<!-- * Project Link: [https://github.com/your_username/project-name](https://github.com/your_username/project-name) -->

'''
