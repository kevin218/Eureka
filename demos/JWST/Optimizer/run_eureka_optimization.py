import os
import eureka.optimizer.S1opt_optimizer as s1opt
import eureka.optimizer.S3opt_optimizer as s3opt

"""
Eureka! Optimization
--------------------

Description:
This script is designed to optimize the Stage 1,3, and 4 ECF parameters for
JWST time-series observations.

Inputs:
- Loaded from an input text file, 'optimizer_inputs_nirspec_PRISM.txt'.
- Parameters not specified for optimization in the optimizer input text file will assume ECF values by default.

Outputs:
- Optimized parameter values saved in the "best_params" dictionary.
- Optimized ECFs saved in the "opt_ECFs" folder.
- Log file containing the results of each optimization step.
"""

eventlabel = 'nirspec_fs_template'
ecf_path = '.'+os.sep

if __name__ == "__main__":
    # To skip one or more stages that were already run,
    # just comment them out below

    # Stage 1 optimization
    s3opt_meta, history, best = s1opt.optimize(eventlabel,
                                               ecf_path=ecf_path,
                                               initial_run=True,
                                               final_run=True)

    # Stages 3 and 4 optimization
    s3opt_meta, history, best = s3opt.wrapper(eventlabel,
                                              ecf_path=ecf_path,
                                              initial_run=True,
                                              final_run=True)
