import os

import eureka.lib.plots
import eureka.optimizer.S1opt_optimizer as s1opt
import eureka.optimizer.S3opt_optimizer as s3opt

# Set up some parameters to make plots look nicer.
# You can set usetex=True if you have LaTeX installed
eureka.lib.plots.set_rc(style='eureka', usetex=False, filetype='.png')

"""
Eureka! Optimization
--------------------

Description:
This script is designed to optimize the Stage 1, 3, and 4 ECF parameters for
JWST time-series observations.

Inputs:
- Loaded from a Eureka! control file, see S1opt_template.ecf/S3opt_template.ecf.
- Parameters not specified for optimization will adopt the values listed in the
    standard ECF (e.g., S1_<eventlabel>.ecf or S3_<eventlabel>.ecf).

Outputs:
- The metadata object
- The fitness score after optimizing each parameter.
- The best parameter values found during the optimization.
"""

eventlabel = 'nirspec_fs_template'
ecf_path = '.'+os.sep

if __name__ == "__main__":
    # To skip one or more stages that were already run,
    # just comment them out below

    # Stage 1 optimization
    s1opt_meta, history, best = s1opt.wrapper(eventlabel,
                                              ecf_path=ecf_path,
                                              initial_run=True,
                                              final_run=True)

    # Stages 3 and 4 optimization
    s3opt_meta, history, best = s3opt.wrapper(eventlabel,
                                              ecf_path=ecf_path,
                                              initial_run=True,
                                              final_run=True)
