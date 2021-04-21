#! /usr/bin/env python

# Generic Stage 3 reduction pipeline

"""
# Proposed Steps
# -------- -----
# 1.  Read in all data frames and header info from Stage 2 data products
# 2.  Record JD and other relevant header information
# 3.  Apply light-time correction (if necessary)
# 4.  Calculate trace and 1D+2D wavelength solutions (if necessary)
# 5.  Make flats, apply flat field correction (if necessary)
# 6.  Manually mask regions
# 7.  Compute difference frames
# 8.  Perform outlier rejection of BG region
# 9.  Background subtraction
# 10. Compute 2D drift, apply rough (integer-pixel) correction
# 11. Full-frame outlier rejection for time-series stack of NDRs
# 12. Apply sub-pixel 2D drift correction
# 13. Extract spectrum through summation
# 14. Compute median frame
# 15. Optimal spectral extraction
# 16. Save Stage 3 data products
# 17. Produce plots
"""
