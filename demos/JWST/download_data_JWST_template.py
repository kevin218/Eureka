#!/usr/bin/env python3

import eureka.lib.mastDownload as md

# Proposal/Program ID, can be string or int
proposal_id = '02734'

# List of one or more visit numbers
visits = [2]
# visits = np.arange(1,4)

# Calibration level, list
# (0 = raw, 1 = uncalibrated, 2 = calibrated, 3 = science product,
# 4 = contributed science product)
calib_level = [1]

# FITS file type, varies by calib_level.
# 1: UNCAL, GS-ACQ1, GS-ACQ2, GS-FG, GS-ID, GS-TRACK
# 2: CAL, CALINTS, RATE, RATEINTS, X1DINTS, ANNNN_CRFINTS,
# GS-ACQ1, GS-ACQ2, GS-FG, GS-ID, GS-TRACK, RAMP
# 3: X1DINTS, WHTLT
subgroup = 'UNCAL'

# Temporary download directory will be 'download_dir'/mastDownload/...
download_dir = '.'

# MAST API token for accessing data with exclusive access
# This can be generated at https://auth.mast.stsci.edu/token
mast_token = None

# Final destination of files after calling mast.consolidate
final_dir = './wasp96b/S1'

# If data are public, no need to call md.login() or md.logout()
md.login(mast_token)
for vis in visits:
    # Download data from MAST Archive
    result = md.downloadJWST(proposal_id, vis, calib_level, subgroup,
                             download_dir)
    if result is not None:
        # Consolodate and move data into new directory
        md.consolidate(result, final_dir)
        # Sort data into science and calibration folders (scan vs direct image)
        # md.sort(final_dir)

# Delete empty temporary directory structure
md.cleanup(download_dir)
md.logout()
