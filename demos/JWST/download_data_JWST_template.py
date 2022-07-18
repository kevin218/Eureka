#!/usr/bin/env python3

from astroquery.mast import Observations
import eureka.lib.mastDownload as md
from astropy.io import ascii

# Proposal/Program ID, can be string or int
proposal_id = '02734'

# Observation number
observation = 1

# Visit number
visit = 1

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
download_dir = './wasp96b'

# Final destination of files after calling mast.consolidate
final_dir = './wasp96b/S0'

# MAST API token for accessing data with exclusive access
# This can be generated at https://auth.mast.stsci.edu/token
mast_token = None
# If data are public, no need to call md.login() or md.logout()
md.login(mast_token)

# Apply standard filters to identify files for download
table = md.filterJWST(proposal_id, observation, visit, calib_level, subgroup)

# Optional, write out list of products from filtered table
ascii.write(table, download_dir+"/output.csv", format='csv')
# Optional, write out all products from given program, observation, visit
# md.writeTable_JWST(proposal_id, observation, visit,
#                    download_dir+"/output.csv", format='csv')

# Download data products, returns manifest of files downloaded.
manifest = Observations.download_products(table, curl_flag=False,
                                          download_dir=download_dir)

# Consolidate and move data into new directory
md.consolidate(manifest, final_dir)

# Sort files by their type
md.sortJWST(final_dir, final_dir+'/S0', "uncal.fits")

# Delete empty temporary directory structure
md.cleanup(download_dir)

# Logout
md.logout()
