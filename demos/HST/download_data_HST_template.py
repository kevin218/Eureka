#!/usr/bin/env python3

import eureka.lib.mastDownload as md

# Proposal/Program ID, can be string or int
proposal_id = '13467'

# List of one or more visit numbers
visits = [60]
# visits = np.arange(60,85)

# Instrument name, can be upper or lower case.
# Supported options include: WFC3, STIS, COS, or FGS.
inst = 'WFC3'

# Temporary download directory will be 'download_dir'/mastDownload/...
download_dir = '.'

# FITS file type (usually IMA, sometimes FLT)
subgroup = 'IMA'

# MAST API token for accessing data with exclusive access
# This can be generated at https://auth.mast.stsci.edu/token
mast_token = None

# Final destination of files after calling mast.consolidate
final_dir = './HD209458/ima'

# If data are public, no need to call md.login() or md.logout()
md.login(mast_token)
for vis in visits:
    # Download data from MAST Archive
    result = md.downloadHST(proposal_id, vis, inst, download_dir, subgroup)
    if result is not None:
        # Consolodate and move data into new directory
        md.consolidate(result, final_dir)

# Delete empty temporary directory structure
md.cleanup(download_dir)
md.logout()
