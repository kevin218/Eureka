
def fixipmapping(ipparams, posflux, etc = [], retbinflux = False, retbinstd = False):
    """
  This function returns the fixed best-fit intra-pixel mapping.

    Parameters
    ----------
	ipparams :  tuple
                unused
    bestmip :   1D array, size = # of measurements
                Best-fit ip mapping
    
    Returns
    -------
    output :    1D array, size = # of measurements
                Intra-pixel-corrected flux multiplier

    Revisions
    ---------
    2010-08-03  Kevin Stevenson, UCF
			    kevin218@knights.ucf.edu
                Original version
    """
    
    bestmip, binflux, binstd = posflux

    #Return fit with or without binned flux
    if retbinflux == False and retbinstd == False:
        return bestmip
    elif retbinflux == True and retbinstd == True:
        return [bestmip, binflux, binstd]
    elif retbinflux == True:
        return [bestmip, binflux]
    else:
        return [bestmip, binstd]
