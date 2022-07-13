from exotic_ld import StellarLimbDarkening
import numpy as np


def exotic_ld(meta, spec):
    '''Generate limb-darkening coefficients using 
       the exotic_ld package
       code based on exotic_ld documentation

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    spec :  Astreaus object 
        Data object of wavelength-like arrrays.

    Returns
    -------
    ld_coeffs : array
        Linear, Quadratic, Non-linear (3 and 4) limb-darkening coefficients
        
    
    '''
    # Set the observing mode
    if meta.inst == 'miri':
        mode = 'JWST_MIRI_' + meta.inst_filter
    elif meta.inst == 'nircam':
        mode = 'JWST_NIRCam_' + meta.inst_filter
    elif meta.inst == 'nirspec':
        mode = 'JWST_NIRSpec_' + meta.inst_filter
    elif meta.inst == 'niriss':
        mode = 'JWST_NIRISS_' + meta.inst_filter
    elif meta.inst == 'wfc3':
        mode = 'HST_WFC3_' + meta.inst_filter

    # Compute wavelength range, needs to be in Angstrom
    wsdata = np.array(meta.wave_low)
    wsdata = np.append(wsdata, meta.wave_hi[-1])
    wavelength_range = []
    for i in range(meta.nspecchan):
        wavelength_range.append([wsdata[i], wsdata[i+1]])
    wavelength_range = np.array(wavelength_range)
    if spec.wave_1d.attrs['wave_units'] == 'microns':
        wavelength_range *= 1e4

    # Set up the stellar parameters
    M_H = meta.metallicity
    Teff = meta.teff
    logg = meta.logg

    # Tell it where the data from the Zenodo link has been placed
    ld_data_path = meta.exotic_ld_direc

    # Tell it which stellar model grid you would like to use: '1D' or '3D'
    ld_model = meta.exotic_ld_grid
    
    # compute stellar limb darkening model
    sld = StellarLimbDarkening(M_H, Teff, logg, ld_model, ld_data_path) 
    
    lin_c1 = np.zeros((meta.nspecchan, 1))
    quad = np.zeros((meta.nspecchan, 2))
    nonlin_3 = np.zeros((meta.nspecchan, 3))
    nonlin_4 = np.zeros((meta.nspecchan, 4))
    for i in range(meta.nspecchan):
        # generate limb-darkening coefficients for each bin
        lin_c1[i] = [sld.compute_linear_ld_coeffs(wavelength_range[i], mode)[0]]
        quad[i] = sld.compute_quadratic_ld_coeffs(wavelength_range[i], mode)
        nonlin_3[i] = \
            sld.compute_3_parameter_non_linear_ld_coeffs(wavelength_range[i], 
                                                         mode)
        nonlin_4[i] = \
            sld.compute_4_parameter_non_linear_ld_coeffs(wavelength_range[i], 
                                                         mode)
    return lin_c1, quad, nonlin_3, nonlin_4