from exotic_ld import StellarLimbDarkening
import numpy as np
import pandas as pd


def exotic_ld(meta, spec, log, white=False):
    '''Generate limb-darkening coefficients using the exotic_ld package.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    spec :  Astreaus object
        Data object of wavelength-like arrrays.
    log : logedit.Logedit
        The open log in which notes from this step can be added.
    white : bool; optional
        If True, compute the limb-darkening parameters for the white-light
        light curve. Defaults to False.

    Returns
    -------
    ld_coeffs : tuple
        Linear, Quadratic, Non-linear (3 and 4) limb-darkening coefficients

    Notes
    -----
    History:

    - July 2022, Eva-Maria Ahrer
        Initial version based on exotic_ld documentation.
    '''

    log.writelog("...using exotic-ld package...",
                 mute=(not meta.verbose))

    # Set the observing mode
    custom_wavelengths = None
    custom_throughput = None

    if hasattr(meta, 'exotic_ld_file') and meta.exotic_ld_file is not None:
        mode = 'custom'
        log.writelog("Using custom throughput file " +
                     meta.exotic_ld_file,
                     mute=(not meta.verbose))
        # load custom file
        custom_data = pd.read_csv(meta.exotic_ld_file)
        custom_wavelengths = custom_data['wave'].values
        custom_throughput = custom_data['tp'].values
        if (custom_wavelengths[0] > 0.3) and (custom_wavelengths[0] < 30):
            log.writelog("Custom wavelengths appear to be in microns. " +
                         "Converting to Angstroms.")
            custom_wavelengths *= 1e4
    elif meta.inst == 'miri':
        mode = 'JWST_MIRI_' + meta.inst_filter
    elif meta.inst == 'nircam':
        mode = 'JWST_NIRCam_' + meta.inst_filter
    elif meta.inst == 'nirspec':
        mode = 'JWST_NIRSpec_' + meta.inst_filter
    elif meta.inst == 'niriss':
        mode = 'JWST_NIRISS_' + meta.inst_filter
    elif meta.inst == 'wfc3':
        mode = 'HST_WFC3_' + meta.inst_filter

    # Compute wavelength ranges
    if white:
        wavelength_range = np.array([meta.wave_min, meta.wave_max],
                                    dtype=float)
        wavelength_range = np.repeat(wavelength_range[np.newaxis],
                                     meta.nspecchan, axis=0)
    else:
        wsdata = np.array(meta.wave_low, dtype=float)
        wsdata = np.append(wsdata, meta.wave_hi[-1])
        wavelength_range = []
        for i in range(meta.nspecchan):
            wavelength_range.append([wsdata[i], wsdata[i+1]])
        wavelength_range = np.array(wavelength_range, dtype=float)

    # wavelength range needs to be in Angstrom
    if spec.wave_1d.attrs['wave_units'] == 'microns':
        wavelength_range *= 1e4

    # compute stellar limb darkening model
    if hasattr(meta, "custom_si_grid") and meta.exotic_ld_grid == 'custom':
        # read the wavelengths, Mus, and intensity grid from file
        # 1st column is the wavelengths. Skip the header and row of Mus
        # also convert to angstrom!
        s_wvs = (np.genfromtxt(meta.custom_si_grid,
                               skip_header=2, usecols=[0]).T)*1e4
        # 1st row after the header is the Mus. Skip header, read 1 line
        # file has increasing Mus, Exotic requires decreasing, so flip
        s_mus = np.flip(np.genfromtxt(meta.custom_si_grid,
                                      skip_header=1, max_rows=1))
        # Now get the rest of the file. Skip header and row of Mus.
        # file has increasing Mus, Exotic requires decreasing, so flip
        custom_si = np.flip(np.genfromtxt(meta.custom_si_grid,
                                          skip_header=2)[:, 1:], axis=1)

        sld = StellarLimbDarkening(ld_data_path=meta.exotic_ld_direc,
                                   ld_model="custom",
                                   custom_wavelengths=s_wvs,
                                   custom_mus=s_mus,
                                   custom_stellar_model=custom_si)
    else:
        sld = StellarLimbDarkening(meta.metallicity, meta.teff, meta.logg,
                                   meta.exotic_ld_grid, meta.exotic_ld_direc)

    lin_c1 = np.zeros((meta.nspecchan, 1))
    quad = np.zeros((meta.nspecchan, 2))
    nonlin_3 = np.zeros((meta.nspecchan, 3))
    nonlin_4 = np.zeros((meta.nspecchan, 4))
    for i in range(meta.nspecchan):
        # generate limb-darkening coefficients for each bin
        lin_c1[i] = sld.compute_linear_ld_coeffs(wavelength_range[i],
                                                 mode, custom_wavelengths,
                                                 custom_throughput)[0]
        quad[i] = sld.compute_quadratic_ld_coeffs(wavelength_range[i], mode,
                                                  custom_wavelengths,
                                                  custom_throughput)
        nonlin_3[i] = \
            sld.compute_3_parameter_non_linear_ld_coeffs(wavelength_range[i],
                                                         mode,
                                                         custom_wavelengths,
                                                         custom_throughput)
        nonlin_4[i] = \
            sld.compute_4_parameter_non_linear_ld_coeffs(wavelength_range[i],
                                                         mode,
                                                         custom_wavelengths,
                                                         custom_throughput)
    return lin_c1, quad, nonlin_3, nonlin_4


def spam_ld(meta, white=False):
    '''Read limb-darkening coefficients generated using SPAM.

    Parameters
    ----------
    meta : eureka.lib.readECF.MetaClass
        The metadata object.
    white : bool; optional
        If True, compute the limb-darkening parameters for the white-light
        light curve. Defaults to False.

    Returns
    -------
    ld_coeffs : tuple
        Linear, Quadratic, Non-linear (3 and 4) limb-darkening coefficients

    Notes
    -----
    History:

    - February 2024, Kevin Stevenson
        Initial version based on exotic_ld above.
    '''
    # Compute wavelength ranges
    if white:
        wavelength_range = np.array([meta.wave_min, meta.wave_max],
                                    dtype=float)
        wavelength_range = np.repeat(wavelength_range[np.newaxis],
                                     meta.nspecchan, axis=0)
    else:
        wsdata = np.array(meta.wave_low, dtype=float)
        wsdata = np.append(wsdata, meta.wave_hi[-1])
        wavelength_range = []
        for i in range(meta.nspecchan):
            wavelength_range.append([wsdata[i], wsdata[i+1]])
        wavelength_range = np.array(wavelength_range, dtype=float)

    # Load SPAM file
    # First column is wavelength in microns
    # Remaining 1-4 colums are LD parameters
    sld = np.genfromtxt(meta.spam_file, unpack=True)
    sld_wave = sld[0]

    num_ld_coef = sld.shape[0] - 1
    ld_coeffs = np.zeros((meta.nspecchan, num_ld_coef))
    for i in range(meta.nspecchan):
        # Find valid indices
        wl_bin = wavelength_range[i]
        ii = np.where(np.logical_and(sld_wave >= wl_bin[0],
                                     sld_wave <= wl_bin[1]))[0]
        # Average limb-darkening coefficients for each bin
        for j in range(num_ld_coef):
            ld_coeffs[i, j] = np.mean(sld[j+1, ii])

    # Create list of NaNs
    nan1 = np.empty([meta.nspecchan, 1])*np.nan
    nan2 = np.empty([meta.nspecchan, 2])*np.nan
    nan3 = np.empty([meta.nspecchan, 3])*np.nan
    nan4 = np.empty([meta.nspecchan, 4])*np.nan
    ld_list = [nan1, nan2, nan3, nan4]
    # Replace relevant item with actual values
    ld_list[num_ld_coef-1] = ld_coeffs
    return ld_list
