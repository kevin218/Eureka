from astropy.table import QTable
from astropy.io import ascii
import numpy as np

from .split_channels import get_trim


def savetable_S1(filename, scale_factor):
    """Save the scale factor from Stage 1 as an ECSV.

    Parameters
    ----------
    filename : str
        The fully qualified filename that the results will be stored in.
    scale_factor : ndarray (2D)
        The bias scale factor of dimension nints by ngroup

    Raises
    ------
    ValueError
        There was a shape mismatch between your arrays
    """
    nint, ngroup = scale_factor.shape

    names = []
    for ii in range(1, ngroup+1):
        names.append(f'group{ii}')
    names = tuple(names)

    table = QTable(scale_factor, names=names)
    ascii.write(table, filename, format='ecsv', overwrite=True,
                fast_writer=True)
    return


def savetable_S5(filename, meta, time, wavelength, bin_width, lcdata, lcerr,
                 individual_models, model, residuals):
    """Save the results from Stage 5 as an ECSV file.

    Parameters
    ----------
    filename : str
        The fully qualified filename that the results will be stored in.
    time : ndarray (1D)
        The times for each data point.
    wavelength : ndarray (1D)
        The wavelengths of each data point.
    bin_width : ndarray (1D)
        The width of each wavelength bin.
    lcdata : ndarray (1D)
        The normalized flux measurements for each data point.
    lcerr : ndarray (1D)
        The normalized uncertainties for each data point.
    individual_models : ndarray (2D)
        An array containing pairs of model names and evaluated models.
    model : ndarray (1D)
        The predicted values from the fitted model.
    residuals : ndarray (1D)
        The residuals from lcdata - model.

    Raises
    ------
    ValueError
        There was a shape mismatch between your arrays
    """
    dims = [len(time), len(wavelength)]

    if not meta.multwhite:
        time = np.tile(time, dims[1])
        wavelength = np.repeat(wavelength, dims[0])
        bin_width = np.repeat(bin_width, dims[0])
    else:
        terse_waves = np.copy(wavelength)
        terse_widths = np.copy(bin_width)
        wavelength = np.zeros(dims[0])
        bin_width = np.zeros(dims[0])
        for chan in range(dims[1]):
            trim1, trim2 = get_trim(meta.nints, chan)
            wavelength[trim1:trim2] = terse_waves[chan]
            bin_width[trim1:trim2] = terse_widths[chan]

    orig_shapes = [str(time.shape), str(wavelength.shape),
                   str(bin_width.shape), str(lcdata.shape), str(lcerr.shape)]
    orig_shapes.extend([str(individual_models[i, 1].shape)
                        for i in range(individual_models.shape[0])])
    orig_shapes.extend([str(model.shape), str(residuals.shape)])

    lcdata = lcdata.flatten()
    lcerr = lcerr.flatten()
    model_names = individual_models[:, 0]
    model_values = individual_models[:, 1]
    full_model = model.flatten()
    residuals = residuals.flatten()

    arr = [time, wavelength, bin_width, lcdata, lcerr, *model_values,
           full_model, residuals]

    try:
        table = QTable(arr, names=('time', 'wavelength', 'bin_width',
                                   'lcdata', 'lcerr', *model_names, 'model',
                                   'residuals'))
        ascii.write(table, filename, format='ecsv', overwrite=True,
                    fast_writer=True)
    except ValueError as e:
        raise ValueError("There was a shape mismatch between your arrays which"
                         " had shapes:\n"
                         "time, wavelength, bin_width, lcdata, lcerr, " +
                         ', '.join(model_names) +
                         ", model, residuals\n" +
                         ", ".join(orig_shapes)) from e


def savetable_S6(filename, key, wavelength, bin_width, value, error):
    """Save the results from Stage 6 as an ECSV.

    Parameters
    ----------
    filename : str
        The fully qualified filename that the results will be stored in.
    key : str
        The parameter being saved.
    wavelength : ndarray (1D)
        The wavelengths of each data point.
    bin_width : ndarray (1D)
        The width of each wavelength bin.
    value : ndarray (1D)
        The fitted value at each wavelength.
    error : ndarray (1D)
        The uncertainty on each value.

    Raises
    ------
    ValueError
        There was a shape mismatch between your arrays
    """
    orig_shapes = [str(wavelength.shape), str(bin_width.shape),
                   str(value.shape), str(error[0].shape),
                   str(error[1].shape)]

    arr = [wavelength.flatten(), bin_width.flatten(), value.flatten(),
           error[0].flatten(), error[1].flatten()]

    try:
        table = QTable(arr, names=('wavelength', 'bin_width', key+'_value',
                                   key+'_errorneg', key+'_errorpos'))
        ascii.write(table, filename, format='ecsv', overwrite=True,
                    fast_writer=True)
    except ValueError as e:
        raise ValueError("There was a shape mismatch between your arrays which"
                         " had shapes:\n"
                         f"wavelength, bin_width, {key}_value, {key}_errorneg,"
                         " {key}_errorpos\n"
                         ",".join(orig_shapes)) from e


def savetable_S6_ul(filename, key, wavelength, bin_width, value, error,
                    f_3sig, f_bool):
    """Save the results from Stage 6 as an ECSV.

    Parameters
    ----------
    filename : str
        The fully qualified filename that the results will be stored in.
    key : str
        The parameter being saved.
    wavelength : ndarray (1D)
        The wavelengths of each data point.
    bin_width : ndarray (1D)
        The width of each wavelength bin.
    value : ndarray (1D)
        The fitted value at each wavelength.
    error : ndarray (1D)
        The uncertainty on each value.
    f_3sig : ndarray (1D)
        The 3-sigma upper limit on the flux.
    f_bool : ndarray (1D)
        True/False boolean array, where True indicates values should be
        reported as an upper limit.

    Raises
    ------
    ValueError
        There was a shape mismatch between your arrays
    """
    orig_shapes = [str(wavelength.shape), str(bin_width.shape),
                   str(value.shape), str(error[0].shape),
                   str(error[1].shape), str(f_3sig.shape),
                   str(f_bool.shape)]

    arr = [wavelength.flatten(), bin_width.flatten(), value.flatten(),
           error[0].flatten(), error[1].flatten(), f_3sig.flatten(),
           f_bool.flatten()]

    try:
        table = QTable(arr, names=('wavelength', 'bin_width', f'{key}_value',
                                   f'{key}_errorneg', f'{key}_errorpos',
                                   f'{key}_3sigma', 'upper_limit?'))
        ascii.write(table, filename, format='ecsv', overwrite=True,
                    fast_writer=True)
    except ValueError as e:
        raise ValueError("There was a shape mismatch between your arrays which"
                         " had shapes:\n"
                         f"wavelength, bin_width, {key}_value, {key}_errorneg,"
                         " {key}_errorpos, 3sigma, upper_limit?\n"
                         ",".join(orig_shapes)) from e


def readtable(filename):
    """Read in a saved ECSV file.

    Parameters
    ----------
    filename : str
        The fully qualified filename of the file to read.

    Returns
    -------
    astropy.table.QTable
        The table previously saved by savetable_S5 or savetable_S6.
    """
    return ascii.read(filename, format='ecsv')
