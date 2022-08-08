from astropy.table import QTable
from astropy.io import ascii
import numpy as np


def savetable_S5(filename, time, wavelength, bin_width, lcdata, lcerr, model,
                 residuals):
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

    orig_shapes = [str(time.shape), str(wavelength.shape),
                   str(bin_width.shape), str(lcdata.shape), str(lcerr.shape),
                   str(model.shape), str(residuals.shape)]

    time = np.tile(time, dims[1])
    wavelength = np.repeat(wavelength, dims[0])
    bin_width = np.repeat(bin_width, dims[0])
    lcdata = lcdata.flatten()
    lcerr = lcerr.flatten()
    model = model.flatten()
    residuals = residuals.flatten()

    arr = [time, wavelength, bin_width, lcdata, lcerr, model, residuals]

    try:
        table = QTable(arr, names=('time', 'wavelength', 'bin_width',
                                   'lcdata', 'lcerr', 'model', 'residuals'))
        ascii.write(table, filename, format='ecsv', overwrite=True,
                    fast_writer=True)
    except ValueError as e:
        raise ValueError("There was a shape mismatch between your arrays which"
                         " had shapes:\n"
                         "time, wavelength, bin_width, lcdata, lcerr, model, "
                         "residuals\n"
                         ",".join(orig_shapes)) from e


def savetable_S6(filename, wavelength, bin_width, tr_depth, tr_depth_err,
                 ecl_depth, ecl_depth_err):
    """Save the results from Stage 6 as an ECSV.

    Parameters
    ----------
    filename : str
        The fully qualified filename that the results will be stored in.
    wavelength : ndarray (1D)
        The wavelengths of each data point.
    bin_width : ndarray (1D)
        The width of each wavelength bin.
    tr_depth : ndarray (1D)
        The fitted transit depth at each wavelength.
    tr_depth_err : ndarray (1D)
        The uncertainty on each transit depth.
    ecl_depth : ndarray (1D)
        The fitted eclipse depth at each wavelength.
    ecl_depth_err : ndarray (1D)
        The uncertainty on each eclipse depth.

    Raises
    ------
    ValueError
        There was a shape mismatch between your arrays
    """
    orig_shapes = [str(wavelength.shape), str(bin_width.shape),
                   str(tr_depth.shape), str(tr_depth_err[0].shape),
                   str(tr_depth_err[1].shape), str(ecl_depth.shape),
                   str(ecl_depth_err[0].shape), str(ecl_depth_err[0].shape)]

    arr = [wavelength.flatten(), bin_width.flatten(), tr_depth.flatten(),
           tr_depth_err[0].flatten(), tr_depth_err[1].flatten(),
           ecl_depth.flatten(), ecl_depth_err[0].flatten(),
           ecl_depth_err[1].flatten()]

    try:
        table = QTable(arr, names=('wavelength', 'bin_width', 'tr_depth',
                                   'tr_depth_errneg', 'tr_depth_errpos',
                                   'ecl_depth', 'ecl_depth_errneg',
                                   'ecl_depth_errpos'))
        ascii.write(table, filename, format='ecsv', overwrite=True,
                    fast_writer=True)
    except ValueError as e:
        raise ValueError("There was a shape mismatch between your arrays which"
                         " had shapes:\n"
                         "wavelength, bin_width, tr_depth, tr_depth_errneg, "
                         "tr_depth_errpos, ecl_depth, ecl_depth_errneg, "
                         "ecl_depth_errpos\n"
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
