from astropy.table import QTable
from astropy.io import ascii
import numpy as np


def savetable_S5(filename, time, wavelength, bin_width, lcdata, lcerr, model,
                 residuals):
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


def savetable_S6(filename, key, wavelength, bin_width, value, error):
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


def readtable(filename):
    return ascii.read(filename, format='ecsv')
