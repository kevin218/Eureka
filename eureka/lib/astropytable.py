from astropy.table import QTable
from astropy.table import Table
from astropy.io import ascii
import numpy as np
import os

def savetable_S3(filename, time, wave_1d, stdspec, stdvar, optspec, opterr):
    """
      Saves data in an event as .txt using astropy

      Parameters
      ----------
      event    : An Event instance.

    Description
    -----------
    Saves data stored in an event object as an table

      Returns
      -------
      .txt file

      Revisions
      ---------

    """

    dims = stdspec.shape #tuple (integration, wavelength position)

    orig_shapes = [str(time.shape), str(wave_1d.shape), str(stdspec.shape), str(stdvar.shape), str(optspec.shape), str(opterr.shape)]

    time = np.repeat(time, dims[1])
    wave_1d = np.tile(wave_1d, dims[0])
    stdspec = stdspec.flatten()
    stdvar = stdvar.flatten()
    optspec = optspec.flatten()
    opterr = opterr.flatten()

    arr = [time, wave_1d, stdspec, stdvar, optspec, opterr]
    
    try:
      table = QTable(arr, names=('time', 'wave_1d', 'stdspec', 'stdvar', 'optspec', 'opterr'))
      ascii.write(table, filename, format='ecsv', overwrite=True, fast_writer=True)
    except ValueError as e:
      raise ValueError("There was a shape mismatch between your arrays which had shapes:\n"+
                       "time, wave_1d, stdspec, stdvar, optspec, opterr\n"+
                       ",".join(orig_shapes)) from e


def readtable(filename):
    t = ascii.read(filename, format='ecsv')
    return t
