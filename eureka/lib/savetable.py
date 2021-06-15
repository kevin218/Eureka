from astropy.table import QTable
from astropy.io import ascii
import numpy as np
import os

def savetable(md, bjdtdb, wave_2d, stdspec, stdvar, optspec, opterr):
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

    filename = md.workdir + '/S3_' + md.eventlabel + "_spec_lc.txt"

    dims = stdspec.shape #tuple (integration, wavelength position)

    bjdtdb = np.repeat(bjdtdb, dims[1])
    wave_2d = np.tile(wave_2d[0], dims[0])
    stdspec = stdspec.flatten()
    stdvar = stdvar.flatten()
    optspec = optspec.flatten()
    opterr = opterr.flatten()

    arr = [bjdtdb, wave_2d, stdspec, stdvar, optspec, opterr]
    table = QTable(arr, names=('int_mid_BJD_TDB', 'wavelength', 'stdspec', 'stdvar', 'optspec', 'opterr'))
    ascii.write(table, filename, format='ecsv', overwrite=True, fast_writer=True)
