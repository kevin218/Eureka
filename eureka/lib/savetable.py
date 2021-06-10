from astropy.table import QTable
from astropy.io import ascii
import numpy as np
import os

def savetable(ev):
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

    filename = ev.workdir + '/S3_' + ev.eventlabel + "_spec_lc.txt"

    dims = ev.stdspec.shape #tuple (integration, wavelength position)

    bjdtdb = np.repeat(ev.bjdtdb, dims[1])
    wave_2d = np.tile(ev.wave_2d[0], dims[0])
    stdspec = ev.stdspec.flatten()
    stdvar = ev.stdvar.flatten()
    optspec = ev.optspec.flatten()
    opterr = ev.opterr.flatten()

    arr = [bjdtdb, wave_2d, stdspec, stdvar, optspec, opterr]
    table = QTable(arr, names=('int_mid_BJD_TDB', 'wavelength', 'stdspec', 'stdvar', 'optspec', 'opterr'))
    ascii.write(table, filename, format='ecsv', overwrite=True, fast_writer=True)
