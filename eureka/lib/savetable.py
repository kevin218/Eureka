from astropy.table import QTable
from astropy.io import ascii
import numpy as np
import os

def savetable(ev):
    filename = ev.workdir + '/S3_' + ev.eventlabel + "_spec_lc.txt"
    #if not os.path.exists(filename):
    fits_filenames = ev.segment_list

    n_files = len(ev.shapes)

    def format_like_time(array):
        res = np.concatenate(
            [np.concatenate([[array[j][i] for ii in range(ev.shapes[j][1])] for i in np.arange(ev.shapes[j][0])]).ravel() for j in
             np.arange(len(ev.shapes))]).ravel()
        return res

    m = np.int8(np.concatenate([np.ones(np.prod(ev.shapes[j]))*j for j in np.arange(len(ev.shapes))]).ravel())
    n = np.int8(np.concatenate([np.repeat(np.arange(ev.shapes[j][0]), ev.shapes[j][1]) for j in np.arange(len(ev.shapes))]).ravel())

    wave = np.concatenate([[ev.wave[j] for ii in range(ev.shapes[j][0])] for j in np.arange(len(ev.shapes))]).ravel()

    mjdutc = format_like_time(ev.mjdutc)
    bjdtdb = format_like_time(ev.bjdtdb)

    stdspec = ev.stdspec.flatten()
    stdvar = ev.stdvar.flatten()

    arr = [m, n, wave, mjdutc, bjdtdb, stdspec, stdvar]#, fits_filenames[int(ev.m[0])]]
    table = QTable(arr, names=('m', 'n', 'wavelength', 'int_mid_MJD_UTC', 'int_mid_BJD_TDB', 'flux(MJy/sr)', 'err(MJr/sr)'))#, 'filename'))
    ascii.write(table, filename, format='ecsv', overwrite=True)



    #arr = [event.wave, event.mjdutc, event.bjdtdb, event.stdspec, event.stdvar, np.zeros(len(event.stdvar)), np.ones(len(event.stdvar))]
    #table = QTable(arr, names=('wavelength', 'int_mid_MJD_UTC', 'int_mid_BJD_TDB', 'flux(MJy/sr)', 'err(MJr/sr)', 'posx', 'posy'))
    #ascii.write(table, event.dirname + '/S3_' + event.eventlabel + "_spec_lc.txt", format='ecsv', overwrite=True)
