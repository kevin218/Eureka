import numpy as np
from importlib import reload
from . import sort_nicely as sn
import os, time
import re


def readfiles(meta):
    """
    Reads in the files saved in topdir + datadir and saves them into a list

    Args:
        meta: metadata object

    Returns:
        meta: metadata object but adds segment_list to metadata containing the sorted data fits files
    """

    rootdir = os.path.join(meta.topdir, *meta.datadir.split(os.sep))
    if rootdir[-1]!='/':
      rootdir += '/'

    meta.segment_list = []
    for fname in os.listdir(rootdir):
        if fname.endswith(meta.suffix + '.fits'):
            meta.segment_list.append(rootdir + fname)
    meta.segment_list = sn.sort_nicely(meta.segment_list)
    return meta


def trim(data, meta):
    """
    Removes the edges of the data arrays

    Args:
        dat: Data object
        md: Metadata object

    Returns:
        subdata arrays with trimmed edges depending on xwindow and ywindow which have been set in the S3 ecf
    """
    data.subdata = data.data[:, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    data.suberr  = data.err[:, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    data.subdq   = data.dq[:, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    data.subwave = data.wave[meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    data.subv0   = data.v0[:, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
    meta.subny = meta.ywindow[1] - meta.ywindow[0]
    meta.subnx = meta.xwindow[1] - meta.xwindow[0]

    return data, meta


def check_nans(data, mask, log, name=''):
    """
    Checks where the data array has NaNs

    Args:
        data: a data array (e.g. data, err, dq, ...)
        mask: input mask
        log: log file where NaNs will be mentioned if existent

    Returns:
        mask: output mask where 0 will be written where the input data array has NaNs
    """
    num_nans = np.sum(np.isnan(data))
    if num_nans > 0:
        log.writelog(f"  WARNING: {name} has {num_nans} NaNs.  Your subregion may be off the edge of the detector subarray. Masking NaN region and continuing, but you should really stop and reconsider your choices.")
        inan = np.where(np.isnan(data))
        #subdata[inan]  = 0
        mask[inan]  = 0
    return mask


def makedirectory(meta, stage, **kwargs):
    """
    Creates file directory

    Args:
        meta: metadata object
        stage : 'S#' string denoting stage number (i.e. 'S3', 'S4')
        **kwargs

    Returns:
        run number
    """

    # Create directories for Stage 3 processing
    datetime = time.strftime('%Y-%m-%d')

    # This code allows the input and output files to be stored outside of the Eureka! folder
    rootdir = os.path.join(meta.topdir, *meta.outputdir.split(os.sep))
    if rootdir[-1]!='/':
      rootdir += '/'

    workdir = rootdir + stage + '_' + datetime + '_' + meta.eventlabel +'_'

    for key, value in kwargs.items():

        workdir += key+str(value)+'_'

    workdir += 'run'

    counter=1

    while os.path.exists(workdir+str(counter)):
        counter += 1

    meta.workdir = workdir+str(counter)+'/'
    if not os.path.exists(meta.workdir):
        os.makedirs(meta.workdir)
    if not os.path.exists(meta.workdir + "figs"):
        os.makedirs(meta.workdir + "figs")

    return counter

def pathdirectory(meta, stage, run, **kwargs):
    """
    Creates file directory

    Args:
        meta: metadata object
        stage : 'S#' string denoting stage number (i.e. 'S3', 'S4')
        run : run #, output from makedirectory function
        **kwargs

    Returns:
        directory path for given parameters
    """

    # Create directories for Stage 3 processing
    datetime = time.strftime('%Y-%m-%d')

    # This code allows the input and output files to be stored outside of the Eureka! folder
    rootdir = os.path.join(meta.topdir, *meta.outputdir.split(os.sep))
    if rootdir[-1]!='/':
      rootdir += '/'

    workdir = rootdir + stage + '_' + datetime + '_' + meta.eventlabel +'_'

    for key, value in kwargs.items():

        workdir += key+str(value)+'_'

    workdir += 'run'

    path = workdir+str(run)+'/'

    return path
