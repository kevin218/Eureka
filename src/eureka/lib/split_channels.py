import numpy as np


def get_trim(nints, channel):
    """Get the slicing indices that should be used to get the values
    corresponding to channel.

    Parameters
    ----------
    nints : list
        The number of integrations for each channel
    channel : int
        The channel currently being worked on.

    Returns
    -------
    trim1 : int
        The first index to use when slicing an array.
    trim2 : int
        The second index to use when slicing an array.
    """
    if len(nints) == 1:
        trim1 = 0
        trim2 = nints[0]
    else:
        trim1 = int(np.nansum(nints[:channel]))
        trim2 = trim1 + int(nints[channel])
    return trim1, trim2


def split(arrays, nints, channel):
    """Split a collection of arrays to get the values for just one specific
    channel.

    Parameters
    ----------
    arrays : list
        A list of arrays that should be split.
    nints : list
        The number of integrations for each channel
    channel : int
        The channel currently being worked on.

    Returns
    -------
    list
        The same input arrays, only containing the values corresponding to
        channel.
    """
    trim1, trim2 = get_trim(nints, channel)
    arrays_return = []
    for array in arrays:
        arrays_return.append(array[trim1:trim2])

    return arrays_return
