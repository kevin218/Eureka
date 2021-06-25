## Determine source position for data where it's not in the header (MIRI)

#Last edit Megan Mansfield 6/24/21

import numpy as np

## Determine source position for data where it's not in the header (MIRI)

# Last edit Megan Mansfield 6/24/21

import numpy as np


def source_pos(data, meta):
	'''
    A simple function to find the brightest row for source location
    Parameters
    ----------
    data              : data object
    meta              : metadata object
    Returns
    -------
    y_pos             : The central position of the star
    History
    -------
    Written by Megan Mansfield          6/24/21
    '''

	subdata = data[0, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
	y_pos = np.argmax(np.sum(subdata, axis=1))

	return y_pos


def source_pos_FWM(data, meta):
	'''
    An alternative function to find the source location using a flux-weighted mean approach
    Parameters
    ----------
    data              : data object
    meta              : metadata object
    Returns
    -------
    y_pos             : The central position of the star
    History
    -------
    Written by Taylor Bell          2021-06-24
    '''

	subdata = data[0, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
	y_pixels = np.arange(meta.ywindow[0], meta.ywindow[1]).reshape(-1, 1)
	y_pos = np.sum(np.sum(subdata, axis=1) * y_pixels) / np.sum(y_pixels)

	return y_pos
