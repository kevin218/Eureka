## Determine source position for data where it's not in the header (MIRI)

import numpy as np
import matplotlib.pyplot as plt


def source_pos(data, meta, m, header=False):
	if header:
		src_ypos = data.shdr['SRCYPOS'] - meta.ywindow[0]
	# find the source location using a flux-weighted mean approach
	elif meta.src_pos_type == 'weighted':
		src_ypos = source_pos_FWM(data, meta, m) - meta.ywindow[0]
	# find the source location using a gaussian fit
	elif meta.src_pos_type == 'gaussian':
		src_ypos = source_pos_gauss(data, meta, m) - meta.ywindow[0]
	# brightest row for source location
	else:
		src_ypos = source_pos_max(data, meta, m) - meta.ywindow[0]


	return round(src_ypos)


def source_pos_max(data, meta, m, plot=True):
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
    Modified by Sebastian Zieba		2021-07-14
    '''

	# OLD
	#subdata = data[0, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
	#y_pos = np.argmax(np.sum(subdata, axis=1))

	# NEW
	x_dim = data.shape[1]

	sum_row = np.sum(data[0], axis=1)
	pos_max = np.argmax(sum_row)

	y_pixels = np.arange(0, x_dim)

	# Diagnostic plot
	if meta.isplots_S3 > 0 and plot==True:
		plt.plot(y_pixels, sum_row, label= 'data')
		plt.axvline(pos_max, ls='--', label= 'brightest row', c='r')
		plt.ylabel('row flux')
		plt.xlabel('row pixel position')
		plt.legend()
		plt.tight_layout()
		plt.savefig(meta.workdir + '/figs/fig-file' + str(m+1) + '-source_pos.png')
		# plt.pause(0.1)

	return pos_max


def source_pos_FWM(data, meta, m):
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
    Modified by Sebastian Zieba		2021-07-14
    '''

	# OLD
	#subdata = data[0, meta.ywindow[0]:meta.ywindow[1], meta.xwindow[0]:meta.xwindow[1]]
	#y_pixels = np.arange(meta.ywindow[0], meta.ywindow[1]).reshape(-1, 1)
	#y_pos = np.sum(np.sum(subdata, axis=1) * y_pixels) / np.sum(y_pixels)

	# NEW
	x_dim = data.shape[1]

	pos_max = source_pos_max(data, meta, m, plot=False)

	y_pixels = np.arange(0, x_dim)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]

	sum_row = np.sum(data[0], axis=1)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]
	sum_row -= (sum_row[0]+sum_row[-1])/2

	y_pos = np.sum(sum_row * y_pixels) / np.sum(sum_row)

	# Diagnostic plot
	if meta.isplots_S3 > 0:
		plt.plot(y_pixels, sum_row, label= 'data')
		plt.axvline(pos_max, ls='--', label= 'brightest row', c='r')
		plt.axvline(y_pos, ls='--', label= 'weighted row')
		plt.ylabel('row flux')
		plt.xlabel('row pixel position')
		plt.legend()
		plt.tight_layout()
		plt.savefig(meta.workdir + '/figs/fig-file' + str(m+1) + '-source_pos.png')
		# plt.pause(0.1)

	return y_pos


def source_pos_gauss(data, meta, m):
	'''
    A function to find the source location using a gaussian fit
    Parameters
    ----------
    data              : data object
    meta              : metadata object
    Returns
    -------
    y_pos             : The central position of the star
    History
    -------
    Written by Sebastian Zieba          2021-07-14
    '''

	from scipy.optimize import curve_fit


	x_dim = data.shape[1]

	# Data cutout around the maximum row
	pos_max = source_pos_max(data, meta, m, plot=False)
	x = np.arange(0, x_dim)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]
	y = np.sum(data[0], axis=1)[pos_max-meta.spec_hw:pos_max+meta.spec_hw]

	# Gaussian Function
	def gauss(x, a, x0, sigma, off):
		return a * np.exp(-(x-x0)**2/(2*sigma**2))+off

	# Initial Guesses
	sigma0 = np.sqrt(sum(y * (x - pos_max)**2) / sum(y))
	p0 = [max(y), pos_max, sigma0, np.median(y)]

	# Fit
	popt, pcov = curve_fit(gauss, x, y, p0)

	# Diagnostic plot
	if meta.isplots_S3 > 0:
		plt.plot(x, y, label= 'data')
		plt.plot(np.linspace(0,x_dim,500), gauss(np.linspace(0,x_dim,500), *popt), 'r-', label= 'gaussian fit')
		plt.xlim(pos_max-meta.spec_hw, pos_max+meta.spec_hw)
		plt.axvline(pos_max, ls='--', label= 'brightest row', c='r')
		plt.ylabel('row flux')
		plt.xlabel('row pixel position')
		plt.legend()
		plt.tight_layout()
		plt.savefig(meta.workdir + '/figs/fig-file' + str(m+1) + '-source_pos.png')
		# plt.pause(0.1)

	return popt[1]
