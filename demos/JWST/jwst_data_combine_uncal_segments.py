import glob
from sys import argv

import numpy as np
from astropy.io import fits


def combine_segments(path_to_uncal_files, verbose=True, testing=True):

	data_filenames = np.sort(glob.glob(path_to_uncal_files+"*/*seg*_mirimage_uncal.fits"))

	if verbose:
		print("Found these uncal files")
		print(data_filenames)

	# open up first fits file
	hdu_base = fits.open(data_filenames[0])

	# update the headers in hdu_base so that it looks like just 1 segment
	hdu_base[0].header['EXSEGTOT'] = 1
	hdu_base[0].header['INTEND']   = hdu_base[0].header['NINTS'] # this value is the total integrations

	# updates all headers and data
	for datafile in data_filenames[1:]:
	    hdu_add = fits.open(datafile)

	    hdu_base[1].header['NAXIS4'] += hdu_add[1].header['NAXIS4']
	    hdu_base[1].data = np.append(hdu_base[1].data, hdu_add[1].data, axis=0)

	    hdu_base[2].header['NAXIS4'] += hdu_add[2].header['NAXIS4']
	    hdu_base[2].data = np.append(hdu_base[2].data, hdu_add[2].data, axis=0)

	    nrows_base = hdu_base[3].data.shape[0]
	    nrows_add  = hdu_add[3].data.shape[0]
	    nrows_new  = nrows_base + nrows_add
	    hdu_new = fits.BinTableHDU.from_columns(hdu_base[3].columns, nrows=nrows_new, header=hdu_base[3].header)
	    for colname in hdu_base[3].columns.names:
	        hdu_new.data[colname][nrows_base:] = hdu_add[3].data[colname]
	    hdu_base[3].data = hdu_new.data

	    nrows_base = hdu_base[4].data.shape[0]
	    nrows_add  = hdu_add[4].data.shape[0]
	    nrows_new  = nrows_base + nrows_add
	    hdu_new = fits.BinTableHDU.from_columns(hdu_base[4].columns, nrows=nrows_new, header=hdu_base[4].header)
	    for colname in hdu_base[4].columns.names:
	        hdu_new.data[colname][nrows_base:] = hdu_add[4].data[colname]
	    hdu_base[4].data = hdu_new.data

	    # really not sure what to do abotu this asdf metadata thing;
	    # going to just take the bigger array? totally unmotivated
	    if hdu_add[5].header['NAXIS1'] > hdu_base[5].header['NAXIS1']:
	        hdu_base[5].header['NAXIS1'] = hdu_add[5].header['NAXIS1']
	        hdu_base[5].data = hdu_add[5].data

	    hdu_add.close()

	# figure out new name that makes sense:
	old_name = data_filenames[0].split('/')[-1][:-5].split('-')
	old_name_base = old_name[0]
	old_name_minus_seg = old_name[-1].split('_')
	new_name = f'{old_name_base}_{old_name_minus_seg[1]}_{old_name_minus_seg[2]}.fits'

	if verbose: print('This will be new uncal fits file name:', new_name)

	# save new fits file
	hdu_base.writeto(path_to_uncal_files+new_name, overwrite=True)
	hdu_base.close()

	if testing:
		hdu_test = fits.open(path_to_uncal_files+new_name)
		print("Shape of combined uncal file:", hdu_test[1].data.shape)

	return new_name

path_to_uncal_files = f"../JWST_data/{argv[1]}/JWST/"
combine_segments(path_to_uncal_files)
