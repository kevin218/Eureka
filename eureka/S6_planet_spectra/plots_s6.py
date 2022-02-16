import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(meta, wavelength, spectrum, err, wavelength_error=None, ylabel=r'$R_{\rm p}/R_{\rm *}$', xlabel=r'Wavelength ($\mu$m)'):

	fig = plt.figure('6101', figsize=(8, 4))
	plt.clf()
	ax = fig.subplots(1,1)

	plt.errorbar(wavelength, spectrum, fmt='o', capsize=3, xerr=wavelength_error, yerr=err, color='k')
	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)

	fname = 'figs/fig6101.png'
	fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
	if meta.hide_plots:
		plt.close()
	else:
		plt.pause(0.2)

	return