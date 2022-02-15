import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(meta, wavelength, spectrum, err, ylabel='$R_{\rm p}/R_{\rm *}$ (%)', xlabel='Wavelength ($\mu$m)'):

	fig = plt.figure('6101', figsize=(8, 6))
	plt.clf()
	ax = fig.subplots(1,1)

	ax.errorbar(wavelength, spectrum, yerr=err, color='k')
	ax.ylabel(ylabel)
	ax.xlabel(xlabel)

	fname = 'figs/fig6101.png'
	fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
	if meta.hide_plots:
		plt.close()
	else:
		plt.pause(0.2)

	return