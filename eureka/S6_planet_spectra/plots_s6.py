import numpy as np
import matplotlib.pyplot as plt

def plot_spectrum(meta, wavelength, spectrum, err, wavelength_error=None, model_x=None, model_y=None, ylabel=r'$R_{\rm p}/R_{\rm *}$', xlabel=r'Wavelength ($\mu$m)'):

	fig = plt.figure('6101', figsize=(8, 4))
	plt.clf()
	ax = fig.subplots(1,1)

	ax.errorbar(wavelength, spectrum, fmt='o', capsize=3, ms=3, xerr=wavelength_error, yerr=err, color='k')
	if (model_x is not None) and (model_y is not None):
		in_range = np.logical_and(model_x>=wavelength[0]-wavelength_error[0], model_x<=wavelength[-1]+wavelength_error[-1])
		ax.plot(model_x[in_range], model_y[in_range], color='r', zorder=0)
		if wavelength_error is not None:
			binned_model = []
			for wav, width in zip(wavelength, wavelength_error):
				binned_model.append(np.mean(model_y[np.logical_and(model_x>=wav-width, model_x<wav+width)]))
			ax.plot(wavelength, binned_model, 'o', ms=3, color='r', mec='k', mew=0.2, zorder=0)
		print(spectrum-binned_model)
		print(np.mean(spectrum-binned_model))
	ax.set_ylabel(ylabel)
	ax.set_xlabel(xlabel)

	fname = 'figs/fig6101.png'
	fig.savefig(meta.outputdir+fname, bbox_inches='tight', dpi=300)
	if meta.hide_plots:
		plt.close()
	else:
		plt.pause(0.2)

	return