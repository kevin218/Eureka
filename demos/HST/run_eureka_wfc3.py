import os
import eureka.lib.plots
import eureka.S3_data_reduction.s3_reduce as s3
import eureka.S4_generate_lightcurves.s4_genLC as s4
import eureka.S5_lightcurve_fitting.s5_fit as s5
import eureka.S6_planet_spectra.s6_spectra as s6

eventlabel = 'wfc3'
ecf_path = '.'+os.sep

if __name__ == '__main__':
    # To skip one or more stages that were already run,
    # just comment them out below

    spec, meta = s3.reduce(eventlabel, ecf_path=ecf_path)

    spec, lc, meta = s4.genlc(eventlabel, ecf_path=ecf_path)

    meta = s5.fitlc(eventlabel, ecf_path=ecf_path)

    meta, lc = s6.plot_spectra(eventlabel, ecf_path=ecf_path)
