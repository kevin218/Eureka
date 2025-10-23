.. _outputs:

Eureka! Outputs
===============

Stage 2 through Stage 6 of ``Eureka!`` can be configured to output plots of the pipeline's interim results as well as the data required to run further stages.


.. _s2-out:

Stage 2 Outputs
---------------

If ``skip_extract_1d`` is set in the Stage 2 ECF, the 1-dimensional spectrum will not be extracted, and no plots will be made. Otherwise, Stage 2 will extract the 1-dimensional spectrum from the calibrated images, and will plot the spectrum.

.. figure:: ../media/S2_out/fig2101_file1_x1dints.png
   :alt: Stage 2 1-dimensional spectrum plot

   Fig 2101: 1-Dimensional Spectrum Plot


.. _s3-out:

Stage 3 Outputs
---------------

In Stage 3 through Stage 5, output plots are controlled with the ``isplots_SX`` parameter. The resulting plots are cumulative: setting ``isplots_S3 = 5`` will also create the plots specified in ``isplots_S3 = 3`` and ``isplots_S3 = 1``.

In Stage 3:
   - If ``isplots_S3`` = 1: ``Eureka!`` will plot the 2-dimensional, non-drift-corrected light curve, as well as variations in the source position and width on the detector.

   .. figure:: ../media/S3_out/fig3101-2D_LC.png
      :alt: Stage 3 2-dimensional spectrum plot

      Fig 3101: 2-Dimensional Spectrum Plot with a linear wavelength x-axis

   .. figure:: ../media/S3_out/fig3102-2D_LC.png
      :alt: Stage 3 2-dimensional spectrum plot

      Fig 3102: 2-Dimensional Spectrum Plot with a linear detector pixel x-axis

   .. figure:: ../media/S3_out/fig3103_file0_int00_source_pos.png
      :alt: Stage 3 source position fit

      Fig 3103: Source Position Fit Plot

   .. figure:: ../media/S3_out/fig3104_DriftYPos.png
      :alt: Stage 3 y drift

      Fig 3104: Variations in the spatial-axis position

   .. figure:: ../media/S3_out/fig3105_DriftYWidth.png
      :alt: Stage 3 y PSF width changes

      Fig 3105: Variations in the spatial-axis PSF width

   .. figure:: ../media/S3_out/fig3106_Drift2D.png
      :alt: Stage 3 2D drift fit

      Fig 3106: 2D drift fit (currently only produced for WFC3)

   .. figure:: ../media/S3_out/fig3107_file0_Curvature.png
      :alt: Stage 3 trace curvature

      Fig 3107: Measured, smoothed, and integer-rounded position of trace

   - If ``isplots_S3`` = 3: ``Eureka!`` will plot the results of the background and optimal spectral extraction steps for each exposure in the observation as well as the cleaned median frame.

   .. figure:: ../media/S3_out/fig3301_file0_int001_ImageAndBackground.png
      :alt: Stage 3 background subtracted flux plot

      Fig 3301: Background Subtracted Flux Plot

   .. figure:: ../media/S3_out/fig3302_file0_int001_Spectrum.png
      :alt: Stage 3 1-dimensional spectrum plot

      Fig 3302: 1-Dimensional Spectrum Plot

   .. figure:: ../media/S3_out/fig3303_file0_int001_Profile.png
      :alt: Stage 3 weighting profile

      Fig 3303: Weighting Profile Plot

   .. figure:: ../media/S3_out/fig3304_file0_ResidualBG.png
      :alt: Stage 3 residual background

      Fig 3304: Residual Background Plot

   .. figure:: ../media/S3_out/fig3308_file0_MedianFrame.png
      :alt: Stage 3 clean median frame plot

      Fig 3308: Clean Median Frame Plot

   - If ``isplots_S3`` = 5: ``Eureka!`` will plot the Subdata plots from the optimal spectral extraction step.

   .. figure:: ../media/S3_out/fig3501_file0_int001_col0117_subdata.png
      :alt: Stage 3 subdata plot

      Fig 3501: Spectral Extraction Subdata Plot

   .. figure:: ../media/S3_out/fig3507a_file0_int011_tilt_events.png
      :alt: Stage 3 tilt event plots

      Fig 3507a: Tilt event frame plot. Figures 3507b and 3507c are GIF versions of this figure, b is at the segment level and c is over all segments

.. _s4-out:

Stage 4 Outputs
---------------

In Stage 4:
   - If ``isplots_S4`` = 1: ``Eureka!`` will plot the spectral drift per exposure, the drift-corrected 2-dimensional lightcurve with extracted bins overlaid, and the 1D light curves.

   .. figure:: ../media/S4_out/fig4101_2D_LC.png
      :alt: Stage 4 2-dimensional spectrum

      Fig 4101: 2-Dimensional Spectrum with a linear wavelength x-axis.

   .. figure:: ../media/S4_out/fig4102_ch0_1D_LC.png
      :alt: Stage 4 1-dimensional binned spectrum

      Fig 4102: 1-Dimensional Binned Spectrum

   .. figure:: ../media/S4_out/fig4103_DriftXPos.png
      :alt: Stage 4 spectral drift plot

      Fig 4103: Spectral Drift Plot

   .. figure:: ../media/S4_out/fig4105_ch00_1D_BG.png
      :alt: Stage 4 background flux plot

      Fig 4105: Background Flux Plot

   .. figure:: ../media/S4_out/fig4106_MAD_Outliers.png
      :alt: Stage 4 MAD outliers plot

      Fig 4106: MAD Outliers Plot

   - If ``isplots_S4`` = 3: ``Eureka!`` will plot the cross-correlated reference spectrum with the current spectrum for each integration, and the cross-correlation strength for each integration.

   .. figure:: ../media/S4_out/fig4301_int00_CC_Spec.png
      :alt: Stage 4 cross correlated reference spectrum

      Fig 4301: Cross-Correlated Reference Spectrum

   .. figure:: ../media/S4_out/fig4302_int00_CC_Vals.png
      :alt: Stage 4 cross correlation strength

      Fig 4302: Cross-Correlation Strength


.. _s5-out:

Stage 5 Outputs
---------------

In Stage 5:
   - If ``isplots_S5`` = 1: ``Eureka!`` will plot the fitted lightcurve model over the data in each channel. If fitting with a GP, an additional figure will be made showing the GP component. If fitting a sinusoid_pc model, another zoomed-in figure with binned data will be made to emphasize the phase variations. Finally, an additional plot compares the fits from different fitters.

   .. figure:: ../media/S5_out/fig5101_ch0_lc_emcee.png
      :alt: Stage 5 fit data and lightcurve

      Fig 5101: Fitted Lightcurve, Model, and Residual Plot

   .. figure:: ../media/S5_out/fig5102_ch0_lc_GP_emcee.png
      :alt: Stage 5 GP plot

      Fig 5102: Fitted Lightcurve, GP Model, and Residual Plot

   .. figure:: ../media/S5_out/fig5103_ch0_all_fits.png
      :alt: Stage 5 All fits comparison

      Fig 5103: Comparison of Different Fitters

   Fig 5104: *(Demo figure to come)* Zoomed-in Figure Emphasizing Phase Variations Using Temporally Binned Data.

   - If ``isplots_S5`` = 3: ``Eureka!`` will plot an RMS deviation plot for each channel to help check for correlated noise, plot the normalized residual distribution, and plot the fitting chains for each channel. If fitting a sinusoid_pc model, another zoomed-in figure with binned data in front of the unbinned data will be made to emphasize the phase variations.

   .. figure:: ../media/S5_out/fig5301_ch0_RMS_TimeAveraging_emcee.png
      :alt: Stage 5 RMS time-averaging plot

      Fig 5301: RMS Deviation Plot

   .. figure:: ../media/S5_out/fig5302_ch0_res_distri_emcee.png
      :alt: Stage 5 residual distribution

      Fig 5302: Residual Distribution

   .. figure:: ../media/S5_out/fig5303_ch0_burninchain.png
      :alt: Stage 5 fitting chains

   .. figure:: ../media/S5_out/fig5303_ch0_chain.png
      :alt: Stage 5 fitting chains

      Figs 5303: Fitting Chains. Only made for ``emcee`` runs. Two version of the plot will be saved, one including the burn in steps and one without the burn in steps.

   Fig 5304: *(Demo figure to come)* Zoomed-in Figure Emphasizing Phase Variations Using Temporally Binned Data Over Unbinned Data.

   .. figure:: ../media/S5_out/fig5309_ch00_harmonica_string_emcee.png
      :alt: Stage 5 Harmonica transmission string

   Fig 5309: Harmonica transmission string. The blue solid line depicts the measured shape of the planet.  Any deviation from the reference circle (gray dashed line) points to a non-spherical planet.  The morning limb is on the right; the evening limb is on the left.

   - If ``isplots_S5`` = 5, and if ``emcee`` or ``dynesty`` were used as the fitter: ``Eureka!`` will plot a corner plot for each channel.

   .. figure:: ../media/S5_out/fig5501_ch0_corner_emcee.png
      :alt: Stage 5 corner plot

      Fig 5501: Corner Plot


.. _s6-out:

Stage 6 Outputs
---------------

In Stage 6:
   - If ``isplots_S6`` = 1: ``Eureka!`` will plot the transmission or emission spectrum, depending
     on the setting of ``y_unit``. If a model is provided, it will be plotted on the same figure
     along with points binned from that model to the resolution of the data.

   .. figure:: ../media/S6_out/fig6101_transmission.png
      :alt: Stage 6 transmission spectrum.

      Fig 6101: Transmission Spectrum.

   .. figure:: ../media/S6_out/fig6101_emission.png
      :alt: Stage 6 emission spectrum.

      Fig 6101: Emission Spectrum.

   - If ``isplots_S6`` = 3: ``Eureka!`` will make another transmission plot (if ``y_unit`` is
     transmission type) with a second y-axis which is in units of atmospheric scale height.

   .. figure:: ../media/S6_out/fig6301_transmission.png
      :alt: Stage 6 transmission spectrum with a second y-axis in units of atmospheric scale height.

      Fig 6301: Transmission Spectrum with Double y-axis.
