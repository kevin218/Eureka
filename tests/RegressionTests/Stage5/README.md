# Stage 5 regression tests

This suite has one focused, deterministic LSQ regression case: two NIRSpec
wavelength channels independently fit with a linear polynomial baseline. It
starts from the approved NIRSpec Stage 4 `SpecData.h5` and `LCData.h5`
references, reconstructs previous metadata, and runs only Stage 5 in
pytest's temporary workspace.

The regression-only ECF selects `run_myfuncs [polynomial]`. For each channel,
the EPF fits only these independent coefficients:

- `c0`: the constant normalized-flux baseline;
- `c1`: the linear trend with time.

The test compares each channel's saved LSQ parameter CSV and `Table_Save`
ECSV product. It checks the fitted parameter names (within the CSV), time and wavelength axes, input light-curve data and uncertainties, the polynomial component, the unity-valued astrophysical component that S5 saves, the final model, and
residuals (all within the ECSV). Array comparisons use the shared `regression_helpers.assert_array` utility, so failure reporting and tolerance behavior are consistent with the Stage 3 and Stage 4 suites.

The short instrument fixtures are not sufficient to validate transit,
eclipse, phase-curve, GP, or other model recovery. Those behaviors are tested within unit tests; this regression test verifies the two-channel linear-baseline S5 path and its saved science products.

Run the suite with:

```bash
pytest tests/RegressionTests/Stage5
```

To intentionally create or replace this case's four native reference files,
run:

```bash
pytest -s -q tests/RegressionTests/Stage5 \\
  -k nirspec_linear_baselines --overwrite-ref-files
```

Pytest displays a warning and requires `OVERWRITE` before it changes
references. Normal runs never overwrite them.
