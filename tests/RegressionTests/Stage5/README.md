# Stage 5 regression tests

Each case runs only Eureka Stage 5. It starts from the corresponding approved
Stage 4 `SpecData.h5` and `LCData.h5` references and runs S5 in pytest's
temporary workspace. The fixture reconstructs predecessor metadata from
`SpecData.h5` and reads the S4 channel definitions from `LCData.h5`.

The current deterministic LSQ cases are NIRCam spectroscopy, NIRCam
photometry, NIRSpec spectroscopy, and MIRI POET spectroscopy. NIRISS and WFC3
do not currently have supported S5 configurations and are intentionally not
covered here.

Each `references/<case>/` directory contains the native S5 output files:

- `S5_lsq_fitparams_*.csv`: fitted free-parameter names and LSQ means;
- `S5_*_Table_Save_*.txt`: the final saved light curve, model components,
  composite model, and residuals.

The MIRI POET case also has an `initial_fitparams/` folder. Its parameter set
(phase-curve, ramp, baseline, etc.) has platform-dependent LSQ Powell
search paths from the raw EPF start values. The fitparams in the `initial_fitparams/` folder supply an approved LSQ starting point in the temporary test workspace so the regression test checks the fitted science products rather than Powell's global search path. Basically, this MIRI test no longer tests whether Powell can discover the correct global solution from the raw EPF starting values on every numerical platform. It tests that, starting from an approved valid LSQ solution, the MIRI Stage 5 fitting/model/output path remains correct and does not move to a worse solution.

The test checks expected output tags/files and schemas exactly. It compares
parameter values by name, uses per-parameter absolute tolerances where scale
requires them, compares time/wavelength axes near-exactly, and compares the
remaining science values with relative tolerance. Figures, sampler products,
and transient fitting objects are not regression targets.

Run the suite with:

```bash
pytest tests/RegressionTests/Stage5
```

To create or intentionally replace references for selected cases, run:

```bash
pytest -s -q tests/RegressionTests/Stage5 -k nircam_spectroscopy \
  --overwrite-ref-files
```

Pytest will display a warning and require you to type `OVERWRITE`. The command
updates only the selected cases' (in this case NIRCAM Spectroscopy) S5 reference products. Normal test runs never overwrite references.
