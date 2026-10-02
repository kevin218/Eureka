# Stage 5 regression tests

Each case runs only Eureka Stage 5. It starts from the corresponding approved
Stage 4 `SpecData.h5` and `LCData.h5` references and runs S5 in pytest's
temporary workspace. The fixture reconstructs predecessor metadata from
`SpecData.h5` and reads the S4 channel definitions from `LCData.h5`.

The current cases are four deterministic LSQ configurations (NIRCam
spectroscopy and photometry, NIRSpec spectroscopy, and MIRI POET
spectroscopy), plus seeded NIRCam ``emcee`` and MIRI ``dynesty`` sampler
configurations. The dynesty case is a MIRI POET configuration, matching the
existing MIRI LSQ regression model. NIRISS and WFC3 do not currently have
supported S5 configurations and are intentionally not covered here.

Each `references/<case>/` directory contains the native S5 output files:

- `S5_lsq_fitparams_*.csv`: fitted free-parameter names and LSQ means;
- `S5_emcee_fitparams_*.csv` and `S5_dynesty_fitparams_*.csv`: fitted
  free-parameter names with mean, uncertainty, and posterior-quantile
  summaries;
- `S5_*_Table_Save_*.txt`: the final saved light curve, model components,
  composite model, and residuals.

The test checks expected output tags/files and schemas exactly. It compares
parameter values by name, uses per-parameter absolute tolerances where scale
requires them, compares time/wavelength axes near-exactly, and compares the
remaining science values with relative tolerance. The sampler chains (`.h5`)
are not golden references; the final saved sampler summaries are. Figures and
transient fitting objects are also outside the regression contract.
The NIRCam ``emcee`` case also writes an LSQ-initialization CSV; its expected
presence is checked, but it is not a reference product because the final
``emcee`` CSV is the regression target.

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
