# Stage 4 regression tests

Each regression test case runs only Eureka Stage 4. The Stage 3
`SpecData.h5` reference files for the corresponding case is used as the input file. We rename this file to a different temporary filename so Eureka can follow its normal standalone Stage 4 metadata-loading path.

Each Stage 4 reference directory contains:

- `SpecData.h5`: the approved corrected spectrum;
- `LCData.h5`: the approved binned light curves.

`SpecData.h5` also stores the unbinned Stage 4 MAD and applied detector-column
mask. `LCData.h5` stores the binned science products, their wavelength-bin
coordinates, and the binned MAD metrics.

Run the suite with:

```bash
pytest tests/RegressionTests/Stage4
```

References are updated manually and intentionally. To update selected cases
directly from the regression suite, run:

```bash
pytest -s -q tests/RegressionTests/Stage4 -k nircam_spectroscopy \
  --overwrite-ref-files
```

Pytest will display a warning and require you to type `OVERWRITE`. The command
replaces both tracked HDF5 products for every selected case. Review the resulting
Git diff before committing.
