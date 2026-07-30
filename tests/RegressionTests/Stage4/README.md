# Stage 4 regression tests

Each regression test case runs only Eureka Stage 4. The Stage 3
`SpecData.h5` reference files for the corresponding case is used as the input file. We rename this file to a different temporary filename so Eureka can follow its normal standalone Stage 4 metadata-loading path.

Each Stage 4 reference directory contains:

- `SpecData.h5`: the approved corrected spectrum;
- `LCData.h5`: the approved binned light curves; and
- `metadata.json`: quality metrics and bin definitions that are not all
  available in a saved Stage 4 science product.

Run the suite with:

```bash
pytest tests/RegressionTests/Stage4
```

References are generated only from the existing full-pipeline integration
tests.
