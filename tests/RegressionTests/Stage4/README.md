# Stage 4 regression tests

Each case runs only Eureka Stage 4. Its input is the approved Stage 3
`SpecData.h5` reference for the corresponding case, copied to a temporary
production-style filename so Eureka follows its normal standalone Stage 4
metadata-loading path.

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
tests. Review their S4 products and metadata sidecar before replacing a
reference; the regression suite never updates them automatically.
