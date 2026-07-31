# Stage 3 regression tests

Each case runs only Eureka Stage 3 and compares its science product against an
approved reference in `references/<case>/`.

Each reference directory contains the following S3 reference data:

- `SpecData.h5`: the approved Stage 3 data product, including `mad_s3` in its
  stored metadata attributes.

Run the suite with:

```bash
pytest tests/RegressionTests/Stage3
```

References are updated manually and intentionally. To regenerate, run the
relevant existing integration test with `--keep-s3-output`, review the new
`SpecData.h5`, then replace the corresponding reference file in
the RegressionTests/Stage3/references/ directory.

The suite compares selected Stage 3 arrays rather than whole HDF5 files. This
avoids failures from paths, timestamps, and other runtime metadata while still
protecting the science products.
