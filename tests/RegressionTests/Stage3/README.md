# Stage 3 regression tests

Each case runs only Eureka Stage 3 and compares its science product against an
approved reference in `references/<case>/`. The suite always forces
`save_output=True` and writes its normal S3 output only to pytest's temporary
workspace.

Each reference directory contains the following S3 reference data:

- `SpecData.h5`: the approved Stage 3 data product, including `mad_s3` in its
  stored metadata attributes.

Run the suite with:

```bash
pytest tests/RegressionTests/Stage3
```

TODO: REMOVE PARAGRAPH ONCE OLDER FULL PIPELINE TESTS ARE REMOVED.
References are updated manually and intentionally. To regenerate, run the
relevant existing integration test with `--keep-s3-output`, review the new
`SpecData.h5`, then replace the corresponding reference file in
the RegressionTests/Stage3/references/ directory.

To update references directly from selected regression cases, use:

```bash
pytest -s -q tests/RegressionTests/Stage3 -k miri_spectroscopy \
  --overwrite_ref_files
```

Pytest will display a warning and require you to type `OVERWRITE`. This option
replaces the existing `SpecData.h5` for every selected test case.

The suite compares selected Stage 3 arrays rather than whole HDF5 files. This
avoids failures from paths, timestamps, and other runtime metadata while still
protecting the science products.
