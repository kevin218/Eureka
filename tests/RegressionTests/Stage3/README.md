# Stage 3 regression tests

Each case runs only Eureka Stage 3 and compares its science product against an
approved reference in `references/<case>/`. The suite always forces
`save_output=True` and writes its normal S3 output only to pytest's temporary
workspace.

Each reference directory contains the following S3 reference data:

- `SpecData.h5`: the approved Stage 3 data product, including `maed_s3` in its
  stored metadata attributes.

Run the suite with:

```bash
pytest tests/RegressionTests/Stage3
```

To update references directly from selected regression cases, use:

```bash
pytest -s -q tests/RegressionTests/Stage3 -k miri_spectroscopy \
  --overwrite_ref_files
```

Pytest will display a warning and require you to type `OVERWRITE`. This option
replaces the existing `SpecData.h5` for every selected test case, in this case, MIRI Spectroscopy. You can edit or remove that flag to overwrite a different case or all cases, respectively.
