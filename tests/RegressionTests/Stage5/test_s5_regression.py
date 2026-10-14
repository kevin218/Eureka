"""Science-product regression tests for Eureka Stage 5."""
from pathlib import Path
from shutil import copy2

import numpy as np
import pytest
from astropy.table import Table

from ..regression_helpers import assert_array
from .cases import CASES
from .conftest import REFERENCE_ROOT

FITPARAM_RTOL = 1e-5
TABLE_AXIS_RTOL = 1e-9
TABLE_VALUE_RTOL = 1e-4


def _reference_paths(case):
    """Return golden fit-parameter and Table_Save paths for one case."""
    reference_dir = REFERENCE_ROOT / case.name
    fitparams = tuple(
        reference_dir / name for name in case.fitparams_filenames
    )
    tables = tuple(reference_dir / name for name in case.table_filenames)
    return fitparams, tables


def _actual_paths(case, meta):
    """Return the paths written by one Stage 5 regression run."""
    output_dir = Path(meta.outputdir)
    fitparams = tuple(output_dir / name for name in case.fitparams_filenames)
    tables = tuple(output_dir / name for name in case.table_filenames)
    return fitparams, tables


def _assert_expected_outputs(case, meta, actual_fitparams, actual_tables):
    """Assert that S5 wrote exactly the configured LSQ science products."""
    for path in actual_fitparams + actual_tables:
        assert path.is_file(), f"{case.name}: missing output {path.name}"

    output_dir = Path(meta.outputdir)
    fitparam_outputs = sorted(
        path.name for path in output_dir.glob("S5_*fitparams*.csv")
    )
    table_outputs = sorted(
        path.name for path in output_dir.glob("S5_*Table_Save*.txt")
    )
    assert fitparam_outputs == sorted(case.fitparams_filenames), (
        f"{case.name}: unexpected fitter fitparams outputs"
    )
    assert table_outputs == sorted(case.table_filenames), (
        f"{case.name}: unexpected Table_Save outputs"
    )


def _parameter_values(path):
    """Return finite LSQ means keyed by their unique parameter names."""
    table = Table.read(path, format="ascii.csv")
    assert table.colnames == ["Parameter", "Mean"], (
        f"{path}: unexpected LSQ fitparams schema {table.colnames}"
    )
    names = list(table["Parameter"])
    assert len(names) == len(set(names)), f"{path}: duplicate parameter names"
    values = np.asarray(table["Mean"], dtype=float)
    assert np.all(np.isfinite(values)), f"{path}: non-finite parameter means"
    return dict(zip(names, values, strict=True))


def _assert_fitparams(case, actual_path, reference_path):
    """Compare one channel's fitted parameter names and LSQ means."""
    actual_values = _parameter_values(actual_path)
    expected_values = _parameter_values(reference_path)
    assert set(actual_values) == case.free_parameters, (
        f"{case.name}: fitted parameter names differ from the case manifest"
    )
    assert set(expected_values) == case.free_parameters, (
        f"{case.name}: reference parameter names differ from the case manifest"
    )
    for name in sorted(case.free_parameters):
        assert_array(case, f"fitparams.{name}",
                     np.asarray(actual_values[name]),
                     np.asarray(expected_values[name]),
                     product=actual_path.name, rtol=FITPARAM_RTOL)


def _assert_table(case, actual_path, reference_path):
    """Compare one channel's saved light curve and model components."""
    actual = Table.read(actual_path, format="ascii.ecsv")
    expected = Table.read(reference_path, format="ascii.ecsv")
    assert tuple(actual.colnames) == case.table_columns, (
        f"{case.name}: unexpected Table_Save schema {actual.colnames}"
    )
    assert tuple(expected.colnames) == case.table_columns, (
        f"{case.name}: reference Table_Save schema {expected.colnames}"
    )
    for column in case.table_columns:
        rtol = TABLE_AXIS_RTOL if column in {
            "time", "wavelength", "bin_width",
        } else TABLE_VALUE_RTOL
        assert_array(case, column, np.asarray(actual[column], dtype=float),
                     np.asarray(expected[column], dtype=float),
                     product=actual_path.name, rtol=rtol)


def _overwrite_references(actual_paths, reference_paths):
    """Replace selected approved native S5 products with current outputs."""
    for actual_path, reference_path in zip(actual_paths, reference_paths,
                                           strict=True):
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        copy2(actual_path, reference_path)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_s5_science_products(case, run_s5, overwrite_ref_files):
    """Stage 5 outputs must match their approved linear-LSQ baseline."""
    meta = run_s5(case)
    actual_fitparams, actual_tables = _actual_paths(case, meta)
    _assert_expected_outputs(case, meta, actual_fitparams, actual_tables)
    reference_fitparams, reference_tables = _reference_paths(case)
    all_actual = actual_fitparams + actual_tables
    all_references = reference_fitparams + reference_tables

    if overwrite_ref_files:
        _overwrite_references(all_actual, all_references)
        return

    for path in all_references:
        assert path.is_file(), f"{case.name}: missing reference {path}"
    for actual_path, reference_path in zip(actual_fitparams,
                                           reference_fitparams, strict=True):
        _assert_fitparams(case, actual_path, reference_path)
    for actual_path, reference_path in zip(actual_tables, reference_tables,
                                           strict=True):
        _assert_table(case, actual_path, reference_path)
