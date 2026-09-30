"""Science-product regression tests for Eureka Stage 5."""
from pathlib import Path
from shutil import copy2

from astropy.table import Table
import numpy as np
import pytest

from .cases import CASES
from .conftest import REFERENCE_ROOT

FITPARAM_RTOL = 1e-5
TABLE_AXIS_RTOL = 1e-9
TABLE_VALUE_RTOL = 1e-4


def _reference_paths(case):
    """Return the path to the golden reference S5 files for one case."""
    reference_dir = REFERENCE_ROOT / case.name
    return (reference_dir / case.fitparams_filename,
            reference_dir / case.table_filename)


def _actual_paths(case, meta):
    """Return the path to the files created by this regression test case."""
    output_dir = Path(meta.outputdir)
    return (output_dir / case.fitparams_filename,
            output_dir / case.table_filename)


def _assert_expected_outputs(case, meta, actual_fitparams, actual_table):
    """Assert that S5 wrote exactly the configured LSQ science products."""
    output_dir = Path(meta.outputdir)
    assert actual_fitparams.is_file(), (
        f"{case.name}: missing fitparams output {actual_fitparams.name}"
    )
    assert actual_table.is_file(), (
        f"{case.name}: missing Table_Save output {actual_table.name}"
    )
    fitparam_outputs = sorted(
        path.name for path in output_dir.glob("S5_*fitparams*.csv")
    )
    table_outputs = sorted(
        path.name for path in output_dir.glob("S5_*Table_Save*.txt")
    )
    assert fitparam_outputs == [case.fitparams_filename], (
        f"{case.name}: unexpected fitter fitparams outputs"
    )
    assert table_outputs == [case.table_filename], (
        f"{case.name}: unexpected Table_Save outputs"
    )


def _read_fitparams(path):
    """Read and validate the two-column LSQ fit-parameter CSV schema."""
    table = Table.read(path, format="ascii.csv")
    assert table.colnames == ["Parameter", "Mean"], (
        f"{path}: unexpected LSQ fitparams schema {table.colnames}"
    )
    return table


def _parameter_values(table):
    """Return finite LSQ means keyed by their unique parameter names."""
    names = list(table["Parameter"])
    assert len(names) == len(set(names)), "fitparams contains duplicate names"
    values = np.asarray(table["Mean"], dtype=float)
    assert np.all(np.isfinite(values)), "fitparams contains non-finite means"
    return dict(zip(names, values, strict=True))


def _assert_fitparams(case, actual_path, reference_path):
    """Compare one case's fitted parameter names and LSQ means."""
    actual = _read_fitparams(actual_path)
    expected = _read_fitparams(reference_path)
    actual_values = _parameter_values(actual)
    expected_values = _parameter_values(expected)
    assert set(actual_values) == case.free_parameters, (
        f"{case.name}: fitted parameter names differ from the case manifest"
    )
    assert set(expected_values) == case.free_parameters, (
        f"{case.name}: reference parameter names differ from the case manifest"
    )

    for name in case.free_parameters:
        np.testing.assert_allclose(
            actual_values[name], expected_values[name], rtol=FITPARAM_RTOL,
            atol=case.parameter_atol.get(name, 0),
            err_msg=f"{case.name}: fitparams.{name}",
        )


def _read_table(path):
    """Read one native S5 Table_Save ECSV file."""
    return Table.read(path, format="ascii.ecsv")


def _assert_table(case, actual_path, reference_path):
    """Compare one case's saved light-curve table and model components."""
    actual = _read_table(actual_path)
    expected = _read_table(reference_path)
    assert tuple(actual.colnames) == case.table_columns, (
        f"{case.name}: unexpected Table_Save schema {actual.colnames}"
    )
    assert tuple(expected.colnames) == case.table_columns, (
        f"{case.name}: reference Table_Save schema {expected.colnames}"
    )
    assert len(actual) == len(expected), (
        f"{case.name}: Table_Save row count changed from {len(expected)} "
        f"to {len(actual)}"
    )

    axis_columns = {"time", "wavelength", "bin_width"}
    for column in case.table_columns:
        rtol = TABLE_AXIS_RTOL if column in axis_columns else TABLE_VALUE_RTOL
        np.testing.assert_allclose(
            np.asarray(actual[column], dtype=float),
            np.asarray(expected[column], dtype=float), rtol=rtol, atol=0,
            equal_nan=True, err_msg=f"{case.name}: Table_Save.{column}",
        )


def _overwrite_references(case, actual_fitparams, actual_table,
                          reference_fitparams, reference_table):
    """Replace one selected case's approved native S5 products."""
    reference_fitparams.parent.mkdir(parents=True, exist_ok=True)
    copy2(actual_fitparams, reference_fitparams)
    copy2(actual_table, reference_table)
    print(f"Updated Stage 5 references for {case.name}: "
          f"{reference_fitparams.parent}")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_s5_science_products(case, run_s5, overwrite_ref_files):
    """Stage 5 outputs must match their approved instrument/mode baseline."""
    meta = run_s5(case)
    actual_fitparams, actual_table = _actual_paths(case, meta)
    _assert_expected_outputs(case, meta, actual_fitparams, actual_table)
    reference_fitparams, reference_table = _reference_paths(case)

    if overwrite_ref_files:
        _overwrite_references(case, actual_fitparams, actual_table,
                              reference_fitparams, reference_table)
        return

    assert reference_fitparams.is_file(), (
        f"{case.name}: missing reference {reference_fitparams}"
    )
    assert reference_table.is_file(), (
        f"{case.name}: missing reference {reference_table}"
    )
    _assert_fitparams(case, actual_fitparams, reference_fitparams)
    _assert_table(case, actual_table, reference_table)
