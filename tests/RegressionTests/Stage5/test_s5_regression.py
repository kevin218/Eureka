"""Science-product regression tests for Eureka Stage 5."""
from pathlib import Path
from shutil import copy2

import numpy as np
import pytest
from astropy.table import Table

from .cases import CASES
from .conftest import REFERENCE_ROOT

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
    """Assert that S5 wrote the configured final and auxiliary products."""
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
    expected_fitparams = sorted((case.fitparams_filename,) +
                                case.auxiliary_fitparams_filenames)
    assert fitparam_outputs == expected_fitparams, (
        f"{case.name}: unexpected fitter fitparams outputs"
    )
    assert table_outputs == [case.table_filename], (
        f"{case.name}: unexpected Table_Save outputs"
    )


def _read_fitparams(path, columns):
    """Read one final fitter summary CSV using its configured schema."""
    table = Table.read(path, format="ascii.csv")
    assert tuple(table.colnames) == columns, (
        f"{path}: unexpected fitparams schema {table.colnames}"
    )
    return table


def _parameter_values(table, columns):
    """Return finite fitter-summary values keyed by unique parameter names."""
    names = list(table["Parameter"])
    assert len(names) == len(set(names)), "fitparams contains duplicate names"
    values = {}
    for column in columns[1:]:
        summary = np.asarray(table[column], dtype=float)
        assert np.all(np.isfinite(summary)), (
            f"fitparams contains non-finite {column} values"
        )
        values[column] = dict(zip(names, summary, strict=True))
    return values


def _assert_fitparams(case, actual_path, reference_path):
    """Compare final named fitter summaries for one regression case."""
    actual = _read_fitparams(actual_path, case.fitparams_columns)
    expected = _read_fitparams(reference_path, case.fitparams_columns)
    actual_values = _parameter_values(actual, case.fitparams_columns)
    expected_values = _parameter_values(expected, case.fitparams_columns)
    assert set(actual_values["Mean"]) == case.free_parameters, (
        f"{case.name}: fitted parameter names differ from the case manifest"
    )
    assert set(expected_values["Mean"]) == case.free_parameters, (
        f"{case.name}: reference parameter names differ from the case manifest"
    )

    for column in case.fitparams_columns[1:]:
        for name in case.free_parameters:
            np.testing.assert_allclose(
                actual_values[column][name], expected_values[column][name],
                rtol=case.fitparams_rtol,
                atol=case.parameter_atol.get(name, 0),
                err_msg=f"{case.name}: fitparams.{column}.{name}",
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
            np.asarray(expected[column], dtype=float), rtol=rtol,
            atol=case.table_atol.get(column, 0),
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
