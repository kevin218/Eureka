"""Definitions of the approved Stage 5 regression cases."""
from dataclasses import dataclass

TABLE_CORE_COLUMNS = ("time", "wavelength", "bin_width", "lcdata", "lcerr")
TABLE_FINAL_COLUMNS = ("model", "residuals")


@dataclass(frozen=True)
class S5RegressionCase:
    """Inputs and expected products for one deterministic S5 configuration."""

    name: str
    eventlabel: str
    s4_reference_case: str
    ecf_dir: str
    ecf_filename: str
    fitparams_filenames: tuple[str, ...]
    table_filenames: tuple[str, ...]
    free_parameters: frozenset[str]
    component_columns: tuple[str, ...]
    exact_variables: tuple[str, ...] = ()
    atol_variables: tuple[str, ...] = ()

    @property
    def table_columns(self):
        """Return the complete persisted Table_Save schema for this case."""
        return (TABLE_CORE_COLUMNS + self.component_columns +
                TABLE_FINAL_COLUMNS)


CASES = (
    S5RegressionCase(
        name="nirspec_linear_baselines",
        eventlabel="NIRSpec",
        s4_reference_case="nirspec_spectroscopy",
        ecf_dir="tests/NIRSpec_ecfs",
        ecf_filename="S5_NIRSpec_regression.ecf",
        fitparams_filenames=(
            "S5_lsq_fitparams_ch0.csv", "S5_lsq_fitparams_ch1.csv",
        ),
        table_filenames=(
            "S5_NIRSpec_ap5_bg10_Table_Save_ch0.txt",
            "S5_NIRSpec_ap5_bg10_Table_Save_ch1.txt",
        ),
        free_parameters=frozenset(("c0", "c1")),
        component_columns=("polynomial", "astrophysical model"),
    ),
)
