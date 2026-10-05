"""Definitions of the approved Stage 5 regression cases."""
from dataclasses import dataclass, field

TABLE_CORE_COLUMNS = ("time", "wavelength", "bin_width", "lcdata", "lcerr")
TABLE_FINAL_COLUMNS = ("model", "residuals")


@dataclass(frozen=True)
class S5RegressionCase:
    """Inputs and expected products for one deterministic S5 configuration."""

    name: str
    eventlabel: str
    ecf_dir: str
    ecf_filename: str
    fitparams_filename: str
    table_filename: str
    free_parameters: frozenset[str]
    component_columns: tuple[str, ...]
    parameter_atol: dict[str, float] = field(default_factory=dict)
    table_atol: dict[str, float] = field(default_factory=dict)

    @property
    def table_columns(self):
        """Return the complete persisted Table_Save schema for this case."""
        return (TABLE_CORE_COLUMNS + self.component_columns +
                TABLE_FINAL_COLUMNS)


# Parameter names are a set because CSV row order is an implementation detail.
# Values are always compared by parameter name.
CASES = (
    S5RegressionCase(
        name="nircam_spectroscopy",
        eventlabel="NIRCam",
        ecf_dir="tests/NIRCam_ecfs",
        ecf_filename="S5_NIRCam_regression.ecf",
        fitparams_filename="S5_lsq_fitparams_ch0.csv",
        table_filename="S5_NIRCam_ap8_bg12_Table_Save_ch0.txt",
        free_parameters=frozenset((
            "rp", "per", "t0", "inc", "ars", "c0", "A", "m",
            "scatter_mult",
        )),
        component_columns=("polynomial", "GP", "astrophysical model"),
        parameter_atol={"t0": 1e-6, "inc": 1e-6},
        # GP predictions approach zero, where relative error is not a useful
        # measure of cross-platform floating-point roundoff.
        table_atol={"GP": 1e-14},
    ),
    S5RegressionCase(
        name="nircam_photometry",
        eventlabel="Photometry_NIRCam",
        ecf_dir="tests/Photometry_NIRCam_ecfs",
        ecf_filename="S5_Photometry_NIRCam.ecf",
        fitparams_filename="S5_lsq_fitparams_ch0.csv",
        table_filename=("S5_Photometry_NIRCam_ap60_bg70_90_"
                        "Table_Save_ch0.txt"),
        free_parameters=frozenset((
            "rp", "per", "t0", "inc", "a", "u1", "u2", "c0",
            "ypos", "xpos",
        )),
        component_columns=(
            "polynomial", "xpos", "ypos", "astrophysical model",
        ),
        parameter_atol={"t0": 1e-6, "inc": 1e-6, "xpos": 1e-6,
                        "ypos": 1e-6},
    ),
    S5RegressionCase(
        name="nirspec_spectroscopy",
        eventlabel="NIRSpec",
        ecf_dir="tests/NIRSpec_ecfs",
        ecf_filename="S5_NIRSpec.ecf",
        fitparams_filename="S5_lsq_fitparams_shared.csv",
        table_filename="S5_NIRSpec_ap5_bg10_Table_Save_shared.txt",
        free_parameters=frozenset((
            "rp", "rp_ch1", "per", "t0", "inc", "a", "c0",
            "c0_ch1", "scatter_mult", "scatter_mult_ch1",
        )),
        component_columns=("polynomial", "astrophysical model"),
        parameter_atol={"t0": 1e-6, "inc": 1e-6},
    ),
    S5RegressionCase(
        name="miri_spectroscopy",
        eventlabel="MIRI",
        ecf_dir="tests/MIRI_ecfs/POET",
        ecf_filename="S5_MIRI.ecf",
        fitparams_filename="S5_lsq_fitparams_ch0.csv",
        table_filename="S5_MIRI_ap4_bg10_Table_Save_ch0.txt",
        free_parameters=frozenset((
            "rprs", "fpfs", "cos1_amp", "cos1_off", "c0", "r0",
            "r1", "scatter_mult",
        )),
        component_columns=(
            "polynomial", "exp. ramp", "astrophysical model",
        ),
        # Powell converges slightly short of these bounded phase-curve
        # parameters on some numerical stacks. Both remain near their EPF
        # upper bounds of 1 and 20 degrees, respectively.
        parameter_atol={"cos1_amp": 2e-5, "cos1_off": 3e-3},
    ),
)
