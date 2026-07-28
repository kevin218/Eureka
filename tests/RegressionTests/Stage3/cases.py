"""Definitions of the approved Stage 3 regression cases."""
from dataclasses import dataclass


@dataclass(frozen=True)
class S3RegressionCase:
    """Inputs and expected products for one Stage 3 reduction mode."""

    name: str
    eventlabel: str
    ecf_dir: str
    input_dir: str
    reference_dir: str
    mode: str
    variables: tuple[str, ...]
    exact_variables: tuple[str, ...] = ()
    atol_variables: tuple[str, ...] = ()
    check_niriss_orders: bool = False


SPECTROSCOPY_VARIABLES = (
    "optspec", "opterr", "optmask", "wave_1d", "medflux",
    "skylev", "skyerr",
)

CASES = (
    S3RegressionCase(
        name="nircam_spectroscopy",
        eventlabel="NIRCam",
        ecf_dir="tests/NIRCam_ecfs",
        input_dir="tests/data/JWST-Sim/NIRCam/Stage2",
        reference_dir="nircam_spectroscopy",
        mode="spectroscopy",
        variables=("optspec", "opterr", "optmask", "wave_1d",
                   "medflux"),
        exact_variables=("optmask", "wave_1d"),
    ),
    S3RegressionCase(
        name="nirspec_spectroscopy",
        eventlabel="NIRSpec",
        ecf_dir="tests/NIRSpec_ecfs",
        input_dir="tests/data/JWST-Sim/NIRSpec/Stage2",
        reference_dir="nirspec_spectroscopy",
        mode="spectroscopy",
        variables=SPECTROSCOPY_VARIABLES,
        exact_variables=("optmask", "wave_1d"),
    ),
    S3RegressionCase(
        name="miri_spectroscopy",
        eventlabel="MIRI",
        ecf_dir="tests/MIRI_ecfs",
        input_dir="tests/data/JWST-Sim/MIRI/Stage2",
        reference_dir="miri_spectroscopy",
        mode="spectroscopy",
        variables=SPECTROSCOPY_VARIABLES,
        exact_variables=("optmask", "wave_1d"),
    ),
    S3RegressionCase(
        name="niriss_spectroscopy",
        eventlabel="NIRISS",
        ecf_dir="tests/NIRISS_ecfs",
        input_dir="tests/data/JWST-Sim/NIRISS/Stage2",
        reference_dir="niriss_spectroscopy",
        mode="spectroscopy",
        variables=SPECTROSCOPY_VARIABLES,
        exact_variables=("optmask", "wave_1d"),
        check_niriss_orders=True,
    ),
    S3RegressionCase(
        name="wfc3_spectroscopy",
        eventlabel="WFC3",
        ecf_dir="tests/WFC3_ecfs",
        input_dir="tests/data/WFC3/ima",
        reference_dir="wfc3_spectroscopy",
        mode="spectroscopy",
        variables=("optspec", "opterr", "optmask", "wave_1d",
                   "medflux"),
        exact_variables=("optmask", "wave_1d"),
    ),
    S3RegressionCase(
        name="nircam_photometry",
        eventlabel="Photometry_NIRCam",
        ecf_dir="tests/Photometry_NIRCam_ecfs",
        input_dir="tests/data/Photometry/NIRCam/Stage2",
        reference_dir="nircam_photometry",
        mode="photometry",
        variables=("centroid_x", "centroid_y", "centroid_sx",
                   "centroid_sy", "skylev", "skyerr", "nappix",
                   "nskypix"),
        exact_variables=("nappix", "nskypix"),
        atol_variables=("centroid_x", "centroid_y"),
    ),
    S3RegressionCase(
        name="nircam_photometry_hex",
        eventlabel="Photometry_NIRCam_hex",
        ecf_dir="tests/Photometry_NIRCam_ecfs",
        input_dir="tests/data/Photometry/NIRCam/Stage2",
        reference_dir="nircam_photometry_hex",
        mode="photometry",
        variables=("centroid_x", "centroid_y", "centroid_sx",
                   "centroid_sy", "skylev", "skyerr", "nappix",
                   "nskypix"),
        exact_variables=("nappix", "nskypix"),
        atol_variables=("centroid_x", "centroid_y"),
    ),
)
