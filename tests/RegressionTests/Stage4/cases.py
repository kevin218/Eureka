"""Definitions of the approved Stage 4 regression cases."""
from dataclasses import dataclass


@dataclass(frozen=True)
class S4RegressionCase:
    """Inputs and expected products for one Stage 4 reduction mode."""

    name: str
    eventlabel: str
    ecf_dir: str
    s3_reference_dir: str
    input_filename: str
    reference_dir: str
    spec_variables: tuple[str, ...]
    lc_variables: tuple[str, ...]
    mode: str
    check_miri_sorting: bool = False
    check_niriss_mask_columns: bool = False
    check_wfc3_read_summing: bool = False


SPECTROSCOPY_SPEC_VARIABLES = (
    "wave_1d", "medflux", "skylev", "skyerr", "centroid_y",
    "centroid_sy", "stdspec", "stdvar", "optspec", "opterr",
    "optmask",
)

SPECTROSCOPY_LC_VARIABLES = (
    "data", "err", "mask", "skylev", "skyerr", "centroid_y",
    "centroid_sy", "wave_low", "wave_hi", "wave_mid", "wave_err",
)

CASES = (
    S4RegressionCase(
        name="nircam_spectroscopy",
        eventlabel="NIRCam",
        ecf_dir="tests/NIRCam_ecfs",
        s3_reference_dir="nircam_spectroscopy",
        input_filename="S3_NIRCam_ap8_bg12_SpecData.h5",
        reference_dir="nircam_spectroscopy",
        spec_variables=("wave_1d", "medflux", "centroid_y", "centroid_sy",
                        "stdspec", "stdvar", "optspec", "opterr",
                        "optmask", "centroid_x", "centroid_sx", "driftmask"),
        lc_variables=("data", "err", "mask", "centroid_y", "centroid_sy",
                      "centroid_x", "centroid_sx", "driftmask", "wave_low",
                      "wave_hi", "wave_mid", "wave_err"),
        mode="spectroscopy",
    ),
    S4RegressionCase(
        name="nirspec_spectroscopy",
        eventlabel="NIRSpec",
        ecf_dir="tests/NIRSpec_ecfs",
        s3_reference_dir="nirspec_spectroscopy",
        input_filename="S3_NIRSpec_ap5_bg10_SpecData.h5",
        reference_dir="nirspec_spectroscopy",
        spec_variables=SPECTROSCOPY_SPEC_VARIABLES,
        lc_variables=SPECTROSCOPY_LC_VARIABLES + ("flux_white", "err_white",
                                                   "mask_white", "skylev_white",
                                                   "skyerr_white"),
        mode="spectroscopy",
    ),
    S4RegressionCase(
        name="miri_spectroscopy",
        eventlabel="MIRI",
        ecf_dir="tests/MIRI_ecfs",
        s3_reference_dir="miri_spectroscopy",
        input_filename="S3_MIRI_ap4_bg10_SpecData.h5",
        reference_dir="miri_spectroscopy",
        spec_variables=SPECTROSCOPY_SPEC_VARIABLES,
        lc_variables=SPECTROSCOPY_LC_VARIABLES + ("flux_white", "err_white",
                                                   "mask_white", "skylev_white",
                                                   "skyerr_white"),
        mode="spectroscopy",
        check_miri_sorting=True,
    ),
    S4RegressionCase(
        name="niriss_spectroscopy",
        eventlabel="NIRISS",
        ecf_dir="tests/NIRISS_ecfs",
        s3_reference_dir="niriss_spectroscopy",
        input_filename="S3_NIRISS_ap17_bg22_SpecData.h5",
        reference_dir="niriss_spectroscopy",
        # NIRISS does not produce the centroid arrays included in the
        # generic spectroscopy manifest.
        spec_variables=("trace", "wave_1d", "medflux", "skylev", "skyerr",
                        "stdspec", "stdvar", "optspec", "opterr", "optmask"),
        lc_variables=("data", "err", "mask", "skylev", "skyerr",
                      "wave_low", "wave_hi", "wave_mid", "wave_err"),
        mode="spectroscopy",
        check_niriss_mask_columns=True,
    ),
    S4RegressionCase(
        name="nircam_photometry",
        eventlabel="Photometry_NIRCam",
        ecf_dir="tests/Photometry_NIRCam_ecfs",
        s3_reference_dir="nircam_photometry",
        input_filename="S3_Photometry_NIRCam_ap60_bg70_90_SpecData.h5",
        reference_dir="nircam_photometry",
        spec_variables=("wave_1d", "centroid_x", "centroid_y", "centroid_sx",
                        "centroid_sy", "aplev", "aperr", "nappix", "skylev",
                        "skyerr", "nskypix", "nskyideal", "status", "betaper",
                        "medflux"),
        lc_variables=("data", "err", "mask", "skylev", "skyerr", "centroid_y",
                      "centroid_sy", "centroid_x", "centroid_sx", "wave_low",
                      "wave_hi", "wave_mid", "wave_err"),
        mode="photometry",
    ),
    S4RegressionCase(
        name="wfc3_spectroscopy",
        eventlabel="WFC3",
        ecf_dir="tests/WFC3_ecfs",
        s3_reference_dir="wfc3_spectroscopy",
        input_filename="S3_WFC3_ap5_bg8_SpecData.h5",
        reference_dir="wfc3_spectroscopy",
        spec_variables=("flatmask", "variance", "guess", "scanHeight", "wave",
                        "scandir", "wave_1d", "medflux", "centroid_x",
                        "centroid_y", "stdspec", "stdvar", "optspec", "opterr",
                        "optmask", "centroid_sx", "driftmask"),
        lc_variables=("data", "err", "mask", "scandir", "centroid_y",
                      "centroid_x", "centroid_sx", "driftmask", "wave_low",
                      "wave_hi", "wave_mid", "wave_err"),
        mode="spectroscopy",
        check_wfc3_read_summing=True,
    ),
)
