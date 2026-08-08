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
    exact_variables: tuple[str, ...] = ()
    atol_variables: tuple[str, ...] = ()


# Every spectroscopy case must retain these extracted-spectrum and binned
# light-curve arrays.  The per-case additions below cover instrument-specific
# diagnostics and optional white-light products.

# Define groups of variables for spectroscopy and photometry that are commonly outputs of the 
# same isntrument and hence tested together. Groups aren't perfect but meant to reduce
# listing the variables repeatedly for each case


SPECTROSCOPY_SPEC_CORE = (
    "wave_1d", "medflux", "stdspec", "stdvar", "optspec", "opterr",
    "optmask",
)

SPECTROSCOPY_LC_CORE = (
    "data", "err", "mask", "wave_low", "wave_hi", "wave_mid", "wave_err",
)

SKY_SPEC_VARIABLES = ("skylev", "skyerr")
SKY_LC_VARIABLES = ("skylev", "skyerr")
Y_CENTROID_VARIABLES = ("centroid_y", "centroid_sy")
WHITE_LIGHT_VARIABLES = (
    "flux_white", "err_white", "mask_white", "skylev_white", "skyerr_white",
)

# Vars in `spec_variables` and `lc_variables` are tested using a relative tolerance
# Vars in `exact_variables` can come from SpecData.h5 or LCData.h5 and are tested exactly
# Vars in `atol_variables` can come from SpecData.h5 or LCData.h5 and are tested using an absolute tolerance
CASES = (
    S4RegressionCase(
        name="nircam_spectroscopy",
        eventlabel="NIRCam",
        ecf_dir="tests/NIRCam_ecfs",
        s3_reference_dir="nircam_spectroscopy",
        input_filename="S3_NIRCam_ap8_bg12_SpecData.h5",
        reference_dir="nircam_spectroscopy",
        spec_variables=(SPECTROSCOPY_SPEC_CORE + Y_CENTROID_VARIABLES +
                        ("centroid_x", "centroid_sx", "driftmask")),
        lc_variables=(SPECTROSCOPY_LC_CORE + Y_CENTROID_VARIABLES +
                      ("centroid_x", "centroid_sx", "driftmask")),
        mode="spectroscopy",
        exact_variables=("optmask", "mask", "driftmask"),
        atol_variables=("centroid_x", "centroid_y"),
    ),
    S4RegressionCase(
        name="nirspec_spectroscopy",
        eventlabel="NIRSpec",
        ecf_dir="tests/NIRSpec_ecfs",
        s3_reference_dir="nirspec_spectroscopy",
        input_filename="S3_NIRSpec_ap5_bg10_SpecData.h5",
        reference_dir="nirspec_spectroscopy",
        spec_variables=(SPECTROSCOPY_SPEC_CORE + SKY_SPEC_VARIABLES +
                        Y_CENTROID_VARIABLES),
        lc_variables=(SPECTROSCOPY_LC_CORE + SKY_LC_VARIABLES +
                      Y_CENTROID_VARIABLES + WHITE_LIGHT_VARIABLES),
        mode="spectroscopy",
        exact_variables=("optmask", "mask", "mask_white"),
        atol_variables=("centroid_y",),
    ),
    S4RegressionCase(
        name="miri_spectroscopy",
        eventlabel="MIRI",
        ecf_dir="tests/MIRI_ecfs",
        s3_reference_dir="miri_spectroscopy",
        input_filename="S3_MIRI_ap4_bg10_SpecData.h5",
        reference_dir="miri_spectroscopy",
        spec_variables=(SPECTROSCOPY_SPEC_CORE + SKY_SPEC_VARIABLES +
                        Y_CENTROID_VARIABLES),
        lc_variables=(SPECTROSCOPY_LC_CORE + SKY_LC_VARIABLES +
                      Y_CENTROID_VARIABLES + WHITE_LIGHT_VARIABLES),
        mode="spectroscopy",
        exact_variables=("optmask", "mask", "mask_white"),
        atol_variables=("centroid_y",),
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
        spec_variables=(SPECTROSCOPY_SPEC_CORE + SKY_SPEC_VARIABLES +
                        ("trace",)),
        lc_variables=SPECTROSCOPY_LC_CORE + SKY_LC_VARIABLES,
        mode="spectroscopy",
        exact_variables=("optmask", "mask"),
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
        exact_variables=("mask", "nappix", "nskypix", "nskyideal", "status"),
        atol_variables=("centroid_x", "centroid_y"),
    ),
    S4RegressionCase(
        name="wfc3_spectroscopy",
        eventlabel="WFC3",
        ecf_dir="tests/WFC3_ecfs",
        s3_reference_dir="wfc3_spectroscopy",
        input_filename="S3_WFC3_ap5_bg8_SpecData.h5",
        reference_dir="wfc3_spectroscopy",
        spec_variables=(SPECTROSCOPY_SPEC_CORE +
                        ("flatmask", "variance", "guess", "scanHeight", "wave",
                         "scandir", "centroid_x", "centroid_y", "centroid_sx",
                         "driftmask")),
        lc_variables=(SPECTROSCOPY_LC_CORE +
                      ("scandir", "centroid_y", "centroid_x", "centroid_sx",
                       "driftmask")),
        mode="spectroscopy",
        exact_variables=("optmask", "mask", "flatmask", "scandir",
                         "driftmask"),
        atol_variables=("centroid_x", "centroid_y"),
    ),
)
