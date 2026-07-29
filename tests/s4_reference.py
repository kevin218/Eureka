"""Utilities for collecting Stage 4 regression reference metadata."""
import json
from pathlib import Path

import numpy as np


METADATA_ATTRIBUTES = (
    "mad_s4",
    "mad_s4_binned",
    "mad_s4_binned_bg",
    "mask_columns",
    "wave",
    "wave_low",
    "wave_hi",
    "nspecchan",
    "n_int",
)


def _json_value(value):
    """Convert a metadata value to a JSON-compatible Python value."""
    if value is np.ma.masked:
        return None
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    return value


def write_s4_reference_metadata(s4_meta):
    """Write Stage 4 metadata needed alongside the HDF5 references.

    Stage 4 quality metrics and binning details live on the metadata object,
    rather than in one of the saved science datasets. Store the relevant values
    in a JSON sidecar while collecting a reviewed golden reference.
    """
    lcdata_filename = Path(s4_meta.filename_S4_LCData)
    payload = {"reference_schema_version": 1}
    for attribute in METADATA_ATTRIBUTES:
        if hasattr(s4_meta, attribute):
            payload[attribute] = _json_value(getattr(s4_meta, attribute))

    missing = {"mad_s4", "wave", "wave_low", "wave_hi", "nspecchan",
               "n_int"} - payload.keys()
    if missing:
        raise AttributeError(
            "Stage 4 metadata is missing required reference attributes: "
            f"{', '.join(sorted(missing))}."
        )

    metadata_filename = lcdata_filename.with_suffix(".metadata.json")
    metadata_filename.write_text(json.dumps(payload, indent=2) + "\n")
    return metadata_filename
