"""Utilities for collecting Stage 3 regression reference metadata."""
import json
from pathlib import Path


def write_s3_reference_metadata(s3_meta):
    """Write Stage 3 metadata needed alongside a ``SpecData.h5`` reference.

    ``mad_s3`` is calculated after reduction but is not part of the saved
    ``SpecData.h5`` dataset.  Store it in a JSON sidecar while collecting a
    golden reference so a later regression test can compare it.
    """
    specdata_filename = Path(s3_meta.filename_S3_SpecData)
    mad_s3 = getattr(s3_meta, "mad_s3", None)
    if mad_s3 is None:
        raise AttributeError("Stage 3 metadata does not contain mad_s3.")

    metadata_filename = specdata_filename.with_suffix(".metadata.json")
    payload = {
        "reference_schema_version": 1,
        "mad_s3": [float(value) for value in mad_s3],
    }
    metadata_filename.write_text(json.dumps(payload, indent=2) + "\n")
    return metadata_filename
