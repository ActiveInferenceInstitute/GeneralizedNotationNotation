"""Bounded metadata-file loading for the numbered export command."""

import json
from pathlib import Path
from typing import Any


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate GEO metadata key: {key}")
        result[key] = value
    return result


def process_export_cli(
    target_dir: Path, output_dir: Path, verbose: bool = False, **kwargs: Any
) -> bool:
    """Load explicit per-model metadata and delegate to the export processor."""
    from .processor import process_export

    options_file = kwargs.pop("geo_infer_options_file", None)
    if kwargs.get("formats") is None:
        kwargs.pop("formats", None)
    if options_file is not None:
        with Path(options_file).open("rb") as stream:
            payload = stream.read(1024 * 1024 + 1)
        if len(payload) > 1024 * 1024:
            raise ValueError("GEO metadata file exceeds one MiB")
        options = json.loads(payload, object_pairs_hook=_unique_keys)
        if not isinstance(options, dict):
            raise ValueError("GEO metadata must map model filenames to options")
        kwargs["geo_infer_options"] = options
    return process_export(target_dir, output_dir, verbose=verbose, **kwargs)
