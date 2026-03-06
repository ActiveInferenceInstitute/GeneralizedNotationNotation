#!/usr/bin/env python3
"""
Content-Addressable Run Hashing — Reproducible pipeline identification.

Provides:
  - compute_run_hash(): SHA256 of input files + config → 12-char hex prefix
  - index_run(): store run metadata in .history/index.json
  - lookup_run(): retrieve run config by hash prefix
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def compute_run_hash(
    target_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    hash_length: int = 12,
) -> str:
    """
    Compute a content-addressable hash for a pipeline run.

    The hash is derived from:
      1. Sorted SHA256 hashes of all input files
      2. SHA256 of the serialized config dict

    Args:
        target_dir: Directory containing input GNN files.
        config: Optional pipeline configuration dict.
        hash_length: Length of the hex prefix to return.

    Returns:
        Hex hash string (default 12 chars).
    """
    hasher = hashlib.sha256()

    # Hash all input files (sorted for determinism)
    target_dir = Path(target_dir)
    file_hashes = []
    if target_dir.exists():
        for f in sorted(target_dir.rglob("*.md")):
            try:
                content = f.read_bytes()
                fh = hashlib.sha256(content).hexdigest()
                file_hashes.append(f"{f.name}:{fh}")
            except OSError:
                pass

    for fh in file_hashes:
        hasher.update(fh.encode())

    # Hash config
    if config:
        config_str = json.dumps(config, sort_keys=True)
        hasher.update(config_str.encode())

    run_hash = hasher.hexdigest()[:hash_length]
    logger.debug(f"Run hash: {run_hash} ({len(file_hashes)} input files)")
    return run_hash


def index_run(
    run_hash: str,
    summary_path: Path,
    history_dir: Optional[Path] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Store run metadata in .history/index.json.

    Args:
        run_hash: The computed run hash.
        summary_path: Path to pipeline_execution_summary.json.
        history_dir: Archive dir. Defaults to summary_path.parent / ".history".
        config: Optional config dict for re-running.

    Returns:
        Path to index.json.
    """
    history_dir = history_dir or summary_path.parent / ".history"
    history_dir.mkdir(parents=True, exist_ok=True)
    index_path = history_dir / "index.json"

    # Load existing index
    index = {}
    if index_path.exists():
        try:
            with open(index_path) as f:
                index = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Add/update entry
    index[run_hash] = {
        "summary_path": str(summary_path),
        "config": config or {},
    }

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"📇 Indexed run {run_hash} in {index_path}")
    return index_path


def lookup_run(
    run_hash_prefix: str,
    history_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Look up a run by its hash prefix.

    Args:
        run_hash_prefix: Partial or full hash to match.
        history_dir: Directory containing index.json.

    Returns:
        Run entry dict if found, None otherwise.
    """
    index_path = history_dir / "index.json"
    if not index_path.exists():
        return None

    try:
        with open(index_path) as f:
            index = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Exact match first
    if run_hash_prefix in index:
        return index[run_hash_prefix]

    # Prefix match
    matches = {k: v for k, v in index.items() if k.startswith(run_hash_prefix)}
    if len(matches) == 1:
        return next(iter(matches.values()))
    elif len(matches) > 1:
        logger.warning(f"Ambiguous hash prefix '{run_hash_prefix}': {len(matches)} matches")

    return None
