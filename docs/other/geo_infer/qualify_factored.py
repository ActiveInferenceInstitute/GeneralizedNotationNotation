"""Create or independently reproduce an explicit GNN factored artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pydantic import JsonValue

from gnn.export.geo_infer_factored import build_geo_infer_factored_artifact


def _distinct_fields(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate artifact field {key}")
        result[key] = value
    return result


def bounded_bytes(path: Path) -> bytes:
    if path.is_symlink():
        raise ValueError("Factored input follows a symlink")
    with path.open("rb") as stream:
        raw = stream.read(4 * 1024 * 1024 + 1)
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("Factored input exceeds four MiB")
    return raw


def qualify(source: Path, artifact: Path, step_seconds: float, *, check: bool) -> str:
    raw = bounded_bytes(source)
    if artifact.is_symlink():
        raise ValueError("Factored artifact follows a symlink")
    produced = build_geo_infer_factored_artifact(
        raw.decode("utf-8"), step_seconds=step_seconds
    )
    canonical = json.dumps(
        produced, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if check:
        actual = json.loads(bounded_bytes(artifact), object_pairs_hook=_distinct_fields)
        if (
            json.dumps(actual, sort_keys=True, separators=(",", ":"), allow_nan=False)
            != canonical
        ):
            raise ValueError(
                "Captured factored artifact differs from actual GNN reproduction"
            )
    else:
        with artifact.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(produced, indent=2, allow_nan=False) + "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--step-seconds", type=float, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    print(qualify(args.source, args.artifact, args.step_seconds, check=args.check))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
