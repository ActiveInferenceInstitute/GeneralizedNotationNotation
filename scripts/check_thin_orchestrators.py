#!/usr/bin/env python3
"""Fail when a numbered thin orchestrator exceeds its registered line caps.

Every ``src/gnn/[0-9]*_*.py`` file must stay a minimal argument-parsing shell
that delegates to its ``src/gnn/<module>/`` implementation package (the thin
orchestrator contract). Physical line count is the drift detector: growth past
the registered caps means step logic leaked back into the orchestrator file.

Mechanics: count the physical lines of every numbered orchestrator and apply
two caps.

- Hard cap ``HARD_CAP_LINES`` (150): the architectural ceiling for the
  pattern. Any file over it fails unconditionally, with guidance to split the
  step (move logic into the corresponding ``src/gnn/<module>/`` package).
- Ratchet cap ``scripts/thin_orchestrator_caps.json["max_lines"]``: the
  measured maximum at registration. Any file over the ratchet also fails,
  with instructions to bump the cap deliberately (edit the JSON to the new
  measured maximum) once the growth is reviewed; lowering the value as files
  shrink is how the ratchet drives the count down over time.

The cap file is required; a missing or malformed file is a gate error, never
a silent pass. Fully deterministic: file reads only; no network, no clock, no
randomness.

Usage: uv run --extra dev python scripts/check_thin_orchestrators.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORCHESTRATOR_GLOB = "src/gnn/[0-9]*_*.py"
HARD_CAP_LINES = 150
RATCHET_CAP_FILE = ROOT / "scripts" / "thin_orchestrator_caps.json"


def load_ratchet_cap() -> int:
    """Read the ratchet cap; a missing/malformed cap file is a gate error."""
    if not RATCHET_CAP_FILE.exists():
        raise SystemExit(
            f"check_thin_orchestrators: missing cap file "
            f"{RATCHET_CAP_FILE.relative_to(ROOT)} - create it as "
            '{"max_lines": <measured maximum>} to register the ratchet.'
        )
    try:
        data = json.loads(RATCHET_CAP_FILE.read_text(encoding="utf-8"))
        cap = int(data["max_lines"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit(
            f"check_thin_orchestrators: unreadable cap file "
            f"{RATCHET_CAP_FILE.relative_to(ROOT)} ({exc}) - expected "
            '{"max_lines": <int>}.'
        ) from exc
    if not 1 <= cap <= HARD_CAP_LINES:
        raise SystemExit(
            f"check_thin_orchestrators: ratchet cap {cap} outside the sane "
            f"range 1..{HARD_CAP_LINES} - fix "
            f"{RATCHET_CAP_FILE.relative_to(ROOT)}."
        )
    return cap


def collect_line_counts() -> dict[str, int]:
    """Map every numbered orchestrator (repo-relative) to its line count."""
    counted: dict[str, int] = {}
    for path in sorted(ROOT.glob(ORCHESTRATOR_GLOB)):
        text = path.read_text(encoding="utf-8")
        counted[path.relative_to(ROOT).as_posix()] = len(text.splitlines())
    return counted


def main() -> int:
    """Apply both caps and report; exit 1 on any breach."""
    parser = argparse.ArgumentParser(
        description=(__doc__ or "Thin orchestrator line-cap gate").splitlines()[0]
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Compatibility flag; cap breaches fail by default.",
    )
    args = parser.parse_args()
    del args  # the gate fails on any breach, with or without --strict

    ratchet_cap = load_ratchet_cap()
    counted = collect_line_counts()
    if not counted:
        print(
            "check_thin_orchestrators: no files match "
            f"{ORCHESTRATOR_GLOB} - the orchestrator layout changed; update "
            "this gate."
        )
        return 1
    max_count = max(counted.values())

    over_hard = {name: n for name, n in counted.items() if n > HARD_CAP_LINES}
    over_ratchet = {name: n for name, n in counted.items() if n > ratchet_cap}

    if over_hard:
        print(
            f"check_thin_orchestrators: {len(over_hard)} orchestrator(s) over "
            f"the {HARD_CAP_LINES}-line hard cap:"
        )
        for name in sorted(over_hard, key=lambda n: (-over_hard[n], n)):
            print(f"  {name}: {over_hard[name]} lines (hard cap {HARD_CAP_LINES})")
        print(
            "Split the step: move domain logic into the corresponding "
            "src/gnn/<module>/ package and keep this file a thin delegating "
            "orchestrator."
        )
        return 1

    if over_ratchet:
        print(
            f"check_thin_orchestrators: {len(over_ratchet)} orchestrator(s) "
            f"over the registered ratchet cap ({ratchet_cap} lines):"
        )
        for name in sorted(over_ratchet, key=lambda n: (-over_ratchet[n], n)):
            print(f"  {name}: {over_ratchet[name]} lines (ratchet {ratchet_cap})")
        print(
            "If the growth is deliberate, bump the cap by editing "
            f"{RATCHET_CAP_FILE.relative_to(ROOT)} to the new measured "
            "maximum (never above the 150-line hard cap); otherwise shrink "
            "the file back under the cap. When files shrink, lower the "
            "ratchet to the new maximum."
        )
        return 1

    print(
        f"check_thin_orchestrators: {len(counted)} orchestrators, max "
        f"{max_count} lines (hard cap {HARD_CAP_LINES}, ratchet {ratchet_cap})."
    )
    if max_count < ratchet_cap:
        print(
            f"  note: ratchet can be lowered to {max_count} in "
            f"{RATCHET_CAP_FILE.relative_to(ROOT)} after this run."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
