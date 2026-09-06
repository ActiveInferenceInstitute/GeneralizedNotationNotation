# Export Tests Agent

## Overview
This directory owns pytest coverage for `src/export/`.

## Purpose
- Validate real export processors, round-trip behavior, and format-specific output contracts.
- Keep tests aligned with `src/export/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest src/tests/export/ -q`.

Gaussian interchange coverage lives in `test_geo_infer_gaussian.py` with the
three-state, two-observation, one-control `gaussian_rectangular.md` source.
Tests exercise strict extraction, explicit units/time, Step 7 metadata and source
containment, partial failures, default-format compatibility and CLI output.

`test_step7_geo_cli.py` runs the numbered Step 3/7 commands, verifies source
provenance, and rejects missing or duplicate physical metadata.
