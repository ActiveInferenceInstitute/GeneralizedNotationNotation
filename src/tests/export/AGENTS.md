# Export Tests Agent

## Overview
This directory owns pytest coverage for `src/export/`.

## Purpose
- Validate real export processors, round-trip behavior, and format-specific output contracts.
- Keep tests aligned with `src/export/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest src/tests/export/ -q`.
