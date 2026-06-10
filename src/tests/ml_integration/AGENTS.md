# ML Integration Tests Agent

## Overview
This directory owns pytest coverage for `src/ml_integration/`.

## Purpose
- Validate real ML integration processors, coverage surfaces, and dependency-aware behavior.
- Keep tests aligned with `src/ml_integration/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest src/tests/ml_integration/ -q`.
