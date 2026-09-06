# Validation Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/validation/`.

## Purpose
- Validate real validation processors, semantic checks, consistency handling, and empty-input behavior.
- Keep tests aligned with `src/gnn/validation/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/validation/ -q`.
