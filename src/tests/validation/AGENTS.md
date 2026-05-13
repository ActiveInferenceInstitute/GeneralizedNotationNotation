# Validation Tests Agent

## Overview
This directory owns pytest coverage for `src/validation/`.

## Purpose
- Validate real validation processors, semantic checks, consistency handling, and empty-input behavior.
- Keep tests aligned with `src/validation/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/validation/ -q`.
