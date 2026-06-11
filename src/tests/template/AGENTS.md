# Template Tests Agent

## Overview
This directory owns pytest coverage for `src/template/`.

## Purpose
- Validate real template scaffold processing and standardized step behavior.
- Keep tests aligned with `src/template/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest src/tests/template/ -q`.
