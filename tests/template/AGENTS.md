# Template Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/template/`.

## Purpose
- Validate real template scaffold processing and standardized step behavior.
- Keep tests aligned with `src/gnn/template/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/template/ -q`.
