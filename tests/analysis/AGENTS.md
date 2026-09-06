# Analysis Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/analysis/`.

## Purpose
- Validate real analysis processors, extraction logic, post-simulation handling, and visualization helpers.
- Keep tests aligned with `src/gnn/analysis/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/analysis/ -q`.
