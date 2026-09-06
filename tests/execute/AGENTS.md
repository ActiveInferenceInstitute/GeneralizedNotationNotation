# Execute Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/execute/`.

## Purpose
- Validate real execution processors, framework runners, PyMDP contracts, script collection, and summary behavior.
- Keep tests aligned with `src/gnn/execute/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/execute/ -q`.
