# Execute Tests Agent

## Overview
This directory owns pytest coverage for `src/execute/`.

## Purpose
- Validate real execution processors, framework runners, PyMDP contracts, script collection, and summary behavior.
- Keep tests aligned with `src/execute/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/execute/ -q`.
