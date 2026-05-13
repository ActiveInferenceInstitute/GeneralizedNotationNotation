# Analysis Tests Agent

## Overview
This directory owns pytest coverage for `src/analysis/`.

## Purpose
- Validate real analysis processors, extraction logic, post-simulation handling, and visualization helpers.
- Keep tests aligned with `src/analysis/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/analysis/ -q`.
