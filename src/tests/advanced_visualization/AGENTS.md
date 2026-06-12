# Advanced Visualization Tests Agent

## Overview
This directory owns pytest coverage for `src/advanced_visualization/`.

## Purpose
- Validate real advanced visualization processors, D2 helpers, shared utilities, and artifact behavior.
- Keep tests aligned with `src/advanced_visualization/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest src/tests/advanced_visualization/ -q`.
