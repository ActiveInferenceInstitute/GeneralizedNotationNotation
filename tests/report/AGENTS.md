# Report Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/report/`.

## Purpose
- Validate real report generation, output formats, integration behavior, and empty-input handling.
- Keep tests aligned with `src/gnn/report/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/report/ -q`.
