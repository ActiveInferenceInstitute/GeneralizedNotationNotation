# Integration Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/integration/`.

## Purpose
- Validate real integration processors, dependency checks, meta-analysis sweep collection/validation, and cross-module summaries.
- Keep tests aligned with `src/gnn/integration/AGENTS.md` and the integration module README.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/integration/ -q`.
