# Integration Tests Agent

## Overview
This directory owns pytest coverage for `src/integration/`.

## Purpose
- Validate real integration processors, dependency checks, meta-analysis behavior, and cross-module summaries.
- Keep tests aligned with `src/integration/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/integration/ -q`.
