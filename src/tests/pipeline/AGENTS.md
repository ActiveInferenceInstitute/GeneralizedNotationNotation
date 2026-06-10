# Pipeline Tests Agent

## Overview
This directory owns pytest coverage for `src/pipeline/` and `src/main.py`.

## Purpose
- Validate real pipeline orchestration, numbered scripts, recovery, integration chains, and performance-facing contracts.
- Keep tests aligned with `src/pipeline/AGENTS.md`, `src/pipeline/README.md`, and `src/AGENTS.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest src/tests/pipeline/ -q`.
