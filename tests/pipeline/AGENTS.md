# Pipeline Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/pipeline/` and `src/gnn/main.py`.

## Purpose
- Validate real pipeline orchestration, numbered scripts, recovery, integration chains, and performance-facing contracts.
- Keep tests aligned with `src/gnn/pipeline/AGENTS.md`, `src/gnn/pipeline/README.md`, and `src/gnn/AGENTS.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/pipeline/ -q`.
