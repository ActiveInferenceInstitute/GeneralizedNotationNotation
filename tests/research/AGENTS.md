# Research Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/research/`.

## Purpose
- Validate real research processors, functional workflows, and dependency-aware behavior.
- Keep tests aligned with `src/gnn/research/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/research/ -q`.
