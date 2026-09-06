# Setup Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/setup/`.

## Purpose
- Validate real setup checks, environment validation, and dependency-management behavior.
- Keep tests aligned with `src/gnn/setup/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/setup/ -q`.
