# Setup Tests Agent

## Overview
This directory owns pytest coverage for `src/setup/`.

## Purpose
- Validate real setup checks, environment validation, and dependency-management behavior.
- Keep tests aligned with `src/setup/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest src/tests/setup/ -q`.
