# Type Checker Tests Agent

## Overview
This directory owns pytest coverage for `src/type_checker/`.

## Purpose
- Validate real type-checking processors, dimension checks, resource estimation, and output contracts.
- Keep tests aligned with `src/type_checker/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/type_checker/ -q`.
