# Utils Tests Agent

## Overview
This directory owns pytest coverage for `src/utils/`.

## Purpose
- Validate real shared utility contracts, validation schemas, pipeline template behavior, and framework availability checks.
- Keep tests aligned with `src/utils/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/utils/ -q`.
