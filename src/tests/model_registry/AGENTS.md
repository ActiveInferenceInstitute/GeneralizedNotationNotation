# Model Registry Tests Agent

## Overview
This directory owns pytest coverage for `src/model_registry/`.

## Purpose
- Validate real model registry processing, metadata extraction, and round-trip behavior.
- Keep tests aligned with `src/model_registry/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/model_registry/ -q`.
