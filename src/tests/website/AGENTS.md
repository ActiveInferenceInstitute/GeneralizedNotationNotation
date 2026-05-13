# Website Tests Agent

## Overview
This directory owns pytest coverage for `src/website/`.

## Purpose
- Validate real website generation, static artifact assembly, and pipeline-output discovery.
- Keep tests aligned with `src/website/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/website/ -q`.
