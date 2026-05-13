# Intelligent Analysis Tests Agent

## Overview
This directory owns pytest coverage for `src/intelligent_analysis/`.

## Purpose
- Validate real intelligent-analysis processors, summary analysis, reporting, and remediation helpers.
- Keep tests aligned with `src/intelligent_analysis/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/intelligent_analysis/ -q`.
