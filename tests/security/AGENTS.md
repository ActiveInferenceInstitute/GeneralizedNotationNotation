# Security Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/security/`.

## Purpose
- Validate real security processors, functional checks, and dependency vulnerability guardrails.
- Keep tests aligned with `src/gnn/security/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/security/ -q`.
