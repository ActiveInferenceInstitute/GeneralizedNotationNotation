# Security Tests Agent

## Overview
This directory owns pytest coverage for `src/security/`.

## Purpose
- Validate real security processors, functional checks, and dependency vulnerability guardrails.
- Keep tests aligned with `src/security/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/security/ -q`.
