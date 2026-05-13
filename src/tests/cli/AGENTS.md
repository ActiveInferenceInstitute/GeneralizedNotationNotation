# CLI Tests Agent

## Overview
This directory owns pytest coverage for `src/cli/`.

## Purpose
- Validate real CLI parsing, subcommand routing, and command behavior.
- Keep tests aligned with `src/cli/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run pytest src/tests/cli/ -q`.
