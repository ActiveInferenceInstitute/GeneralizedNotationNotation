# LSP Tests Agent

## Overview
This directory owns pytest coverage for `src/lsp/`.

## Purpose
- Validate real language-server startup, completions, diagnostics, and protocol-facing behavior.
- Keep tests aligned with `src/lsp/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest src/tests/lsp/ -q`.
