# MCP Tests Agent

## Overview
This directory owns pytest coverage for `src/mcp/`.

## Purpose
- Validate real MCP discovery, tool registration, configurability, audit reports, and performance behavior.
- Keep tests aligned with `src/mcp/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest src/tests/mcp/ -q`.
