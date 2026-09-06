# MCP Tests Agent

## Overview
This directory owns pytest coverage for `src/gnn/mcp/`.

## Purpose
- Validate real MCP discovery, tool registration, configurability, audit reports, and performance behavior.
- Keep tests aligned with `src/gnn/mcp/AGENTS.md` and `README.md`.
- Do not place production implementation logic here.

## Verification
Run `uv run --extra dev python -m pytest tests/mcp/ -q`.
