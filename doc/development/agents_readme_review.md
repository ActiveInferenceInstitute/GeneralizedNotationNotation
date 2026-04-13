# AGENTS.md / README.md review

> **Snapshot disclaimer:** Any pytest or MCP counts mentioned in linked notes or older summaries are **not** authoritative. Use [CLAUDE.md](../../CLAUDE.md) and measured command output.

Historical summary of the first full repo-wide audit (links, pairing, canonical pytest counts, `doc/AGENTS.md` / `doc/dev/` stubs). Ongoing quality gate: **[agents_readme_triple_review.md](agents_readme_triple_review.md)** (three-pass procedure + last run log).

## Automation

- [`docs_audit.py`](docs_audit.py) — `--strict` for CI; see [`docs_audit_report.md`](docs_audit_report.md).
- [`../dev/regenerate_src_doc_inventory.py`](../dev/regenerate_src_doc_inventory.py) — `src/` dir coverage.

## Canonical test line

When counts change, update together: `README.md`, `CLAUDE.md`, root `AGENTS.md`, `src/AGENTS.md` (same `uv run pytest ...` ignores and passed/skipped/date).
