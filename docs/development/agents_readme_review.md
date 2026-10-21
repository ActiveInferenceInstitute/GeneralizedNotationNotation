# AGENTS.md / README.md review

> **Snapshot disclaimer:** Any pytest or MCP counts mentioned in linked notes or older summaries are **not** authoritative. Use [current verification commands](../../TO-DO.md#verification-and-execution-rules) and measured output.

Historical summary of the first full repo-wide audit (links, pairing, canonical pytest counts, `docs/AGENTS.md` / `docs/dev/` coverage gaps). Ongoing quality gate: **[agents_readme_triple_review.md](agents_readme_triple_review.md)** (three-pass procedure + last run log).

## Automation

- [`docs_audit.py`](docs_audit.py) — `--strict` for CI; see [`docs_audit_report.md`](docs_audit_report.md).
- [`../dev/regenerate_src_doc_inventory.py`](../dev/regenerate_src_doc_inventory.py) — `src/` dir coverage.

## Canonical test line

Link counts to their source owner or exact measured receipt instead of copying
them between landing pages. Follow the [documentation review workflow](documentation_review.md)
and [manuscript custody procedure](../../.agent_rules/release.md) when inputs change.
