# Triple review — AGENTS.md / README.md

> **Snapshot disclaimer:** Numbers in the **Last run log** below are historical unless you re-run the same commands today. For current pytest pass/skip totals and CI parity, follow [CLAUDE.md](../../CLAUDE.md) and the commands there; do not treat this table as ground truth.

Formal **three-pass** gate so structural, canonical, and semantic checks do not collapse into a single skim.

| Pass | Focus | Primary tools |
|------|--------|----------------|
| 1 | Structure and links | [`docs_audit.py`](docs_audit.py) `--strict`, [`../dev/regenerate_src_doc_inventory.py`](../dev/regenerate_src_doc_inventory.py) |
| 2 | Single source of truth | Pytest (documented ignores), grep across `README.md`, `CLAUDE.md`, [`../../AGENTS.md`](../../AGENTS.md), [`../../src/AGENTS.md`](../../src/AGENTS.md) |
| 3 | Meaning vs code | Tier A step lines vs `src/N_*.py`; entry README sanity |

See also [`agents_readme_review.md`](agents_readme_review.md) for the first full audit summary and [`docs_audit_report.md`](docs_audit_report.md) for the latest machine report.

## Last run log

Fill in after each triple review.

| Field | Value |
|--------|--------|
| Date | 2026-03-24 |
| Pass 1 | `uv run python doc/development/docs_audit.py --strict` — exit 0; all audit sections empty |
| Pass 1b | `uv run python doc/dev/regenerate_src_doc_inventory.py` — 0 missing AGENTS, 0 missing README under `src/` |
| Pass 2 | `uv run pytest src/tests/ -q --tb=no --ignore=src/tests/test_llm_ollama.py --ignore=src/tests/test_llm_ollama_integration.py` — **1906 passed, 30 skipped**; matches `README.md`, `CLAUDE.md`, root `AGENTS.md`, `src/AGENTS.md` |
| Pass 3 | **25** files `src/N_*.py` for steps 0–24 present; Tier A step lines in module `AGENTS.md` previously spot-checked against orchestrators |
| CI parity | `uv sync --frozen --extra dev`; `PYTHONPATH=src uv run pytest -m "not pipeline and not mcp" --tb=short -q` — **1532 passed, 31 skipped**; MCP tools **133** (`count_mcp_tools` ≥ 131); `uv run ruff check src/`; Bandit medium+ exit 0 |
| Pipeline smoke | `uv run python src/main.py --only-steps "3,5,11,12" --target-dir input/gnn_files --verbose` — 4/4 success; step 12 `SUCCESS_WITH_WARNINGS` (namespace conflict note only) |
| Notes | Triple procedure added to plan; this file is the repo-persisted checklist |
