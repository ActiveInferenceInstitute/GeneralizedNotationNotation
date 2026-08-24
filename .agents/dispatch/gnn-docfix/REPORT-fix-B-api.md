# REPORT — fix-B-api (GNN documentation accuracy, DOC-ONLY)

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation
Date: 2026-08-24

All fixes from .agents/dispatch/gnn-docfix/fix-B-api.md applied. Documentation-only edits;
no .py, test, config, or dependency files were touched. Changes left uncommitted.

## Files changed and edits

### doc/api/README.md (fix 1)
- Replaced `from gnn.render import PyMDPRenderer, RxInferRenderer` (broken import) with the
  real API: `from render import PyMDPRenderer` and
  `from render import render_gnn_to_pymdp, render_gnn_to_rxinfer`.
- `RxInferRenderer` does not exist, so the RxInfer usage was updated to the real functional
  API `render_gnn_to_rxinfer(model, output_path)` (verified signature in src/render/__init__.py).
  PyMDPRenderer usage left as-is (it exists).

### doc/llm/README.md (fixes 2-5)
- Comment `# From pipeline step 11 (11_llm.py)` -> `# From pipeline step 13 (13_llm.py)`.
- `from src.llm import get_processor` -> `from src.llm import get_global_processor`;
  `processor = get_processor()` -> `processor = get_global_processor()` (get_global_processor is
  the exported alias, verified in src/llm/__init__.py).
- Annotated env vars not consumed by src/: ENABLE_FALLBACK and ENABLE_STREAMING are now marked
  "Reserved — not currently consumed by src/"; DEFAULT_TEMPERATURE marked reserved/not consumed.
- Reworded DEFAULT_MAX_TOKENS as a module-level default constant, not read from env.

### doc/security/README.md (fix 6 + gunicorn target)
- `# src/gnn/security/validator.py` -> annotated illustrative, pointing to the real
  src/security/processor.py.
- `# src/llm/security/prompt_sanitizer.py` -> annotated illustrative (src/llm has no security/ subdir).
- `# src/mcp/security/secure_server.py` -> annotated illustrative.
- `# tests/security/test_security.py` -> corrected to `# src/tests/security/test_security_functional.py`.
- `# src/security/audit.py` -> annotated illustrative (path does not exist).
- Added an illustrative note that the gunicorn target `src.main:app` does not exist in this repo
  (fix 8's `src.main:app` reference actually lives here in README.md, so annotated it in place).

### doc/security/security_framework.md (fixes 7-8)
- `from gnn.security import ...` -> annotated as illustrative/aspirational (not importable).
- `from gnn.auth import ...` -> annotated as illustrative/aspirational (not importable).
- `--secure-mode` flag on `src.main.py` -> annotated as illustrative (flag does not exist).

### File-count metadata (fix 9)
- doc/mcp/AGENTS.md: "Files: 3" -> "Files: 5".
- doc/llm/AGENTS.md: "Files: 1" -> "Files: 4".
- doc/security/AGENTS.md: "Files: 2" -> "Files: 10".

## Verification
- `uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write`
  -> Broken links: 0, Bad markdown anchors: 0, AGENTS/SPEC gaps: 0 (green).
- `uv run --extra dev python scripts/check_repo_terminology.py --strict` -> maintained tree clean.
- `uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict` -> no banned patterns.

No commit/stage/push performed, per charter HARD RULE.
