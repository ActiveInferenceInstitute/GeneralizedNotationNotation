# A-dev — GNN documentation accuracy fixes (SAFE DOC-ONLY EDITS)

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation

## Mission
Apply the documentation corrections listed below. These are SAFE, DOCUMENTATION-ONLY
edits: change markdown prose, code examples, import references, file paths, counts,
and doc metadata. Do NOT change any source code (.py), test logic, config, or
dependency. The fixes were verified against the current tree (imports resolve; paths
are tracked); your job is to apply them to the .md files exactly.

## Rules
- Edit ONLY the .md files named below. Do NOT touch src/**, scripts/**, pyproject.toml,
  tests/**, or any .py.
- Preserve surrounding formatting/markdown. Make the minimal change (one line / one
  token) per fix.
- For imports: replace the broken module path with the verified-correct one below.
- For counts: set the number to the value stated below (verified by git ls-files/wc).
- For prose claims that are wrong/unverifiable and no exact replacement exists: reword
  minimally to be accurate (e.g. mark as illustrative, or remove the fabricated claim).
- HARD RULE: do NOT commit, stage, or push. Leave changes uncommitted.

## Specific fixes to apply
FIXES:
1. doc/development/README.md:307 (and :307-354 examples) — `from src.gnn.parser import parse_gnn_file` -> `from src.gnn.processor import parse_gnn_file`
2. doc/development/README.md:168 — `from utils.logging_utils import setup_standalone_logging` -> `from utils.logging.logging_utils import setup_standalone_logging`
3. doc/development/README.md:362 — pytest path `src/tests/unit/` -> a real dir, e.g. `src/tests/gnn/`
4. doc/development/README.md:369 — `src/tests/performance/` -> real path, e.g. `src/tests/pipeline/` (drop `--benchmark-only` or the command)
5. doc/development/README.md:471 — `src/tests/unit/test_specific.py` -> a real file, e.g. `src/tests/gnn/test_gnn_overall.py`
6. doc/development/README.md:120 — count "171 pytest files" -> current tracked value (src/tests/ has ~323 .py) or reword to "mirrored by module" without a hardcoded number
7. doc/development/README.md:287-300 — test-organization tree shows tests/unit, tests/integration, tests/fixtures; rewrite to the real layout (src/tests/<module>/, src/tests/integration/, src/tests/helpers/, etc.)
8. doc/configuration/examples.md — add a prominent note at top (or in each block) that this file is ILLUSTRATIVE/aspirational and the only auto-loaded config is `input/config.yaml` (schema described in doc/configuration/README.md). Do not claim it is loadable.
9. doc/configuration/README.md:65 — add `uv` to the enumerated config sections (top-level keys are: pipeline, setup, uv, io, testing_matrix, llm, logging, validation, performance, security)
10. SKILL.md:25 — "Check all 8 renderer backends" -> 9 renderer backends (framework_registry has 9 incl. bnlearn)
11. SKILL.md:14 — add bnlearn to the framework list if a list is shown
12. SKILL.md:22 — `just  # List all 21 recipes` -> drop the count, e.g. `just  # List recipes`


## Verification
After editing, run:
- uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write   (must stay green)
- uv run --extra dev python scripts/check_repo_terminology.py --strict   (must stay clean — do NOT use banned words: legacy/stub/placeholder/deprecated)
- uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict

## Report
Write a concise report to /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation/.agents/dispatch/gnn-docfix/REPORT-fix-A-dev.md listing each file you changed and the specific
edit(s) applied. Reply with only the absolute path to your report.
