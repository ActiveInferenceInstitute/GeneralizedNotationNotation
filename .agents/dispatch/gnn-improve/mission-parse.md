# Parse & Core-Authoring Scope — mission-parse.md

You own these paths and ONLY these paths within the repo rooted at
/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation:

OWNS:
- src/gnn/  (GNN file discovery, parsing, multi-format serialization)
- src/model_registry/
- src/type_checker/
- src/validation/
- src/export/
- mirror tests: src/tests/gnn/, src/tests/model_registry/,
  src/tests/type_checker/, src/tests/validation/, src/tests/export/

DO NOT TOUCH anything outside that scope. In particular NEVER edit:
- pyproject.toml, justfile, uv.lock, pytest.ini, .gitignore, AGENTS.md,
  CLAUDE.md, README.md, CHANGELOG.md, src/main.py
- src/tests/conftest.py, src/tests/helpers/, src/tests/categories.py
- src/render/, src/execute/, src/analysis/, src/visualization/,
  src/advanced_visualization/, src/integration/, src/mcp/, src/api/,
  src/cli/, src/gui/, src/website/, src/utils/, src/pipeline/, src/lsp/

GOAL
Drive shallow/tactical through deep/strategic improvements in your scope.
Concrete, high-value directions (do NOT pad — every change must earn its keep):
1. Parser & schema robustness: find edge cases the GNN parsers mishandle
   (malformed input, unicode, nested structures, round-trip of list/tuple/
   dict/type-annotated values). Add targeted regression tests.
2. Schema validation completeness: tighten validation where a real invariant
   is currently unguarded, WITHOUT breaking existing allowed forms.
3. Type checking & model registry: richer diagnostics and missing-context
   errors; make failure modes explicit rather than silent.
4. Export: multi-format output correctness and round-trip fidelity.
5. Remove mypy strict smells (union returns, `x | None` plus `raise`, loose
   `Any`) where a cleaner type contract is genuinely better. Prefer removing
   smells over adding `# type: ignore`.

CONSTRAINT: Fix root causes, not symptoms. Do not add noise, artificial
"make it pass" tests, or docstring padding. Keep public API stable.

VERIFY (scoped, do NOT run the full suite — it is the coordinator's job):
- `uv run ruff check src/gnn src/model_registry src/type_checker src/validation src/export`
- `uv run ruff format --check src/gnn src/model_registry src/type_checker src/validation src/export`
- `uv run pytest src/tests/gnn src/tests/model_registry src/tests/type_checker src/tests/validation src/tests/export -q --tb=no -x`
- `uv run mypy src/gnn src/model_registry src/type_checker src/validation src/export --config-file pyproject.toml`

HARD RULE: DO NOT commit, DO NOT push, DO NOT stage. Leave ALL changes
uncommitted in the working tree. Do not touch git index/HEAD. Other agents
work on disjoint paths; ignore files you do not own even if they look
unfinished.

## Finish
Write a concise report to:
`/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation/.agents/dispatch/gnn-improve/REPORT-parse.md`
Summarize exactly what you changed (files + nature), tests added/passing, and
ruff/mypy scoped results. Reply with only the absolute path to your report.