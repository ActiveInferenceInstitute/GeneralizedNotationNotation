# audio-gui — GNN documentation-vs-code audit (REPORT-ONLY)

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation

## Scope (REPORT-ONLY — DO NOT EDIT, STAGE, OR COMMIT ANY FILE)
This is a verication pass. You will WRITE ONLY your report file. Do NOT modify,
create, rename, or delete any repository file (source, docs, or tests). Do NOT
touch git index/HEAD. Your ONLY deliverable is the report described at the end.

## Mission
Audit your assigned documentation region for ACCURACY against the CURRENT
source tree. The repo's automated gates (link audit, doc-patterns, terminology)
already pass; your job is the DEEPER docs-vs-code cross-check that automation
cannot do. Verify — never guess:

1. **Documented commands exist.** For every shell/fenced command in your docs
   (e.g. `uv run python src/5_type_checker.py`, `python src/main.py`, `just test`,
   `scripts/foo.py`, `uv run --extra dev pytest ...`), verify the referenced
   script/path/module actually exists at that path and the command is well-formed.
   Flag stale/moved/renamed script paths.
2. **Documented Python imports/APIs exist.** For each `from X import Y` /
   `import X` or dotted `module.fn(...)` reference, verify the module exists under
   src/ and the symbol is importable (check via `.venv` python or hasattr, or grep
   `^(class|def) \bY\b` / `__all__`). Flag symbols that are fabricated/renamed/
   moved.
3. **Documented file paths exist.** Any relative path to a repo file (src/...,
   input/..., scripts/..., doc/...) — confirm it resolves to a real tracked file
   (`git ls-files <path>` — tracked source is the authority). Flag broken paths.
4. **Documented counts/claims are accurate.** Verify prose counts (number of
   modules, scripts, frameworks, tests) by globbing, not trusting the prose.
   Flag stale numbers.
5. **Config keys / env vars referenced exist.** Verify the env var is actually
   consumed somewhere in src/ (grep the name).

Report ONLY genuine findings. A command/API/path that exists and is well-formed
is NOT a finding. Do not invent problems. Distinguish:
- ERROR: missing module/symbol/script/path (would genuinely break/fail)
- WARNING: exists but moved/renamed (still works or misleading)
- INFO: cosmetic drift

Use the repo's own interpreter for verification: `.venv/bin/python -c ...` or
`uv run --extra dev python -c ...`. Prefer `git ls-files` to check tracked paths.

## Your assigned region
- doc/audio/, doc/sapf/, doc/pedalboard/, doc/gui_oxdraw/, doc/other/
Audit documented audio/sonification functions (src/audio/, src/audio/sapf/),
DSP parameters, GUI/oxdraw interfaces, and any src/gui references against the
current tree.

## Report
Write a concise evidence-first report to:
/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation/.agents/dispatch/gnn-docaudit/REPORT-audio-gui.md
Format: for each finding, one line: `<file>:<line> | <severity> | <finding>` then
a one-line suggested fix. If a region is clean, say so. Anchor claims with real
`ls`/`grep`/`git ls-files` output, not descriptions. Reply with only the absolute
path to your report.
