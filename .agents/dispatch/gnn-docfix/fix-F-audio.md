# F-audio — GNN documentation accuracy fixes (SAFE DOC-ONLY EDITS)

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
1. doc/pedalboard/AGENTS.md:130 and README.md:11 — change "Production Ready" claim to scaffold/planned: src/audio/pedalboard/ is scaffold-only (no .py). Reword the Status and remove/relabel the module-level function signatures (generate_audio_from_gnn, apply_audio_effects) as planned/not-yet-implemented.
2. doc/pedalboard/README.md:37 — "Files: 3 | Subdirectories: 1" -> Files: 5 (AGENTS.md, README.md, SPEC.md, pedalboard_gnn.md, pedalboard.md), Subdirectories: 0
3. doc/sapf/README.md:38 — "Files: 3 | Subdirectories: 0" -> Files: 5, Subdirectories: 0
4. doc/sapf/AGENTS.md metadata "Files: 3 | Subdirectories: 1" -> Files: 5, Subdirectories: 0
5. doc/audio/AGENTS.md:14 — SAPF acronym expansion: pick ONE canonical from src/audio ("Synthetic Audio Processing Framework" per src/audio/README.md, or "Structured Audio Processing Format" per src/audio/AGENTS.md); reconcile doc/audio to match one of these (do not use "Sonified Active Inference Parameter Framework").
6. doc/audio/README.md:15,36 — "Files 1 | Subdirectories 0" -> Files: 3 (README.md, AGENTS.md, SPEC.md)
7. doc/other/README.md:38 — "Files 10+ | Subdirectories 0" -> actual ~43 files / 19 subdirs (count via ls) — update
8. doc/other/README.md:28,91,111 — self-referential "Archive ../other/README.md" pointing to same file — repoint to a real sibling or drop the entry


## Verification
After editing, run:
- uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write   (must stay green)
- uv run --extra dev python scripts/check_repo_terminology.py --strict   (must stay clean — do NOT use banned words: legacy/stub/placeholder/deprecated)
- uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict

## Report
Write a concise report to /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation/.agents/dispatch/gnn-docfix/REPORT-fix-F-audio.md listing each file you changed and the specific
edit(s) applied. Reply with only the absolute path to your report.
