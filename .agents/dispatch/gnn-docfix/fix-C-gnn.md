# C-gnn — GNN documentation accuracy fixes (SAFE DOC-ONLY EDITS)

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
1. doc/gnn/modules/11_render.md:296,309 — `from render.renderer import render_gnn_spec` -> `from render import render_gnn_spec`
2. doc/gnn/modules/01_setup.md:225,232,243,250,261 — `from setup.setup import setup_uv_environment` (and add_uv_dependency/remove_uv_dependency/update_uv_dependencies/lock_uv_dependencies) -> `from setup import setup_uv_environment` (etc., package root)
3. doc/gnn/operations/REPO_COHERENCE_CHECK.md:221,225,127 — `from visualization import process_visualization_main` -> `from visualization import process_visualization`
4. doc/gnn/operations/improvement_analysis.md:159 — `from visualization import process_visualization_main` -> `from visualization import process_visualization`
5. doc/gnn/testing/test_patterns.md:44 — `from audio import backends` -> `from audio import check_audio_backends` (note it returns the backends dict)
6. doc/pymdp/pymdp_pomdp/INTEGRATION_SUMMARY.md:194 — `from src.execute.pymdp import batch_execute_pymdp` -> `from src.execute.pymdp.execute_pymdp import batch_execute_pymdp`
7. doc/gnn/reference/architecture_reference.md:125 — `input/gnn_files/actinf_pomdp_agent.md` -> `input/gnn_files/discrete/actinf_pomdp_agent.md`
8. doc/gnn/modules/04_model_registry.md:224 — `"file_path": "input/gnn_files/actinf_pomdp_agent.md"` -> `input/gnn_files/discrete/actinf_pomdp_agent.md`


## Verification
After editing, run:
- uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write   (must stay green)
- uv run --extra dev python scripts/check_repo_terminology.py --strict   (must stay clean — do NOT use banned words: legacy/stub/placeholder/deprecated)
- uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict

## Report
Write a concise report to /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation/.agents/dispatch/gnn-docfix/REPORT-fix-C-gnn.md listing each file you changed and the specific
edit(s) applied. Reply with only the absolute path to your report.
