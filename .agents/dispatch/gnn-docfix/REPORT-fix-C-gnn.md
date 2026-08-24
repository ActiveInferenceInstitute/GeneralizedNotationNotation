# REPORT-fix-C-gnn

Charter: `.agents/dispatch/gnn-docfix/fix-C-gnn.md`
Scope: DOC-ONLY markdown edits (no .py, no config, no tests). Nothing was committed/staged/pushed.

## Files changed and edits applied

1. **doc/gnn/modules/11_render.md** (fix 1)
   - Lines 296, 309: `from render.renderer import render_gnn_spec` → `from render import render_gnn_spec` (both occurrences)

2. **doc/gnn/modules/01_setup.md** (fix 2)
   - Line 225: `from setup.setup import setup_uv_environment` → `from setup import setup_uv_environment`
   - Line 232: `from setup.setup import add_uv_dependency` → `from setup import add_uv_dependency`
   - Line 243: `from setup.setup import remove_uv_dependency` → `from setup import remove_uv_dependency`
   - Line 250: `from setup.setup import update_uv_dependencies` → `from setup import update_uv_dependencies`
   - Line 261: `from setup.setup import lock_uv_dependencies` → `from setup import lock_uv_dependencies`
   - Line 269 (sibling class fix): `from setup.setup import check_system_requirements` → `from setup import check_system_requirements`.
     The charter listed lines 225/232/243/250/261; line 269 carries the identical broken `from setup.setup import` pattern. `src/setup/__init__.py` exports `check_system_requirements` from the package root (and `src/setup/AGENTS.md` states "All setup helpers are exported from the package root. Prefer `from setup import …`"), so I fixed it as the same class of error.

3. **doc/gnn/operations/REPO_COHERENCE_CHECK.md** (fix 3)
   - Line 127: ``Delegates to `process_visualization_main()` from `visualization` module`` → ``Delegates to `process_visualization()` from `visualization` module``
   - Line 221: `from visualization import process_visualization_main` → `from visualization import process_visualization`
   - Line 225: `process_visualization_main,` → `process_visualization,` (call arg in `create_standardized_pipeline_script`)

4. **doc/gnn/operations/improvement_analysis.md** (fix 4)
   - Line 159: `from visualization import process_visualization_main` → `from visualization import process_visualization`

5. **doc/gnn/testing/test_patterns.md** (fix 5)
   - Line 44: `from audio import backends` → `from audio import check_audio_backends`
   - Line 46: `result = backends.check_backends()` → `result = check_audio_backends()` — `check_audio_backends` returns the backends dict directly (src/audio/__init__.py:63-99), so the CORRECT example was updated to call the real surface instead of the now-removed `backends` attribute.

6. **doc/pymdp/pymdp_pomdp/INTEGRATION_SUMMARY.md** (fix 6)
   - Line 194: `from src.execute.pymdp import batch_execute_pymdp` → `from src.execute.pymdp.execute_pymdp import batch_execute_pymdp`

7. **doc/gnn/reference/architecture_reference.md** (fix 7)
   - Line 125: `input/gnn_files/actinf_pomdp_agent.md` → `input/gnn_files/discrete/actinf_pomdp_agent.md`

8. **doc/gnn/modules/04_model_registry.md** (fix 8)
   - Line 224: `"file_path": "input/gnn_files/actinf_pomdp_agent.md"` → `"file_path": "input/gnn_files/discrete/actinf_pomdp_agent.md"`

## Verification (all green)
- `uv run --extra dev python doc/development/docs_audit.py --strict --check-anchors --no-write`
  → Broken links: 0, Bad markdown anchors: 0, AGENTS/SPEC gaps: 0 (exit 0)
- `uv run --extra dev python scripts/check_repo_terminology.py --strict`
  → "maintained tree clean."
- `uv run --extra dev python scripts/check_gnn_doc_patterns.py --strict`
  → "no banned patterns in doc/ and src/gnn/ (markdown)."

## Notes
- Only the 8 charter-named `.md` files were modified by this task. The working tree also contains unrelated pre-existing dirty files (e.g. SKILL.md, doc/api/README.md, doc/audio/AGENTS.md, etc.) that were modified before this session; these were left untouched.
- Nothing was committed, staged, or pushed (HARD RULE respected).