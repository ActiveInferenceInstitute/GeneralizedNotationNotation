# GUI-Audio-Integration — GNN swarm-2 scope

Repo: /home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation
YOU OWN these paths ONLY (disjoint scope — no other agent touches them):
- src/gui/gui_2/  (ui.py + helpers)
- src/gui/gui_3/  (ui_designer.py)
- src/audio/  (processor + sapf/)  [audio DSP edge cases]
- mirror tests: src/tests/gui/, src/tests/audio/

DO NOT TOUCH anything outside this scope. In particular NEVER edit:
- pyproject.toml, justfile, uv.lock, pytest.ini, .gitignore, AGENTS.md,
  CLAUDE.md, README.md, CHANGELOG.md, src/main.py
- src/tests/conftest.py, src/tests/helpers/, src/tests/categories.py
- files owned by other agents (disjoint paths).

GOAL
Harden and deepen these interactive/DSP modules:
1. GUI (gui_2 and gui_3): exercise the underlying model-building/validation logic
   (not the live GUI event loop) on real GNN specs; fix crashes on unusual or
   empty models; ensure the callback logic is total.
2. Audio/sapf: continue hardening DSP edge cases (empty, short, stereo, NaN
   inputs) and dependency-free WAV fallback; add regression tests.
Fix root causes; add targeted regression tests. Do NOT launch live GUI windows.

Drive shallow/tactical through deep/strategic improvements. Do NOT pad — every
change must earn its keep. Fix root causes, not symptoms. Add targeted
regression tests for any behaviour you pin or bug you fix. Keep public API
stable unless clearly justified.

VERIFY (scoped only — do NOT run the full suite):
- uv run ruff check src/gui src/audio
- uv run pytest src/tests/gui src/tests/audio -q --tb=no -x
- uv run mypy src/gui src/audio --config-file pyproject.toml

HARD RULE: DO NOT commit, DO NOT push, DO NOT stage. Leave ALL changes
uncommitted. Do not touch git index/HEAD. Other agents work on disjoint paths;
ignore files you do not own even if they look unfinished.

## Finish
Write a concise report to {report}
Summarize exactly what you changed (files + nature), tests added/passing,
and scoped verification results. Reply with only the absolute path to your report.

