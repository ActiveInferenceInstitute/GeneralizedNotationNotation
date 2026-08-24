# Research, LLM, Audio, ML, Security, Report, Ontology, Intelligent-Analysis Scope — mission-science.md

You own these paths ONLY within the GNN repo at
`/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation`:

- src/ontology/
- src/llm/  (LLM-enhanced analysis; Ollama default `smollm2:135m-instruct-q4_K_S`)
- src/audio/
- src/ml_integration/
- src/security/
- src/report/
- src/intelligent_analysis/
- src/research/
- mirror tests: src/tests/{ontology,llm,audio,ml_integration,security,report,intelligent_analysis,research}/

DO NOT touch outside your scope (shared no-touch list in mission-parse.md).

GOAL (shallow→deep):
1. Ontology/Active Inference term handling and reasoning paths.
2. LLM: make the analysis path degrade gracefully when Ollama is absent
   (allowed in CI) and use it genuinely when present. Do not break the
   two allowlisted Ollama tests.
3. Audio/sonification: correct DSP edge cases, fallback when backend
   missing.
4. Security: validation/sanitization, access checks, threat-policy
   clarity.
5. Report + intelligent_analysis: aggregated executive reports must be
   complete, deterministic, and not hallucinate results they lack.
6. Research tooling robustness.

VERIFY (scoped):
- `uv run ruff check src/ontology src/llm src/audio src/ml_integration src/security src/report src/intelligent_analysis src/research`
- `uv run ruff format --check ` (same tree)
- `uv run pytest src/tests/ontology src/tests/llm src/tests/audio src/tests/ml_integration src/tests/security src/tests/report src/tests/intelligent_analysis src/tests/research -q --tb=no -x` (Ollama tests already allowlisted — do not run them)

HARD RULE: leave ALL changes uncommitted; no commit/push/stage.

## Finish
Write a concise report to
`/home/trim/Documents/Git/HumOS/projects/outside_of_hum/GeneralizedNotationNotation/.agents/dispatch/gnn-improve/REPORT-science.md`
Reply with only the report's absolute path.