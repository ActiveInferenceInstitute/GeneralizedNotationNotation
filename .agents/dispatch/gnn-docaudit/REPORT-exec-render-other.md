# REPORT — exec-render-other (REPORT-ONLY audit)

Assigned region: doc/quickstart.md, doc/START_HERE.md, doc/INDEX.md, doc/HANDOFF.md,
doc/DOCS_TO_IMPROVE.md, doc/QUICK_REFERENCE.md, doc/learning_paths.md,
doc/single_sentence_overview.md, doc/README.md, doc/CHANGELOG.md, doc/active_inference/,
doc/cognitive_phenomena/, doc/dependencies/.

Method: every documented command and path in the region was verified by `git ls-files
--error-unmatch` (tracked source = authority), grep of `src/`, and reading the live parser
(`src/utils/arg_parsing.py`), registries (`src/render/framework_registry.py`,
`src/execute/executor.py`, `src/execute/processor.py`), and CLI (`src/cli/`).
Verdict: region is largely CLEAN — 55+ doc paths, 40+ src/dir paths, and the whole `gnn`
CLI surface resolved. Three findings (2 WARNING, 1 INFO). No ERRORs, no fabricated or
broken paths found.

---

## Findings

### 1. doc/quickstart.md:74-75 | WARNING
States "there is no generic main-pipeline `--config`, `--profile`, or `--dry-run` option."
`--config` and `--dry-run` are genuinely absent, but `--profile` IS registered and wired into
the main parser. Evidence: `src/utils/arg_parsing.py:108` defines
`"profile": ArgumentDefinition(flag="--profile", action="store_true", help="Enable performance profiling")`,
and `create_main_parser()` (arg_parsing.py:611) adds ALL `ARGUMENT_DEFINITIONS` including
`profile`. So `uv run python src/main.py --profile` is accepted as a performance-profiling
toggle, contradicting the doc's literal "no `--profile`". The doc's intended meaning (no
config/profile *selection* system) is correct, but the wording is misleading.
Fix: reword to "no generic `--config`/`--dry-run`; `--profile` exists only as a performance-
profiling toggle, not a config-profile selector." (doc/README.md:103 already phrases it as
"not a generic `--config` or profile system" — accurate as written; no change needed there.)

### 2. doc/dependencies/OPTIONAL_DEPENDENCIES.md:14,262,288 | WARNING
ActiveInference.jl is documented to install via runtime `Pkg.add("ActiveInference")`.
But the repo ships a committed environment for it — `src/execute/activeinference_jl/Project.toml`
and `Manifest.toml` are tracked, and the runner (`activeinference_runner.py:73,136,319`)
invokes `--project=` plus `Pkg.instantiate()`, NOT `Pkg.add`. The same document explicitly
tells RxInfer users to "instantiate the committed env" with no runtime `Pkg.add` (lines
127,259,285). Treating ActiveInference.jl with `Pkg.add` is inconsistent with both the actual
code and the doc's own RxInfer guidance.
Fix: replace the three `Pkg.add("ActiveInference")` commands with
`julia --startup-file=no --project=src/execute/activeinference_jl -e 'using Pkg; Pkg.instantiate()'`
(matching the RxInfer committed-env pattern).

### 3. doc/active_inference/README.md:225 + doc/active_inference/AGENTS.md:21 | INFO
Both count the directory as 15 files ("Total Files: 15" / "Files: 15"), but `git ls-files
doc/active_inference/` returns 16 tracked `.md` files. The embedded tree (active_inference/README.md:200-224)
lists README + AGENTS + 13 content docs (15 entries) but omits `SPEC.md`, which exists.
Minor count/coverage drift; cosmetic.
Fix: add `SPEC.md` to the tree and bump the count to 16.

---

## Historical / self-labeled snapshots (no change)
- doc/HANDOFF.md opens as "Superseded snapshot" and records a point-in-time 2026-07-30 pass
  (tests passing count, commit `9b7ed48`, doc pages 610). Its commands and referenced files
  (src/tests/runner.py, src/llm/providers/base_provider.py, scripts, TO-DO.md, CLAUDE.md) all
  resolve and are well-formed; counts are historical by design (the doc clears to
  CHANGELOG.md / TO-DO.md for current state). Intentionally not a finding.

---

## Verified clean (no action / no errors)
- All CLI commands in doc/README.md and doc/quickstart.md: `uv run gnn preflight`, `gnn health`,
  `gnn validate input/gnn_files/discrete/actinf_pomdp_agent.md --strict`, `gnn templates list`,
  `gnn templates show actinf-pomdp-2state`, `gnn pull actinf-pomdp-2state --output-dir`,
  `uv run python src/main.py`, `uv run python src/5_type_checker.py`, `python -m json.tool`.
  `gnn` is registered as `src.cli:main` in `pyproject.toml`. The 15-subcommand dispatcher
  (incl. validate/preflight/health/templates/pull) exists in `src/cli/__init__.py` and
  `src/cli/templates.py`. Template `actinf-pomdp-2state` exists in `src/cli/template_assets/`.
- `input/gnn_files/discrete/actinf_pomdp_agent.md` is tracked and contains all enforced
  sections (GNNSection, GNNVersionAndFlags, ModelName, StateSpaceBlock, Connections, plus
  InitialParameterization/Equations/Time/Footer/Signature) — matches quickstart.md:37-47.
- 9 render-targets claim: `src/render/framework_registry.py` has exactly 9 keys (pymdp,
  activeinference_jl, jax, discopy, pytorch, numpyro, stan, bnlearn).
- 8 executor-families claim: `src/execute/processor.py::parse_frameworks_parameter` returns
  8 families (pymdp, jax, discopy, rxinfer, active_inf.jl, pytorch, numpyro, bnlearn).
- `00_pipeline_summary/pipeline_execution_summary.json` naming matches `src/pipeline/context.py:215-216`.
- `--only-steps --skip-steps --frameworks --target-dir --output-dir` are real args
  (src/utils/arg_parsing.py); `input/config.yaml` exists as the auto-loaded config described.
- Every cross-referenced doc path in INDEX.md, START_HERE.md, README.md, learning_paths.md,
  single_sentence_overview.md, QUICK_REFERENCE.md resolves via `git ls-files` — including all
  61 per-folder "Key File" links in single_sentence_overview.md (arc-agi, autogenlib, axiom,
  cerebrum, d2, dspy, glowstick, iroh, kit, klong, muscle-mem, nock, ntqr, onefilellm,
  pedalboard, poe-world, quadray, timep, vec2text, x402, ...); all tracked.
- doc/CHANGELOG.md's `doc/security/codex_security_remediation_2026-06-24.md` exists.
  DOCS_TO_IMPROVE.md's `docs_audit.py` flags (`--strict`, `--check-anchors`, `--no-write`)
  and `_doc_path_is_generated_dump()` all exist; `expected_dirs.txt`, `SPEC.md`,
  `style_guide.md`, `CROSS_REFERENCE_INDEX.md`, `doc/SETUP.md` + its
  `framework-selection-strategies` anchor all resolve.
- cognitive_phenomena "40+ phenomena": 47 `## ModelName` blocks across tracked `.md` files —
  claim supported. Its AGENTS.md "11 subdirectories" matches the 11 real subdirs.
- dependencies/* claims: RxInfer "no runtime Pkg.add" guidance matches the committed
  `src/execute/rxinfer/` Project.toml env.