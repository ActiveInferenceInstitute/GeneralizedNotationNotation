# TO-DO - GNN Pipeline Roadmap

**Last Updated**: 2026-09-07 (wave 2: MAJ-07 closed - both pipeline utilities kept and wired with tests; setup_step_logging delegates + migration fossil retired; local-gates CI workflow added; dependency floors raised)
**Current Version**: 3.3.0
**Next Target**: v4.0.0 (bounded autonomy, pipeline stage consolidation, multi-agent stigmergic topologies, and high-dimensional active inference)

**Recently closed** (audit trail in `CHANGELOG.md` and git history, not here):
MAJ-02 (sparse Kronecker factorized execution + scaling sweep + numbered-pipeline
integration) and MAJ-03 (native stigmergic multi-agent compilation with
env-conditioned action selection; probe:
`uv run pytest tests/render/test_stigmergic_multi_agent.py -q`). The 3.2.0
release receipt (tests, mypy, ruff, documentation audits) is in `CHANGELOG.md`
§3.2.0.

GNN-02 (linear-Gaussian F/control/H/Q/R export:
`src/gnn/export/geo_infer_gaussian.py`, `tests/export/test_geo_infer_gaussian.py`,
paired analytic verification in `docs/development/geo_infer_2026_09.md`) and
GNN-03 (factor/modal dependency axes and multi-step policy enumeration:
`src/gnn/export/geo_infer_factored.py`, `tests/export/test_geo_infer_factored.py`)
closed 2026-09-07 after re-verification against the 3.3.0 tree.

MAJ-07 closed 2026-09-07: both pipeline utilities are KEPT and wired with
direct tests - they are documented public API (`gnn.pipeline.__all__` +
`pipeline/SKILL.md` usage examples) and `pipeline/pipeline_validator.py` is
the health check's live integration probe. Fixes landed while wiring: the
core-dependency check imported PyYAML by distribution name (`pyyaml` vs
`yaml`, so core deps always reported unhealthy), scipy/pathlib were listed
as core (scipy moved to the ml-ai extra in 3.3.0), the runtime validator
shelled out to the retired pre-restructure orchestrator (`main.py` at the
`src/` root; now `src/gnn/main.py`), and its import fallback could raise
NameError. Tests: `tests/pipeline/test_health_check.py` and
`tests/pipeline/test_pipeline_validator.py`.

MAJ-04 closed 2026-09-07: all six >2000-line modules decomposed via the
3.3.0 `execute/processor.py` split pattern (mechanical sibling extraction,
facade re-exports preserved, one module per PR) — `analysis/visualizations.py`
2412→58 (PR #29), `analysis/analyzer.py` 2031→263 (PR #32),
`render/jax/jax_renderer.py` 2200→170 (PR #33), `render/discopy/translator.py`
2150→303 (PR #34), `integration/meta_analysis/visualizer.py` 2871→283 (PR #40,
`Sweep*Mixin` variant preserving byte-identical class-method moves), and
`testing/test_round_trip.py` 2214→1356 (PR #43, `round_trip_*` siblings).
Class-method bodies moved verbatim into mixins where module-level extraction
was impossible. Every facade re-exports every moved name (no consumer
import-path changes; per-module facade-contract probes 31/34/27/36/13/47
names), moved code verified byte-identical modulo imports
(2347/2001/2190/2108/2811/2172 lines), full suite green at every PR
(4263→4272 passed), mypy/ruff 0 throughout. Session benchmark
`oversized_module_lines` 13878 → 0: no tracked `src/gnn` Python file exceeds
2000 lines (`gnn_python_lines` +0.6% across the series — code moved, not
deleted). The row's "shared subprocess envelope the nine per-framework
renderers duplicate" item is RESCOPED to its own future work: renderers
contain zero subprocess code (verified); the duplication is execute-side
(`rxinfer`/`stan`/`lean`/`activeinference` runners + 4 `executor.py` MCP
methods vs the canonical `execute_script_safely` at `execute/executor.py:1089-1200`)
and needs a behavior-preserving refactor with its own tests, not a
mechanical split.

## Open Scoped Roadmap

Every item below is cold-startable: scope, files, verification, and acceptance
are pinned.

| ID | Scope | Acceptance evidence |
| --- | --- | --- |


### Smaller scoped cleanups (independent of the majors)

- ruff F401/F811 policy ignore: RESOLVED 2026-09-07 - the global
  `F401`/`F811` ignore entries are gone from pyproject; nine genuine
  re-export surfaces (six MAJ-04 facades, `round_trip_availability`,
  `visualizer_style`, `execute/processor.py`) and the `src/gnn/parsers/*`
  guarded optional-backend probes hold documented per-file-ignores, and
  66 genuinely dead imports were removed (57 src/gnn, 10 scripts, 9 F811
  re-imports). `ruff --select F401,F811 src/gnn` now reports 0 findings;
  `ruff check src/gnn scripts`, mypy, and the full suite stayed green
  (4285 passed). Consumer safety: AST-resolved `from <module> import`
  scan across src/gnn, tests, and scripts against every removed name.
- Local/CI parity: tokens and skills-health are CI-wired via
  `.github/workflows/local-gates.yml` (2026-09-07; `skills-health` also
  needed a repo-root sys.path bootstrap). `just gridworld` remains
  unwired deliberately - the committed `output/` tree currently fails
  its contract and regeneration needs the Julia toolchains.
  ml-ai/torch extras parity: RESOLVED 2026-09-08 - verified
  `uv sync --extra dev --extra ml-ai --extra torch --frozen` resolves from
  the lock and un-skips the 12 environment-skipped tests (11 sklearn
  inference tests, 1 torch continuous-render test); all 22 tests in the two
  affected files pass with the extras present (no latent failures behind the
  skip). The local test-cov command should therefore run with
  `--extra ml-ai --extra torch` appended. The Ollama-ignore half of this
  item stays open-by-design: no local Ollama daemon exists, so
  `test-cov`'s `--ignore=tests/llm/test_llm_ollama*.py` remains correct
  locally while the CI coverage run exercises those tests where they
  degrade gracefully without a daemon. Coverage selection parity on the
  remaining axis: `just test-cov` now adopts CI's
  `-m "not pipeline and not mcp"` deselect so both invocations apply the
  same pipeline/mcp test policy (4326 collected locally; CI collects
  4352 - the 26-test Ollama delta is the open-by-design asymmetry
  recorded above).
- Dependency floors: RAISED 2026-09-07 for numpy (>=2.0), pandas
  (>=2.0), openai (>=2.0), pytest (>=8.0), mypy (>=1.0) - the lock
  resolved identically (only requires-dist metadata moved; zero package
  pins changed). Remaining cosmetic floors (networkx 2.6, plotly 5.15,
  scipy 1.7, ...) can follow at the next deliberate lock refresh.
- `gnn/utils/pipeline_validator.py` vs `gnn/pipeline/pipeline_validator.py`
  near-name collision: RESOLVED 2026-09-08 — the lower-traffic runtime
  integration tester renamed to
  `gnn/pipeline/pipeline_runtime_validator.py` (compatibility module at the
  old path emits `DeprecationWarning` and re-exports `PipelineValidator`/`main`;
  contract pinned in `tests/pipeline/test_pipeline_runtime_validator.py`);
  import-site grep has zero stragglers.
- Stale singular module paths in maintained docs: RESOLVED 2026-09-08 —
  all 21 occurrences (19 lines) of `src/gnn/parser.py`, `src/gnn/schema.py`,
  and `src/gnn/schema_validator.py` re-pointed to their verified real homes
  (`schema/parser.py`, `schema_validator/syntax.py`, `parsers/system.py`);
  regression gate `scripts/check_doc_path_references.py` is CI-wired
  (local-gates) and strict (cap 0).

## Deep horizon wave 2 - render backends

Scoping for `src/gnn/render/**` (2026-09-08, deep-horizon session). RB-01
through RB-07 are RESOLVED (see `CHANGELOG.md`): the deterministic corpus x
framework conformance benchmark (`bash autoresearch.sh`;
`scripts/bench_render_backends.py`) drove contract modernization to the
maintained delegated-executor output shapes, and the conformance-validated
rendering count moved 142 -> 258 of 258 (0 contract violations, 0 syntax
errors, 0 render errors, 12 by-design unsupported). Remaining open surface:

| ID | Scope | Acceptance evidence |
| --- | --- | --- |
| RB-08 | Minor: `src/gnn/render/rxinfer/toml_generator.py` (46KB) is production-dead (the `rxinfer_toml` target is rejected in `render_gnn_spec` and the CLI choice was removed) but stays importable via two contract-test files and `scripts/check_capability_contracts.py` text markers. Decide: migrate the still-used matrix parsers (`_parse_gnn_matrix`, `_parse_gnn_3d_matrix`, `_parse_gnn_vector`) and topology-structure helpers to a live module (or fold them into the rxinfer renderer), then delete the retired emitter and update the three consumers. | Import-site grep for `toml_generator` returns zero production references; the two test files pin the migrated home; `check_capability_contracts.py` passes against the new location; no render/execute behavior change on the bench. |
| RB-09 | Minor: generator-facade status decision for `src/gnn/render/generators.py` public exports `generate_rxinfer_code` / `generate_activeinference_jl_code` (exported and README-documented but production routes use the dedicated renderers) and the legacy `src/gnn/render/pymdp_template.py` template (alive only through `generate_pymdp_code`, whose only production caller is the non-POMDP basic fallback). Either document the facade as the supported public surface with tests that pin it, or retire the exports with a deprecation window. | Decision recorded here with evidence; either facade tests pin the exports end to end, or a deprecation alias emits `DeprecationWarning` and consumer grep shows zero stragglers. |

---


## Deep horizon wave 2 - MCP + execute

Scope from a six-lens read-only audit (registry, dispatcher/serialization,
subprocess envelope, per-framework executors, resources/docs, test gaps) of
`src/gnn/mcp/**` and `src/gnn/execute/**`. Session benchmark:
`bash autoresearch.sh` (`scripts/run_autoresearch_bench.py`, primary metric
`mcp_execute_bench_ms`, 952 pinned determinism checks) — every row below
must leave it green.

| ID | Sev | Scope | Acceptance evidence |
| --- | --- | --- | --- |
| MAJ-08 | major | MCP transport serialization is strict on the wire, tolerant in the cache, and drops failures silently: tool results returning sets/bytes/datetime/numpy scalars crash the stdio writer thread (swallowed TypeError, client hangs forever — no request timeout exists) and abort the HTTP response; NaN/Inf floats emit bare `NaN`/`Infinity` tokens (invalid JSON, RFC 8259) on all three transports (`server_stdio.py:225-239`, `server_http.py:416-423`, `server_core.py:174-178`); cache-key dump uses `default=str` while wire dump is strict (`mcp.py:1030-1043`), so unserializable-param calls re-execute then fail at write time and distinct params can alias to one cache key; error-path data itself can be unserializable (`MCPValidationError.raw` stored verbatim, `exceptions.py:137-140`). Fix: one shared `serialize_response` helper (allow_nan=False, NaN/Inf sanitizer, default=str fallback) used by all transports and by error-envelope construction. | Transport-parametrized tests in `tests/mcp/`: a tool returning set/bytes/datetime/NaN yields a wire-valid -32603 envelope on stdio + HTTP + server_core (no hang, no abort); a set-shaped param yields a wire-valid -32602 error; cached and live results serialize identically. |
| MAJ-09 | major | `MCPTool.timeout` is accepted at registration, advertised in capabilities, documented (`MCP_DOCUMENTATION.md:299,677`), and `StdioServer(request_timeout=30.0)` accepts the arg — but no execution path consults either: `execute_tool` runs `tool.func(**params)` synchronously (`mcp.py:977`), so a hung tool stalls the worker forever and queued requests never get any response. Fix: enforce tool timeout via the existing executor (`future.result(timeout=...)`) emitting a reserved JSON-RPC timeout code, or remove the field from capabilities. | A tool sleeping > timeout returns a timeout error envelope within timeout+ε on server_core and stdio; contract pinned by a test. |
| MAJ-10 | major | Step-12 processor bypasses the canonical envelope: raw `subprocess.run` at `execute/processor.py:1167-1174` with hand-rolled timeout/OSError handling despite `subprocess_envelope.py:5-7` claiming universal coverage; exit-code vocabulary collides (-1 = timeout AND OSError AND never-started AND executor-unavailable; processor adds -2; lean renames the key to `returncode`, `lean_runner.py:109`); on timeout processor writes literal `stderr="Timeout"` discarding the partial output the envelope would keep (`processor.py:1193` vs `subprocess_envelope.py:85-86`). Fix: shared exit-code constants module (NEVER_STARTED/-1, INTERNAL_ERROR/-2) used by envelope + processor + lean; processor per-script execution calls `run_subprocess_envelope` and derives `error_type` from the envelope; `execute/sandbox.py:171` delegates its inline envelope re-implementation; `execute/julia_setup.py:123` setup run migrates. | Processor timeout test asserts partial stderr retained + `error_type == "TimeoutExpired"`; sandbox/julia_setup results envelope-shaped; full suite green; `determinism_checks` unchanged at 952. |
| MED-01 | medium | Param-validation fidelity: with default non-strict validation, wrong/extra/missing kwargs surface as -32603 "Internal error" with raw `str(TypeError)` leaked to the client (`mcp.py:977`, `mcp.py:1004-1014`, `server_core.py:160-165`) instead of -32602 INVALID_PARAMS; `_validate_output` rejects a `None` return with -32602 (params were valid — wrong class of failure, `mcp.py:1592-1595`); non-strict mode silently skips all schema constraints while tools advertise them (`mcp.py:1318-1330`). | Tests: wrong/missing args → -32602 with tool name; unexpected exceptions redacted (generic message server-side, no `str(e)` on the wire); None-return documented distinct code; capabilities expose a validation-mode hint. |
| MED-02 | medium | MCP resources: `list_available_resources` re-exported from `mcp/__init__.py:36-37` aliases `get_available_tools` — the public API returns TOOLS labeled resources (`SKILL.md:63-66` sends agents down this path); the real lister (`mcp.py:1953`) is not re-exported; HTTP capability filter matches `uri_template` while the read gate matches exact concrete URIs — no config exposes both for the only registered resource `gnn://documentation/{doc_name}` (`server_http.py:184-206` vs `:292-307`); README.md:259 maps resource retrieval errors to -32002 but not-found raises -32601 (`exceptions.py:64`); AGENTS.md:141-144 documents wrong `MCPResource` fields; docs/mcp/README.md:22-33 tree shows nonexistent `src/mcp/`; `npx_inspector.py:158-171` `get_resource` sends a URI as a JSON-RPC method ("This is a guess") and can never work. | Real `list_available_resources` re-exported and wired in SKILL.md; HTTP capability listing and read gate agree on `gnn://documentation/{doc_name}`; error-code table, resource fields, and module tree corrected; npx_inspector dead method deleted; `check_mcp_skills_health.py --strict` green. |
| MED-03 | medium | Executor timeout/classification divergence: pymdp timeout hard-coded 600 not parameterizable (`pymdp_runner.py:153`) while `GNNExecutor` defaults pymdp to 60s (`executor.py:335`) and AGENTS.md:101 documents `execute_script_safely` default 3600 vs implemented 60 (`executor.py:1032`); rxinfer TOML branch runs julia without `--project=` (`rxinfer_runner.py:118`) so RxInfer may not resolve, and rxinfer persists zero execution evidence (no stdout/stderr/log files, unlike `jax_runner.py:201-232`); pymdp rewrites rendered scripts in place (destroys the Step-11 audit trail, `pymdp_runner.py:28-73`); lean leaks `mkdtemp("gnn-lean-verify-")` dirs (`lean_runner.py:80-83`) and classifies missing toolchain as failure while its README promises skip; rxinfer same skip/fail divergence (`rxinfer_runner.py:150-152` vs stan's structured `skipped:True+reason` shape, `stan_runner.py:87-95`). | `execute_pymdp_script_with_outputs(..., timeout=...)` plumbed through; executor default aligned to 3600 + doc matches; rxinfer TOML adds `--project` and persists artifacts; pymdp rendered file byte-identical post-run (cleaning to `.cleaned.py`); lean uses TemporaryDirectory; missing-toolchain runs return skipped records in stan's shape — all pinned by tests/execute tests. |
| MED-04 | medium | No request-size limits: HTTP reads Content-Length unbounded (`server_http.py:249-252`), a garbage header raises uncaught ValueError aborting the connection without a response, stdio reads unbounded lines (`server_stdio.py:139`), and `tools/call` embeds `json.dumps(result, indent=2)` with no response-size policy (`server_core.py:178`). | Oversize-body and malformed-header tests return protocol-valid error envelopes (413/400-class), no traceback/abort; documented response-size policy for large matrix results. |
| MIN-01 | minor | Registry/transport hygiene batch: dead duplicate-registration check in `MCP.register_tool` (`mcp.py:724-732` — a no-op read, duplicate names silently overwrite); dead -32700/non-dict branch in `server_core.handle_request` (`server_core.py:122-126`, unreachable) and -32602 message divergence from the shared helper (`server_core.py:99-113`); result cache stores results by reference so a mutating caller poisons future hits (`mcp.py:993-996`, `:958-961`); `ensure_ascii` differs per transport (stdio False `server_stdio.py:226`, HTTP True `server_http.py:423`) breaking golden-file equality; batch requests rejected -32600 without a documented single-request contract (`jsonrpc.py:24-33`). | Unit tests: cache-hit immutability, duplicate registration warning; unified serializer flags; contract sentence in `model_context_protocol.md`; ruff/mypy clean. |
| MIN-02 | minor | Test-gap batch (highest-regression-risk untested behaviors): `requires_auth` tool gate (`mcp.py:891-899`), registry-level non-dict params (`mcp.py:901-912`), in-process result cache hit/TTL/uncacheable-params (fixtures always disable caching, `mcp.py:968-984`, `:1029-1033`), per-tool sliding-window rate limiter (`mcp.py:1058-1076`), output validation (`mcp.py:989-990`), envelope timeout partial-stdout retention (test asserts only error_type, `test_subprocess_envelope.py:42-49`), `audit_report.json tools_total` vs live registered count parity (`test_mcp_audit.py:546-548` checks only >= 50), `execute/validator.py` 610 lines with zero branch coverage. | Each listed behavior has a failing-on-regression test in `tests/mcp/test_mcp_functional.py` / `tests/execute/`; audit JSON parity test added. |
| MIN-03 | minor | Envelope `input=` support + audit coverage: `run_subprocess_envelope` lacks stdin support, forcing raw bypasses in `llm/providers/ollama_provider.py:167`, `security/processor.py:1037`, `manuscript/variables.py:179`; `validate_tools.py` spot-checks only 14 of 141 registered tools' schema-vs-signature (`validate_tools.py:129`, `audit_report.json` spot_checks_ok: 14). | Envelope `input=` parameter with test (stdin-consuming child); audit regenerates `spot_checks_ok: 141, issues: []`. |

## v4.0.0 - Bounded Autonomy & Reviewed Self-Editing

The local bounded-autonomy surface emits proposal-only artifacts via
`--autonomous`: candidate scores, review gates, rollback descriptors, audit
events, and non-mutating security policy. No source edit, commit, container
run, or cluster mutation is automatic.

Concrete, cold-startable v4.0.0 work is scoped in the Open Scoped Roadmap
table above; this section records the unscoped vision and the current
proposal-only `--autonomous` surface.

---

## Verification Commands

Use `uv run` for roadmap verification checks:

```bash
PYTHONPATH=src uv run python scripts/run_v3_orchestration_acceptance.py --strict
PYTHONPATH=src uv run python scripts/emit_run_manifest.py output --out /tmp/gnn-v3-run-manifest
PYTHONPATH=src uv run python scripts/generate_pipeline_container_plan.py --config input/config.yaml --out /tmp/gnn-v3-container-plan.json
PYTHONPATH=src uv run python scripts/run_session_acceptance.py --manifest input/model_family_manifest.json --output-dir /tmp/gnn-v3-session-acceptance --session /tmp/gnn-v3-session.json --strict
PYTHONPATH=src uv run python src/gnn/main.py --autonomous --target-dir input/gnn_files --output-dir /tmp/gnn-autonomous-smoke

uv run python docs/development/docs_audit.py --strict --check-anchors --no-write
uv run python scripts/check_gnn_doc_patterns.py --strict
uv run python scripts/check_maintained_doc_terms.py --strict
uv run python scripts/check_repo_terminology.py --strict
uv run python scripts/check_doc_path_references.py
uv run python scripts/check_capability_contracts.py
uv run python scripts/run_semantic_fidelity_gate.py --output-dir /tmp/semantic_fidelity --strict
uv run python scripts/run_cross_framework_reliability.py --output-dir /tmp/cross_framework --strict
git diff --check
```

## Conventions

- Keep this file limited to unchecked, forward-looking work.
- Move shipped-version details to release notes, changelog entries, or durable
  verification artifacts.
- Keep closed work out of this file: completed items are removed when they
  land; the audit trail lives in `CHANGELOG.md` and git history.
- Scope open items with concrete tasks, file paths, verification commands, and
  acceptance criteria so the next session can execute without re-deriving them.

## GEO-INFER contract expansion

The delivered opt-in v1 format is specified in `src/gnn/export/geo_infer_contract.md`.
Further work must preserve independently installable runtimes and explicit matrix,
space and time semantics.

| ID | Scope | Acceptance evidence |
| --- | --- | --- |
| GNN-04 | Pin paired repository revisions in cross-repository CI on the GNN side; the GEO side already hosts paired CI retaining both revisions plus categorical/H3/Gaussian/factored digests (`docs/development/geo_infer_2026_09.md`), and `.github/` has no GNN-side equivalent. | A GNN-side workflow (or documented receipt-pinning procedure) completes paired categorical and H3 round trips and records source/artifact digests for both revisions. |
| GNN-05 | Notation-driven metadata discovery for GEO-INFER export: derive step seconds/units/space kind from the GNN notation instead of explicit user JSON. The explicit-CLI wiring and original-source provenance already landed (`src/gnn/7_export.py`, `src/gnn/export/processor.py`, `tests/export/test_export_geo_pipeline.py`, `tests/export/test_geo_infer_gaussian.py`). | Notation-derived metadata passes the same visible-failure and unchanged-five-format-default tests that pin the explicit path. |
