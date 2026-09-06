# GNN reliability lane receipt — 2026-09-04

## Scope and baseline

Implemented the approved reliability lane after parent GO. Baseline HEAD remains
`64d49355acf197a0570b06ab334d97570774be64`; source/hash baseline:
`/tmp/gnn-fep-implementation-20260904/GeneralizedNotationNotation-baseline.json`.
All initially assigned API/MCP/render-processor/execute-metadata/CI files matched
that snapshot at intake, including earlier dirty fleet changes. No commits or
pushes were performed. Validation/parsers/CLI and bridge/ISA ownership remained
with their assigned workers/parent; sibling FEP input was read only. Parent later
transferred `src/gnn/main.py` and narrow orchestration docs/tests for run-ID integration;
that integration and the requested test-infrastructure README are complete.

Parent attempted full baseline suites but interrupted stalled pre-existing native
fleets. The parent focused GNN baseline log
`/tmp/gnn-fep-gnn-focused-baseline.log` subsequently recorded a subprocess timeout.
This is not a clean full-suite baseline. No additional full suite or native build
was launched by this lane. Parent owns final full validation.

GitNexus did not index this nested checkout. Impact analysis used direct symbol,
import, registry, test, and documentation searches; graph confidence is reduced.

## Implemented contracts

- **API:** shared rc0/rc1/rc2 completion policy; rc2 fails under strict policy.
  Both request surfaces expose strict mode. Malformed summary roots are ignored.
  Step lists reject coercible booleans/strings/floats before Pydantic coercion.
  Active run DELETE returns 409 without removing the record. Deduplication includes
  normalized output paths, sorted effective skipped steps, and strict policy.
  Each subprocess receives a unique `GNN_RUN_ID`; summary ingestion requires the
  matching `run_id` and current-invocation modification time, preventing concurrent
  jobs from consuming one another's summary.
- **Orchestrator run identity:** top-level calls preserve incoming API IDs or create
  a unique UUID; `run_hash` remains stable content/config identity. Serial and
  parallel step dispatch pass explicit child environments. A reentrant lock
  serializes overlapping programmatic top-level calls, and `finally` restores
  the incoming environment on success, early return, failure, and interruption.
  Canonical full/minimal/failure summaries retain the ID and use atomic writes.
  Startup failures after argument resolution publish a minimal identified receipt;
  argument-parser failures before an output location is resolved remain log-only.
  Minimal recovery now returns failure rather than success after writing FAILED.
- **Render receipts:** canonical source path/content and configuration/run identity;
  scoped replacement on retry/removal; aggregates recomputed from surviving records.
  Only verified same-run/config records carry across sequential folder invocations.
  Source and artifact hashes are checked; prior valid receipts are archived under
  `history/render-<digest>.json`. Empty retries clear their scope. Atomic replacement
  publishes complete JSON and cleans temporary files on serialization failure.
- **Execute receipts:** snapshot source scope and scripts before dispatch, reject
  changed bytes before publication, verify identified render sources/artifacts at
  ingestion, and reject a mismatched explicit run ID. Legacy render receipts remain
  readable only without an explicit expected run ID; they offer no freshness proof.
  Same-run/config invocation records replace retries, preserve current-call verdicts,
  recompute counts/framework/global statuses, and archive prior receipts separately.
  Mixed successful/skipped scopes report success with skips rather than stale failure.
  Aggregation preserves classified failure reasons such as
  `requested_framework_execution_incomplete` instead of replacing them with the
  generic script failure reason; the parent integration failure is repaired.
  Slim/detail JSON publication uses atomic replacement; full in-memory details are
  restored even on publication failure.
- **MCP:** HTTP now uses the shared JSON-RPC envelope helpers. HTTP/stdio/core reject
  malformed requests/object parameters consistently; valid notifications receive no
  JSON-RPC response (HTTP 204), while explicit null IDs receive null-ID responses.
  Authentication, rate limits, and HTTP tool/resource allowlists remain intact.
- **Capabilities/CI:** registry metadata exposes `supports_execution`, false for
  bnlearn. Maintained README wording states bnlearn is render-only with no Step 12
  executor. CI explicitly runs functional MCP, HTTP, transport, capability registry,
  and documentation contract checks on Python 3.12, retaining JUnit evidence. Optional
  absence is reported explicitly; a tool-count threshold is not the functional gate.
- **FEP boundary regression:** `render_gnn_spec` publicly accepts both a parser mapping
  and `GNNInternalRepresentation`. Both reproduced the accepted FEP source's 1-state
  inference error from singleton-row D/E vectors. The renderer boundary now flattens
  singleton-row/column C/D/E vectors before inferring dimensions. The accepted source
  is embedded verbatim in `test_fep_bridge_render.py`, including provenance, so CI
  does not depend on a sibling checkout. Both forms now render parseable PyMDP code
  with two hidden states and the 0.25/0.75 policy prior. No parser or Lean code changed.

## Focused evidence

Commands used the existing environment via `uv run --offline --no-sync`, with
`UV_CACHE_DIR=/tmp/gnn-reliability-uv-cache` and, for renderer tests,
`MPLCONFIGDIR=/tmp/gnn-reliability-mpl`. This avoided dependency mutation during the
fleet run. A read-only copy of the existing font cache avoided repeated discovery.

| Probe | Measured result | Log |
| --- | --- | --- |
| Initial API lifecycle/summary/strict regressions | 12 failed, 1 passed before fixes | `/tmp/gnn-reliability-api-red.log` |
| HTTP byte-stream protocol regressions | 9 failed, 1 passed before fixes | `/tmp/gnn-reliability-mcp-red.log` |
| Receipt counter/freshness/root regressions | 8 failed before; 8 passed after | `/tmp/gnn-reliability-receipts-red.log`, `/tmp/gnn-reliability-receipts-green.log` |
| Additional artifact/coercion regressions | 5 failed before fixes | `/tmp/gnn-reliability-additional-red.log` |
| Cross-job API summary identity | failed before expected-run validation | `/tmp/gnn-reliability-api-identity-red.log` |
| FEP parser-mapping and typed-IR regressions | 2 failed before vector normalization | `/tmp/gnn-reliability-fep-red.log` |
| Expanded API/MCP/render/execute focused suite | 110 passed | `/tmp/gnn-reliability-final-focused.log` |
| Caller/capability/docs tests, including all 29 exemplar RxInfer code generations | 56 passed | `/tmp/gnn-reliability-caller-contracts.log` |
| Final FEP/API and adjacent canonical renderer regressions | 40 passed | `/tmp/gnn-reliability-last-regressions.log` |
| Top-level run identity regressions | 6 failed, 1 passed before implementation | `/tmp/gnn-reliability-run-id-red.log` |
| Final run-ID/API/pipeline helper regressions | 48 passed, including 10 run-ID cases | `/tmp/gnn-reliability-run-id-final.log` |
| Parent-discovered strict execution reason regression | 1 failed before repair; entire script-safety and execution-receipt files 26 passed after | `/tmp/gnn-reliability-outcome-red.log`, `/tmp/gnn-reliability-execute-compatibility.log` |
| Scoped Ruff | all checks passed, including main/run-ID additions | `/tmp/gnn-reliability-ruff-final.log`, `/tmp/gnn-reliability-ruff-run-id.log`, `/tmp/gnn-reliability-ruff-compatibility.log` |
| Scoped mypy, no incremental cache | 36 source files passed, including main/API/FEP additions | `/tmp/gnn-reliability-mypy-run-id.log`; final main/metadata recheck `/tmp/gnn-reliability-mypy-compatibility.log` |
| Final strict docs audit | zero links, anchors, AGENTS/README coverage or structure failures | `/tmp/gnn-reliability-docs-closure.log` |
| Capability audit | `Capability contracts verified` | `scripts/check_capability_contracts.py --strict` |
| Workflow lint | exit 0 | `actionlint .github/workflows/ci.yml` |

The passing test runs overlap; their counts must not be summed as unique tests.
The 110-test pass preceded the final per-API invocation-ID and FEP additions;
the 40-test run covers those additions and the final 48-test run covers run-ID
integration. Child identity tests execute only `python -c` probes, never numbered
pipeline/test scripts. Full parent gates remain separate.

The configured Cato startup declined because its ISA/advisor metadata was absent;
no Cato audit ran. A fresh-context read-only adversarial reviewer substituted,
identified concrete API/protocol/receipt issues, and rechecked their fixes. Its
last concurrency finding led to expected-run-ID API summary validation and a
focused regression. The final orchestrator review found overlapping in-process
environment scopes could share/leak IDs; serialized scopes and a controlled
overlap test address that finding. No external advisor verdict is claimed.

## Parent integration and limitations

1. **Run identity integration complete:** parent transferred the emitter scope;
   current top-level summaries are accepted by the API with an expected run ID,
   as verified against the real writer/reader and real probe subprocesses.
2. **Live HTTP socket:** this lane's sandbox denied localhost bind with
   `PermissionError: [Errno 1] Operation not permitted`; local tests exercised
   actual HTTP byte streams and the registry. Parent subsequently reported all
   12 live HTTP socket tests passed. This is parent-provided evidence, not a
   socket run performed by this lane; CI also runs the live HTTP tests.
3. **Docs audit complete:** the requested `tests/tests/README.md` pairs the
   parent-added AGENTS file. The final strict audit reports all zero failures.
4. **Publication/concurrency:** JSON files are individually atomic. Same-run folder
   aggregation assumes sequential invocations, as the pipeline currently uses;
   atomic replacement is not a cross-process merge lock. Previous receipts are
   retained separately and never added blindly into current counters.
5. Parent broad applicable GNN gate logged 1704 passed, 1 skipped, 37 deselected,
   and one execution-reason failure (`/tmp/gnn-fep-gnn-integration-final.log`,
   232.19 seconds). That failure was reproduced and repaired here, with all 26
   script-safety/receipt tests passing afterward. This is not a claim that the
   entire broad gate was rerun after the repair. Parent retains final validation
   and any real framework/Lean execution; this lane launched no native builds.

## Changed surfaces

Production: `src/gnn/api/{pipeline_runner,app,models,processor}.py`,
`src/gnn/mcp/{jsonrpc,server_core,server_stdio,server_http}.py`,
`src/gnn/render/{processor,framework_registry}.py`,
`src/gnn/execute/{metadata,processor}.py`, `src/gnn/main.py`, `.github/workflows/ci.yml`.

Focused tests: `tests/api/{test_reliability_contract,test_api_endpoints}.py`,
`tests/mcp/test_transport_reliability.py`,
`tests/render/{test_render_receipt_reliability,test_render_process_discovery,test_fep_bridge_render}.py`,
`tests/execute/test_receipt_reliability.py`,
`tests/pipeline/test_run_identity.py`.

Documentation: root `README.md`, API/MCP/render/execute READMEs,
`src/gnn/pipeline/AGENTS.md`, `tests/tests/README.md`, and this report.
Existing unrelated dirty work was retained; git diff totals against HEAD include
pre-existing fleet edits and must not be treated as this lane's isolated patch.
