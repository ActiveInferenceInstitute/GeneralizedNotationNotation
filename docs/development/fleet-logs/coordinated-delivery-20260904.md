# Coordinated GNN / fep_lean reliability delivery

Date: 2026-09-04. This implements the approved reliability-first work, source
custody, concrete PyMDP artifact proof, and audit of the existing H2 terminal
certificate. It preserves the separate responsibilities of the two projects:
GNN parses, validates, renders, and executes models; fep_lean owns formal
statements, source-bound evidence, and bounded numerical comparisons.

## GNN changes

Validation now shares typed semantics across dictionary, file, CLI, and MCP
entry points. Invalid results remain invalid through the pipeline. Connection
annotations survive the supported serializer/parser paths, including absent
annotations and Unicode. CLI composition tests mock the actual imported
boundaries; they no longer recursively launch the test pipeline.

Both APIs use the same return-code policy, validate requested steps strictly,
reject deletion of active jobs, and accept subprocess summaries only for the
expected run identity. The main orchestrator carries one run ID through child
steps, writes it into success and failure summaries, restores environment
state, and serializes overlapping in-process run scopes.

Render and execution receipts bind source/artifact hashes and current run/config
scope. Retries replace current entries, changed artifacts reject old success,
and archived receipts do not inflate current counts. JSON output publication
is atomic per file. Same-run folder aggregation still assumes the pipeline's
sequential invocations; it is not a cross-process merge lock.

HTTP and stdio MCP transports share JSON-RPC validation, including malformed
parameters, notifications, explicit null IDs, authentication, rate limits,
and allowed methods. Notifications return no JSON-RPC response (HTTP 204).
Bnlearn advertises rendering only. Functional transport/capability tests are
wired into CI.

The actual FEP render exposed a vector-shape mismatch in the parser mapping to
typed renderer input. Singleton C/D/E vectors are flattened before dimension
inference. The final broader tests also exposed an existing missing rule-based
summary return, a malformed composition-test name, and a missing logging import;
these were repaired along with three source Ruff findings.

## Verified GNN results

| Gate | Result |
| --- | --- |
| Applicable parser/validation/API/CLI/MCP/render/execute/test-pipeline/audio/analysis suite | 2,102 passed, 1 skipped, 277 deselected; 302.02 seconds |
| Actual localhost MCP HTTP/auth tests | 12 passed; 5.20 seconds |
| Full `ruff check src` | passed |
| Strict documentation audit, without writes | all issue counts zero |
| Validation lane scoped typing | 54 source files passed |

The applicable suite used `-m 'not slow and not pipeline'`. The skip is an
unavailable PyTorch dependency. This is not a claim that every repository test,
paid provider, long-running simulation, or entire pipeline was exercised.

Reproduction:

```bash
uv run --offline --no-sync pytest \
  tests/gnn tests/validation tests/api tests/cli tests/mcp \
  tests/render tests/execute tests/tests tests/pipeline \
  tests/audio tests/intelligent_analysis \
  -m 'not slow and not pipeline' --timeout=120 -q
uv run --offline --no-sync pytest tests/mcp/test_mcp_http_auth.py -q
uv run --offline --no-sync ruff check src
uv run --offline --no-sync python doc/development/docs_audit.py --strict --no-write
```

Local logs: `/tmp/gnn-fep-gnn-accepted-suite.log`,
`/tmp/gnn-fep-mcp-live-socket.log`, `/tmp/gnn-fep-gnn-ruff-full.log`, and
`/tmp/gnn-fep-gnn-docs-final.log`. Detailed ownership and focused evidence are in
[validation delivery](reliability-validation-REPORT.md) and
[receipt delivery](reliability-receipts-REPORT.md). This final suite supersedes
the earlier broad run's repaired execution-reason failure.

## FEP and joint delivery

The sibling fep_lean checkout contains:

- `specs/gnn-bridge-w2-source-custody/REPORT.md`: shared bridge operations,
  actual owner pins, finite/continuous emission, strict custody refresh, and
  independently validated numerical comparisons. Fifty-five focused bridge
  and watchdog tests pass; ten actual check commands preserved fifteen files'
  bytes and mtimes.
- `specs/gnn-bridge-q5-artifact-proof/REPORT.md`: one current canonical rendered
  PyMDP artifact and one handcrafted asymmetric control, exact dyadic static
  extraction, independent Lean payloads, six theorem/axiom checks, and explicit
  source/toolchain/import-bound native receipts. Static artifact evidence does
  not claim runner execution or arbitrary extractor correctness.
- `specs/horizon-2-smooth-stochastic/readiness/07-terminal-audit-20260904.md`:
  independent scientific, Lean/API, and skeptical audits; explicitly typed
  scalar/Fin4 consumers; negative carrier checks; a thirty-theorem axiom census;
  and appended R0 custody evidence preserving the historical receipt.

FEP additionally repairs OpenGauss nested-home creation, wheel smoke portability,
formal-test mocks at the real subprocess boundary, and inverted subprocess
deadline watchdog logic. Native probes use coordinated serial execution.
The FEP full nonserial baseline passed 1,159 tests (seven skips; 523 native
deselections), followed by 82 additional passing failure-contract tests on
unchanged production bytes. Combined coverage is **89.19%**, above the unchanged
89% gate. The first run's coverage shortfall and the successful appended
coverage run are recorded separately in the FEP ISA. All eight source
projections and the refreshed manuscript projection/placeholder checks pass.

## Review and limits

Codex and Omp worked in named Herdr tabs with disjoint ownership. Parent review
integrated the lanes and corrected concrete findings. Fresh independent Codex
review rejected earlier AST-shadowing, malformed-manifest, generator race, and
axis-control gaps; final reprobes found no remaining blockers in those deltas.
The configured planning advisor timed out, and the validation lane's Cato
startup refused missing metadata; neither is represented as an approval.
Independent completed reviews provide the evidence described above.

GitNexus did not index either nested repository, so source/consumer searches
provided the available impact review. Baseline snapshots preserve the initial
dirty state. No baseline files were deleted; pre-existing unrelated work was
retained, including FEP's unchanged W1 REPORT. Both repository HEADs remain at
their initial values; commits alone do not identify the tested working bytes.

Wider H2 acceptance still needs its complete predecessor matrix, numerical
diagnostics, and overall decision. ActiveInference.jl concrete artifact proofs,
reviewed continuous-dynamics proofs, and H3 remain separate gated work. This
delivery does not change historical release/provider evidence into current
evidence or authorize publication.
