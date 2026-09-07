---
phase: complete
---

# Coordinated repository improvement ISA

## Coordinated reliability and artifact-proof delivery (2026-09-04)

### Goal

Deliver reliable model/run outcomes, read-only content-bound bridge checks,
and concrete PyMDP artifact proofs, preserving existing fleet changes.

### Criteria

- [x] BRIDGE-OPS: new bridge operations regression suite passes.
- [x] GNN-VALIDITY: current invalidity propagates across validation entry points.
- [x] GNN-ROUNDTRIP: annotated connections survive JSON/Markdown round trips.
- [x] GNN-RECEIPTS: retries and changed inputs do not inflate current outcomes.
- [x] GNN-API: both APIs agree on warning exits and reject active deletion.
- [x] GNN-MCP: transport behavior has functional PR tests.
- [x] PROOF-PAYLOAD: a current rendered PyMDP artifact and an independent
  asymmetric control have checked concrete payloads.
- [x] PROOF-NEGATIVE: literal/axis/artifact/custody mutations reject acceptance.
- [x] H2-AUDIT: existing H2.7 source is audited against its review gates.
- [x] Anti-CURRENT: no stale numerical artifact is promoted to native proof or
  current execution evidence.
- [x] Anti-WORKTREE: baseline unrelated changes remain intact.

### Verification strategy

Use repository-native focused tests followed by full applicable gates. Source
snapshots were captured before edits under `/tmp/gnn-fep-implementation-20260904`.
Full baselines were attempted and interrupted in native dependency checks amid
pre-existing fleets; logs retained under `/tmp/gnn-fep-*-baseline.log`.
Focused fep_lean baseline: 65 passed (CLI/formal composition/native evidence).
GNN focused baseline and subsequent regression results must be reported with
exact scope. No baseline aggregate pass is inferred from partial progress.

### Ownership

Codex validation lane: parsers/validation/CLI. Codex receipt lane: API/MCP/render/
execute. Omp: concrete artifact-proof slice. Parent: bridge operations, contract,
source pins, ISA and integration. Generated owners settle before receipt refresh.
No publication, paid provider runs, or H3 execution is authorized by this slice.

### Verified delivery

The coordinated delivery criteria above are complete. Evidence:

- GNN applicable suite: 2,102 passed, one unavailable-PyTorch skip, 277 slow/
  pipeline deselections. Twelve additional actual HTTP/auth socket tests passed.
  Full source Ruff and strict documentation audit pass.
- FEP full nonserial baseline: 1,159 passed, seven skipped, 523 native-marked
  deselections in 760.17 seconds. That run initially missed coverage at 88.34%.
  All 82 added failure-contract tests then passed in 4.02 seconds with
  `--cov=src --cov-append --cov-fail-under=89`, bringing combined coverage to
  **89.19%**. All 77 production Python files stayed byte-identical across those
  runs. This records a full baseline plus the additional tests, not a claimed
  second full-suite run. Logs: `/tmp/gnn-fep-fep-python-final.log` and
  `/tmp/gnn-fep-contract-coverage-final.log`.
- Q5: current actual canonical render, two concrete native probes, six standard-
  axiom theorem checks, and three passing native regression tests including a
  normalization-preserving wrong-axis rejection. The asymmetric control is
  handcrafted; only the symmetric fixture has current render provenance.
- H2 terminal audit: seven direct terminal tests, sixteen R0 tests including
  three native probes, and two final prerequisite/custody checks passed.
  Overall H2 acceptance remains open; this audit does not open H3.
- Ten actual read-only bridge/Q5 checks preserved all fifteen watched artifacts'
  bytes and mtimes. Numerical comparisons keep current-execution verification
  and native-claim readiness false. The separate native Q5 receipt validates.
- Eight source projections, manuscript projection/placeholder checks, Markdown
  hygiene and manuscript references pass. Strict typing passes for ten bridge
  and verification files. Both repository diffs pass whitespace checks.

At the wave-1 checkpoint, the source-bound native receipt and source pin were
current. Baseline files were preserved, the pre-existing FEP W1 REPORT is byte-identical to its initial
snapshot, and neither repository HEAD changed. Later backend proofs, continuous
semantics, wider H2 acceptance, H3, current-release/provider evidence, and
publication retain their independent acceptance boundaries.

Detailed cross-repository evidence: [delivery report](coordinated-delivery-20260904.md).

## Comprehensive continuation (2026-09-04, wave 2)

User-authorized wave-2 implementation and verification are complete. Prior evidence above describes the
wave-1 bytes; subsequent changes require fresh source-bound checks. Worktree
snapshots and baseline logs are retained under
`/tmp/gnn-fep-comprehensive-wave2-20260904`. Existing work remains preserved.

### Acceptance criteria

- [x] W2-GNN-QUALITY: declared GNN test scope, full typing, lint and format pass.
- [x] W2-GNN-RUNS: source/config/artifact identity governs hashing, resume and reproduction.
- [x] W2-GNN-CONTAINERS: reviewed container settings survive composition and paths reject unsafe aliases.
- [x] W2-CONTINUOUS-ROUTE: public continuous dispatch and JAX output routing pass actual regressions.
- [x] W2-Q6: actual Julia embedded input tables have positive and wrong-axis native evidence.
- [x] W2-Q7: actual OU coefficients have source-bound exact-real error-bound evidence.
- [x] W2-RECEIPTS: shared immutable contract verification preserves tamper and checked-byte safeguards.
- [x] W2-H2-DIAGNOSTICS: scalar and Fin4 diagnostics use the existing typed witness registry.
- [x] W2-H2-EXIT: mandatory predecessor/native results, diagnostics and three fresh reviews validate together.
- [x] W2-H3-G0: eligibility checks the accepted H2 carrier and preserves prospective study boundaries.
- [x] W2-FEP-QUALITY: relevant full Python gates and native additions pass with exact exclusions reported.
- [x] W2-DOCS: current architecture/status and reviewable evidence match final code.
- [x] Anti-W2-CLAIMS: static proofs do not imply runtime, empirical or whole-program equivalence.
- [x] Anti-W2-CUSTODY: receipt regeneration is explicit; stale or edited evidence fails closed.
- [x] Anti-W2-WORKTREE: unrelated baseline files and repository HEADs remain preserved.

Native Lean/Lake commands are serialized. New FEP implementations are drafted
outside the source trees until the frozen-source baseline settles. GNN's full
applicable baseline completed with 4,028 passes, 11 skips, 557 deselections and
two failures; those failures and the full typing findings are implementation
inputs. Publication and paid provider execution are outside this continuation.

### Wave 2 GNN verification checkpoint

The integrated GNN run (`-m 'not pipeline and not mcp'`) completed with
4,157 passes, 18 skips, 557 deselections and one UV environment-check failure.
All 2,870 captured source/workspace files were unchanged during the run. The
failure was an exact-sync check rejecting the concurrent GEO lane's optional
`h3` package. Its documented non-pruning contract now uses `--inexact`; all
35 tests in that environment file pass, with required dependencies still
checked. This is the full run plus a scoped repair/recheck, not a second full
run. Full typing (986 files), Ruff, formatting and four strict documentation
audits pass; the changed environment test also passes typing and lint.

Additional evidence includes 128 independent durable-run/pipeline tests,
13 OpenAI synchronous-call tests, and 102 tests with optional scikit-learn
present. Deserialization rejects cached pickle extension opcodes and trailing
payloads at the shared GNN loader and the restricted classifier loader.

The following final checkpoint supersedes the preceding in-progress status.

### Final wave-2 checkpoint

All wave-2 criteria above are verified. FEP's clean integrated Python run passed
1,460 tests with seven skips, 529 native deselections, and 89.83% coverage.
The frozen full native baseline and enabled Fin4 supplement passed. Q5/Q6/Q7
schema-2 receipts validate 23 standard-axiom theorem reports. H2.7 acceptance
binds 328 mandatory cases, 180 source hashes, independent diagnostics, and three
fresh reviews. All 15 retained checks preserve 559 files' bytes and mtimes.

H3.G0 machinery is implemented and tested; actual prospective study metadata
remains unselected. No G0 study acceptance or H3.0--H3.7 execution is claimed.
Broader release/provider criteria elsewhere in this ISA remain independent.

Detailed changes, exact test scopes, repairs, and evidence: [wave-2 report](comprehensive-wave2-20260904.md).
