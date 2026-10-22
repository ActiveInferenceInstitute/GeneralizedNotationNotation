# Remaining roadmap contracts

Design scope: 2026-10-16 review, based on GNN
`2b105fde2c0f8fcd50eec5a45ab22d9925c01383` plus the uncommitted review patch.
This document specifies implementation boundaries and acceptance cases. It does
not complete the seven items in [TO-DO](../../TO-DO.md), certify a release, or
replace [scientific admission](backend_scientific_contracts.md) and
[run ownership](run_ownership_migration.md).

Target-version sequencing for **4.3.0–4.9.0** is owned by
[TO-DO.md](../../TO-DO.md#versioned-delivery-plan-430-through-490).
These contracts define acceptance boundaries and do not certify those targets
as implemented. Historical numerical/archive observations keep their dated scope.

## Ownership and delivery

| Item | Implementation owners | Next independently reviewable unit |
| --- | --- | --- |
| S1 | Step 13 corpus scheduler, request worker, provider adapters | Source-preserving request-plan admission and segment checkpoints |
| E4 | Extraction, semantic fidelity, analysis, FEP/GEO companions | Independent source-to-numerics witness schema and negative controls |
| E1 | Composed-kind dispatch and JAX/RxInfer Gaussian renderers | Explicit joint Gaussian coupling contract |
| E2 | THRML admission, payload compiler, runtime and analysis | One separately named inference extension at a time |
| E3 | cpomdp option admission, generator, executor and analysis | Explicit finite action-set contract with bounded exhaustive search |
| S3 | THRML dependency and native scientific tests | Verify a suitable ordinary released wheel on both lock splits |
| E5 | Release artifacts, citation metadata and archive receipt | Verify the exact version archive and public artifact hashes |

Content changes to bridge-bound owners require the
[paired revision procedure](fep_lean_paired_revision.md). Count-changing manuscript
inputs require the complete render ritual. Historical receipts retain their
original source identities; a local patch cannot refresh hosted or companion
acceptance by changing metadata.

## S1: complete multi-request processing

The current whole-context path remains authoritative. Context refusal is
unfinished work. A token-limited reply is retained with `incomplete` status and
cannot enter completion checkpoints or the cache. The review patch binds that
rule through `terminal_stop_v1` and request-cache version 3.

The next implementation should admit an immutable request plan before issuing
provider calls. Each plan must contain:

- Exact source-byte digest, encoding, byte length and ordered segment offsets.
  Reading, reassembly and hashing must preserve newlines, Unicode and the final
  byte. Segments must cover the source exactly once, without gaps or overlaps.
- Exact provider, endpoint identity, model identity and prompt-definition digest.
  Context capacity and tokenizer/chat-template identity must have verifiable
  origins. Character counts and guessed tokens cannot certify context admission.
- The context reservation for system messages, task instructions, prior segment
  receipts and output generation. Every request must fit its admitted budget.
- Every selected source/prompt unit, its ordered segment requests and synthesis
  dependencies. A segment summary is evidence about that segment; it cannot
  satisfy a task requiring the entire original source by itself.
- A versioned task contract identifying what synthesis may conclude from segment
  outputs. Tasks requiring simultaneous full-source context must refuse when it
  cannot fit. Segmentation alone does not preserve arbitrary reasoning semantics.
- The invocation's monotonic deadline, per-request limit and cancellation token.
  No automatic model substitution, truncation or budget increase is permitted.

Checkpoint each segment response with its request digest, content digest,
provider/model receipt and terminal finish reason. Revalidate those records on
resume before scheduling dependent requests. Only a complete, accepted synthesis
may complete the original prompt unit. Interrupted segments, failed synthesis,
missing dependencies and late publication remain unfinished. Cache identity must
include the complete request plan and completion policy.

Acceptance must include exact reconstruction of CRLF and Unicode sources,
changed source/model/endpoint/tokenizer/prompt rejection, interrupted segment and
synthesis resume, duplicate stems, cancellation, exhausted context/output budgets,
corrupt checkpoints, and genuine provider runs beyond one context window. Test
injections validate scheduling only. S1 remains open until native coverage is
complete for every required source/prompt in the selected acceptance corpus.

## E4: independent semantic witnesses

Source hashes, matching values and shapes are necessary identity checks. They do
not prove extraction correctness or equivalence between inference algorithms.
The proposed witness should record:

- Source digest and exact declared equations, variable/axis identities, initial
  time convention, parameters, observations, actions and conditioning horizon.
- Independently extracted reference parameters. A reference that calls the same
  extractor or copies generated parameters cannot detect that extractor's error.
- The named inference question: filtering, smoothing, prediction, policy scoring
  or control. Preserve log-preference sign, EFE convention and policy/action axes.
- Reference algorithm and provenance, native library/runtime versions, generated
  script digest, dtype and justified absolute/relative or Monte Carlo tolerances.
- Formal statement and assumptions, theorem/proof identity and bridge revision as
  separate optional fields. Missing proof evidence stays unavailable.
- Per-quantity comparisons and explicit refusal reasons. A global Boolean must
  not erase unavailable or incompatible comparisons.

For finite models, begin with independent path enumeration on a small asymmetric
HMM. Compare filtering and full-sequence smoothing separately on the same fixed
observations/actions. For Gaussian models, begin with independent NumPy Kalman
updates using asymmetric state/observation dimensions, nonidentity observation
matrices, nonzero means and correlated positive-definite covariances. Check the
first observation against the declared prior without an extra predict step.
Existing cpomdp NumPy Kalman/EFE and THRML log-space smoothing tests are useful
numerical controls; they do not establish independent source extraction or Lean
proofs.

Negative controls must swap observation/state axes, perturb one transition,
shift an action in time, add an extra predict step, change conditioning horizon,
change EFE sign, omit cross covariance, substitute a source digest and change
precision. Each must fail or refuse with a quantity-specific diagnosis. Add a
new witness version before extending its scientific meaning.

### E4 foundation: source-bound numerical witness v1

The [test-only reference reader](../../tests/helpers/numerical_witness.py) and
[actual source-to-native suite](../../tests/render/test_source_numerical_witnesses.py)
implement one deliberately restricted subset. The reader imports no GNN parser,
canonicalizer, renderer or backend. It accepts one-line JSON-valued assignments in
`InitialParameterization`, colon-delimited scalar `ModelParameters`, exact declared
equations/state identities and a versioned `NumericalWitness` JSON section. Other
source grammars and inference contracts refuse explicitly; this is not a second
complete GNN parser. Syntax accepts CRLF/LF lines while source-byte hashes retain
the untouched bytes, including line endings and the final newline.

- **Finite:** one stationary categorical state, observation and exogenous action
  factor; canonical `A[observation,state]` and
  `B[next_state,previous_state,action]`; normalized A/B/D and zero C. Exact latent
  path enumeration computes each filtering prefix independently and full-sequence
  smoothing separately. The asymmetric native fixture has two states, three
  observations, two transition actions and four observation times. Native bnlearn
  posterior comparisons cover filtering only; smoothing remains reference evidence.
- **Gaussian:** one passive stationary LGSSM, with no authored control or policy
  contract. The asymmetric fixture has three states, two observations, a
  nonidentity rectangular H, nonzero prior means and correlated positive-definite
  prior/Q/R. The reference constructs the full joint Gaussian from independent
  initial/process innovations and conditions each observation prefix. It does not
  call a Kalman update. Every mean and full covariance is compared with the actual
  generated JAX program. Its first observation conditions the declared prior;
  generated one-based time notation maps to result index zero. The test inspects
  actual JAX float64 arrays and x64 configuration in the execution process.

Reader limits are eight observation times, four state/observation dimensions,
four finite actions and 4,096 enumerated paths. Deterministic comparisons use
float64, absolute tolerance 1e-12 and relative tolerance 1e-10 for these small,
well-conditioned fixtures. These bounds are deliberately wider than observed
roundoff; they are not Monte Carlo tolerances or permission to substitute float32.
Records name source, axes, ordered identities, equations, initial time, inference
question, every conditioning horizon, actual observations/actions and each compared
quantity. Native receipts additionally retain checkout/dirty state, generated
source/program/result bytes and hashes, plus actual Python/NumPy/framework
versions. All-extras CI
requires both receipts from its actual checkout and retains them with its JUnit
artifact; missing native execution cannot pass that receipt step.

Counterexamples cover transposed rectangular source matrices, changed transition
probabilities, shifted action timing, extra initial prediction, erased cross
covariance, covariance-only corruption, future conditioning, source substitution
and precision/axis mismatch. Unsupported EFE/sign questions refuse rather than
borrowing this filtering contract. The suite also rejects duplicate, nonfinite,
boolean, ambiguous and over-budget source declarations.

These finite tests establish their named numerical comparisons. Formal evidence
is explicitly unavailable in the witness schema. They do not refresh Q7/native
Lean receipts or synchronize FEP_Formal, and they do not prove general extraction,
backend-wide equivalence, planning, causal discovery, coupled models or arbitrary
GNN compositions. Broader E4 work remains open.

### E4-P1: authored planning admission

The shared `render/planning_admission.py` guard refuses unsupported source
horizons and policy identities on the six ordinary one-step planning adapters,
including direct and Step 11 entry points. It checks all supported parameter
containers before canonicalization can hide conflicts. This closes silent
one-step fallback for these explicit declarations only. The tracked five-step,
64-policy example retains its original source and remains unsupported for
planning until it supplies an executable policy-set/pruning contract. Broader
policy search and independent numerical qualification remain open.

### E4-BN1: finite bnlearn source filtering

The new `render/bnlearn/source_program.py` installs declared finite A/B/D CPTs
and performs native conditional filtering under exogenous transition actions.
Asymmetric native witnesses compare every posterior against independent forward
filtering. Source values, canonical axes, normalization provenance, timesteps,
seed and source/script identities remain observable. Metadata-only compatibility
calls retain a separately labelled synthetic smoke contract.

This closes the finite stationary filtering slice only. Source-specific causal
DAG discovery, independent/dependent component composition, C/E control semantics
and universal translation proofs remain open. Missing/invalid CPTs and unsupported
compositions refuse instead of reusing the synthetic demonstration. Extend the
witness contract before admitting those semantics.

## E1: coupled continuous agents

The existing independent-agent contract is not a coupling declaration. A first
coupled contract should define a joint linear-Gaussian system explicitly, with
ordered agent IDs and separate state, observation and control dimensions. Record
all block offsets and every cross-agent dynamics, observation and covariance
block; symmetry and positive definiteness apply to the assembled covariances.
Missing cross blocks must have a declared meaning rather than an inferred zero.

Start with centralized joint filtering and explicitly declared exogenous controls.
Do not call it decentralized inference or introduce policy optimization under the
same contract. Define observations and action timing before generator dispatch.
JAX and RxInfer must consume the same declared joint system, retain full posterior
cross covariances, and expose per-agent views by the recorded offsets.

Acceptance requires two agents with unequal dimensions, directional coupling,
nonzero cross covariance and independently specified observations. Compare against
E4's joint Gaussian reference. Setting coupling to zero must recover the existing
independent model. Permuting agent identities must permute all corresponding
blocks and results consistently. Missing controls, ambiguous joint priors and
unsupported factored/hybrid/nonstationary compositions must refuse explicitly.
Native RxInfer acceptance requires a provisioned Julia project environment.

## E2 and S3: THRML extensions

The maintained released contract is fixed-action categorical Gibbs smoothing with
strictly positive probabilities. The upstream
[padding issue](https://github.com/extropic-ai/thrml/issues/72) was closed on
2026-10-05, while the [PyPI metadata](https://pypi.org/pypi/thrml/json) inspected
on 2026-10-16 still identifies 0.1.4 as latest. Closure alone does not verify a
new released wheel. Preserve structural-zero refusal until S3's native witnesses
pass on both supported Python/JAX splits.

A future structural-zero witness must preserve active negative-infinite log
potentials while ensuring inactive padding cannot contaminate unrelated nodes.
Check conditional logits, independent-node empirical frequencies, source/seed/
sample identity, finite outputs and zero-frequency impossible states. Do not
substitute epsilon clipping or a local monkeypatch for released compatibility.

Coupled categorical factors, policy search, continuous/hybrid models and hardware
execution require different named contracts. For each, define factor semantics,
conditioning, normalization, resource bounds and independent witnesses before
adding capability registration. Hardware operation needs actual device receipts;
software sampling cannot establish hardware, convergence, energy or speed claims.

## E3: cpomdp action/search extension

The maintained compass-rose action set and exhaustive search remain unchanged.
Direct options now share typed public admission; invalid options refuse without
replacing a caller's existing artifact.

E3-FA1 now admits a source-declared finite action set with an explicit
version, ordered action IDs, units and a control matrix mapping each action into
state space. Validate finite, nonboolean components and exact asymmetric control
dimensions. Preserve the authored action order for ties and policy identity.
Do not silently deduplicate actions or reduce the horizon.

Compute policy count by bounded multiplication before enumeration. Bound total
policy-step evaluations and retained score/allocation estimates before building
arrays, and retain the process deadline/RSS supervision. Passive mode must not
allocate a planner. Receipts must record the complete declared action set,
horizon, source contract, every score and selected policy/action. All-invalid
scores fail before selection. Native acceptance must compare Kalman posteriors
and every EFE component against an independent reference, including tie and
invalid-score cases. Estimated allocation remains separate from measured RSS.

## E5: exact-version archive

The [v4.1.0 version record](https://zenodo.org/records/23245091) resolves via
`10.5281/zenodo.23245091`, separately from concept DOI
`10.5281/zenodo.7803313`. A fresh 2026-10-16 download matched all 3,289 tagged
Git blobs and the release manuscript. Public wheel, sdist and manuscript bytes
matched GitHub digests and published SHA-256 sums. The automatic Zenodo record
contains the tagged source ZIP only; separate wheel, sdist and distribution
checksums are absent. E5 remains open for those archived assets and requires
archive-owner access.

Archive only the selected immutable release, with its source SHA, tag, wheel,
sdist, manuscript and checksums. The current working tree is a later review patch
and cannot substitute for v4.1.0 release bytes. Verify a version-specific archive
record independently of the concept DOI. Download every advertised artifact from
its public record and compare its digest with the selected release inventory;
check citation metadata and DOI resolution after publication.

A local package build or an unverified archive URL does not satisfy E5. Preserve
failed provider checks and leave the item open when public resolution or artifact
hash parity cannot be established. Archive publication and package-index delivery
remain distinct operations with their own destination and ownership contracts.

### E3-FA1 boundary

The source-declared finite-control implementation is specified in the
[cpomdp guide](../gnn/implementations/cpomdp.md#source-declared-finite-controls-e3-fa1).
It preserves authored action order, distinct IDs for equal vectors, units,
asymmetric control maps and complete per-policy EFE components. Admission also
bounds total evaluations across timesteps. This does not close E3's broader
control/search workstream, admit continuous action spaces, or establish E4's
independent source extraction or universal proof obligations.
