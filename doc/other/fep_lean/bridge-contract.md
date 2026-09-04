# Bridge contract: fep_lean and GNN

| Field | Value |
| --- | --- |
| Version | 0.1 (draft) |
| Date | 2026-09-03 |
| Canonical copy | `../fep_lean/docs/design/gnn-bridge/bridge-contract.md` — edit there first |
| Mirror copy | this file, `doc/other/fep_lean/bridge-contract.md` |
| Change rule | substance changes bump the version and land in both checkouts in the same working session |

## 1. Purpose and scope

This contract is the formal articulation between two sibling repositories:

- `fep_lean` — a standalone catalogue of 155 FEP / Active Inference /
  Bayesian Mechanics / Information Geometry / Thermodynamics topics with
  Lean 4 theorem bodies, a pinned workspace, and distinct evidence planes.
- `GeneralizedNotationNotation` (GNN) — a text-based notation for Active
  Inference generative models with a 25-step pipeline that parses, type
  checks, renders, and executes model documents.

The contract scopes two directions of collaboration and nothing else. It
creates no capabilities, no catalogue rows, and no evidence claims. All
cross-references between the repositories are inline code paths, never
markdown links.

## 2. Parties and authority

| Concern | Authority |
| --- | --- |
| Semantics of FEP objects (laws, kernels, blankets, free energy, policies) | fep_lean |
| Syntax, parsing, rendering, and execution of model documents | GNN |
| The mapping between them | this contract, edited in both repositories |
| Evidence classification (what counts as what) | each repository's own contract; see section 7 |

## 3. Shared object and model kinds

The shared object is a **generative model instance** expressed on both
sides. The bridge recognizes exactly two model kinds, mirroring the two
kinds the GNN pipeline distinguishes:

1. **Discrete POMDP family** — categorical `A` (likelihood), `B`
   (transition, ordered `(next_state, previous_state, action)`), `C`
   (preferences), `D` (initial prior), optional `E` (habit), and an `F[1]`
   variational-free-energy readout. Counterpart carriers: `FiniteLaw`,
   `FiniteKernel`, the `active_inference.lean` `GenerativeModel`, and the
   `FiniteHMM` filtering stack.
2. **Continuous linear-Gaussian family** — `F/H/Q/R` with
   `prior_mean`/`prior_cov`, dynamics `x_t = F x_{t-1} + u_{t-1} + N(0,Q)`
   and observation `y_t = H x_t + N(0,R)`, optional closed loop via
   `goal_mean`/`control_gain`. Counterpart carriers: `LinearGaussianParameters`
   and the scalar/multivariate OU semigroup family.

Model-kind detection stays with GNN's own rules (the `GNNSection` value
drives detection, e.g. a `continuous` keyword or the presence of `F/H/Q/R`).
Emitted documents must be detectable under those rules without special
casing on the GNN side.

## 4. The interchange artifact

The only interchange artifact is a **GNN document** (`.md` in GNN syntax,
version `GNN v1.x`), whose section inventory is normatively defined by
`GeneralizedNotationNotation/doc/gnn/gnn_syntax.md`:

| Section | Status | Bridge use |
| --- | --- | --- |
| `GNNSection` | required | short identifier without spaces; bridge convention prefixes `FepLean` (e.g. `FepLeanPOMDPAgent`, `FepLeanContinuousOU`) and includes the `continuous` keyword for the continuous kind so kind detection stays mechanical |
| `GNNVersionAndFlags` | required | pins the syntax version the projection targets |
| `ModelName` | required | human-readable title |
| `StateSpaceBlock` | required | variable and matrix declarations derived from Lean index types |
| `Connections` | required | dependency edges derived from the Lean carrier structure |
| `InitialParameterization` | optional | concrete values; see the rounding policy (section 9) |
| `Equations` | optional | LaTeX restatement of the Lean-defined dynamics |
| `Time` | optional | static/dynamic and time-horizon facts from the Lean definition |
| `ActInfOntologyAnnotation` | optional | variable bindings validated by GNN step 10 against its canonical vocabulary |
| `ModelParameters` | optional | numeric parameters (`num_hidden_states`, `num_obs`, `num_actions`, `num_timesteps`, ...) |
| `Footer` | optional | closing marker |
| `Signature` | optional | **bridge convention: mandatory provenance** |

**Provenance rule.** Every bridge-emitted document carries, in its
provenance section: source repository and commit digest, the Lean module and
definition projected, the projection tool identity, and the syntax version
targeted. A document without provenance is not a bridge artifact.

## 5. Direction 1: Lean to GNN (render and execute)

| Stage | Owner | Input to output | Acceptance |
| --- | --- | --- | --- |
| S1 Selection | fep_lean | a named Lean definition family (e.g. the finite posterior–decision–action certificate carrier or an `active_inference.lean` `GenerativeModel` instance) | the named definition compiles warning-free under `fep-lean verify` before projection |
| S2 Projection | fep_lean (prospective module beside `src/fep_lean/formal/`) | Lean definition → typed projection: index types, law/kernel components, dependencies, parameters, timescale | deterministic and regenerable; no judgment calls |
| S3 Emission | bridge tooling | projection → GNN document with provenance | passes `uv run gnn validate <file> --strict` in the GNN checkout |
| S4 Toolchain | GNN | steps 3 (parse), 5 (type check), 10 (ontology), 11 (render), 12 (execute) | render and execution summaries produced; statuses reported honestly (`unsupported` is distinct from `failed`) |
| S5 Certificate | joint | execution-derived quantities vs Lean-witnessed properties | every compared quantity names its evidence plane; disagreement is a finding, never silently averaged away |

## 6. Direction 2: GNN to Lean (formalize steps and methods)

| Stage | Owner | Target | Acceptance |
| --- | --- | --- | --- |
| S1 Syntax surface freeze | GNN | the normative section inventory in `doc/gnn/gnn_syntax.md` | a pinned version reference both sides cite |
| S2 Document AST | fep_lean (prospective) | GNN sections, state-space blocks, and connections as Lean inductive types | compiles in the pinned workspace |
| S3 Well-formedness | fep_lean | decidable predicates mirroring parse, connection grammar, state-space typing, and ontology binding checks | every `input/gnn_files` exemplar decides correctly |
| S4 Dynamic semantics | fep_lean | discrete-family denotation over `FiniteLaw`/`FiniteKernel`/`FiniteHMM`; continuous-family denotation over `LinearGaussianParameters` | theorem statements for both families accepted in a slice |
| S5 FEP instantiation | both | GNN conventions ↔ fep_lean definitions (`A/B/C/D` ↔ `GenerativeModel`; `F[1]` ↔ `variationalFreeEnergy`; the `G=ExpectedFreeEnergy` ontology binding ↔ `expectedFreeEnergy_eq_risk_add_ambiguity`) | alignment statements named as prospective until proven |
| S6 Renderer statements | fep_lean | semantics-preservation statements for render targets | statements first; proofs are later slices |

Direction 2 detail lives in
`fep_lean/docs/design/gnn-bridge/direction-2-gnn-to-lean.md` (cross-repo
references are inline code paths, never markdown links).

## 7. Evidence firewall

Each claim names exactly one evidence plane, and planes never substitute for
each other:

| Plane | Producer | Establishes | Never establishes |
| --- | --- | --- | --- |
| Native Lean compilation | `fep-lean verify` | the named body compiles without warnings or `sorry` | semantic reach, executed behavior |
| Semantic review | `config/theorem_maturity.yaml` | how far a theorem proxy reaches toward its topic label | compilation sufficiency |
| Numerical witness | `fep-lean dashboard` | typed, explanatory finite witnesses | proof or empirical validation |
| GNN pipeline run | steps 3–12 | the document parses, renders, and executes | mathematical correctness of the model |

A bridge certificate may only state agreement between quantities that each
carry their own plane; it may not reclassify a simulation statistic as a
proved property.

## 8. Synchronization and ownership

- Canonical contract: `fep_lean/docs/design/gnn-bridge/bridge-contract.md`.
  Mirror: `GeneralizedNotationNotation/doc/other/fep_lean/bridge-contract.md`.
- Contract edits land in both checkouts in the same working session; the
  mirror stays identical to the canonical body except for (a) the
  canonical-pointer header rows and (b) cross-repository markdown links,
  which the mirror renders as inline code paths.
- `fep_lean/docs/design/gnn-bridge/` is maintained by fep_lean-side agents;
  `doc/other/fep_lean/` in the GNN checkout by GNN-side agents.
- Version bumps follow the change rule in the header table.

## 9. Policies and no-go registry

| Trigger | Action |
| --- | --- |
| A field of the GNN document cannot be derived deterministically from the named Lean definition | stop; narrow the extraction contract; never hand-fit the document |
| Syntax surface drift in `doc/gnn/gnn_syntax.md` | freeze the pinned version per slice; re-freeze explicitly before extending |
| Numeric rounding of exact Lean values | rounding policy fixed once in this contract's first slice; the digest of the exact source is recorded in provenance |
| A projected model exceeds a backend (e.g. continuous model on a categorical-only renderer) | report `unsupported`; never distort the model |
| A desired ontology binding is absent from the GNN canonical vocabulary | either use existing terms or open an explicit vocabulary-extension request on the GNN side; never emit bindings that fail step 10 |
| Contract edit without the mirror edit | revert until both land |
| Pressure to claim execution "verifies" a theorem or a theorem "validates" execution | refuse; section 7 is non-negotiable |
