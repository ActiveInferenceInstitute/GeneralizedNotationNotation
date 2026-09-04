# fep_lean: the Lean 4 FEP catalogue

> **Document Metadata**
> **Type**: External project overview | **Audience**: Researchers, Developers | **Complexity**: Advanced
> **Cross-References**: [README.md](README.md) | [Collaboration program](fep_lean_gnn.md)
> **Last Updated**: 2026-09-03

## Overview

`fep_lean` (sibling checkout `../fep_lean`) is a standalone catalogue of 155
FEP / Active Inference / Bayesian Mechanics / Information Geometry /
Thermodynamics topics across 20 reviewed families in five areas. Each row
carries a reviewed invariant, explicit assumptions, and a Lean 4 theorem body;
the pinned Lean workspace is the compilation authority, and a separate
semantic review records how far each theorem reaches toward its topic label.

For GNN readers the essential fact is the layer split: GNN specifies, renders,
and executes model instances; fep_lean states and proves invariants about
those instances. The collaboration program ([fep_lean_gnn.md](fep_lean_gnn.md))
connects the two.

## What is formalized (high-signal inventory)

All paths are inside the sibling checkout (inline code, not links).

| Area | Lean module (`lean/FepSketches/...`) | Verified declarations |
| --- | --- | --- |
| Finite probability | `finite_probability.lean` | `FiniteLaw`, `FiniteKernel`, `pointMass`, `uniform`, `product`, marginals |
| Finite information | `finite_information.lean` | `entropy`, `finiteKL`, `crossEntropy`, `conditionalKL`, `mutualInformation` |
| Bayesian inversion | `measure_bayes.lean` | posterior reconstruction from likelihood ratios, `posterior_involution` |
| Markov blankets | `markov_blanket.lean` | `Blanket`, `StaticModel`, blanket factorization, `conditional_mutualInformation_zero` |
| Native blankets | `native_blanket.lean` | `embeddedLaw`, `embeddedKernel`, `staticJoint_condIndepFun` (native `CondIndepFun`) |
| Active inference | `active_inference.lean` | `GenerativeModel` (A/B/C/D-style POMDP), `variationalFreeEnergy`, `expectedFreeEnergy_eq_risk_add_ambiguity`, `epistemicValue_eq_entropy_sub_ambiguity`, `policyPosterior`, `ActionInterface`, `inferSelectActKernel` |
| Filtering | `temporal_inference.lean` | `FiniteHMM`, `forwardFilter_reconstruction`, `forward_backward_evidence_agree`, smoothing |
| Control and planning | `controlled_markov.lean`, `policy_tree.lean` | `ControlledKernel`, `ReachableBeliefPOMDP`, `boltzmannPosterior`, `SophisticatedEFEModel`, `PolicyTree`, `policyTree_efe_eq_risk_add_ambiguity` |
| Gaussian/OU dynamics | `linear_gaussian_semigroup.lean`, `scalar_gaussian_semigroup.lean` | `LinearGaussianParameters`, `ScalarOUParameters`, `stationaryLaw_invariant`, `ouKL_to_stationary_nonincrease` |
| Continuous-time Markov | `continuous_time_markov.lean` | `FiniteRateGenerator`, `FiniteMarkovSemigroup`, `nativeKL_contraction`, `TwoStateRates` |
| Learning theory | `learning_theory.lean` | PAC-Bayes change-of-measure, posterior-odds recursion, mixture-evidence bounds |

The direct GNN correspondences: `GenerativeModel` matches the GNN discrete
POMDP family (`A` likelihood, `B` transition ordered
`(next_state, previous_state, action)`, `C` preferences, `D` prior, `E` habit,
`F[1]` variational-free-energy readout), and `LinearGaussianParameters` matches
the GNN continuous linear-Gaussian family (`F/H/Q/R` with
`prior_mean`/`prior_cov`). See [fep_lean_gnn.md](fep_lean_gnn.md).

## Evidence planes

fep_lean keeps its evidence classes strictly separate; the bridge contract
inherits this discipline.

| Plane | Command | What it establishes |
| --- | --- | --- |
| Deterministic offline artifacts | `uv run fep-lean catalogue` | validated YAML projections (figures, appendix); zero verified topics claimed |
| Native Lean compilation | `uv run fep-lean verify --fail-on-warnings --receipt output/native-verification.json` | the canonical bodies compile without warnings or `sorry` |
| Coverage graph | `uv run fep-lean atlas --check` | which relations carry qualified Lean witnesses |
| Numerical witnesses | `uv run fep-lean dashboard --check` | typed, explanatory finite witnesses (never proof) |
| Full mode | `uv run fep-lean run` | Hermes + Lean + SQLite verification with strict report semantics |

## Commands of record

From the `fep_lean` checkout root (workspace setup once via
`uv run fep-lean setup`; the Lean workspace pins its toolchain in
`lean/lean-toolchain`):

```bash
uv run fep-lean verify --fail-on-warnings --receipt output/native-verification.json
uv run fep-lean atlas --check
uv run fep-lean dashboard --check
cd lean && lake build FepSketches
```

The read-only report receipt checker is
`uv run python scripts/verify_report_receipt.py <report-dir> --require-complete`.

## Why this matters for GNN

- The quantities GNN documents annotate — variational free energy (`F[1]`),
  expected free energy (`G=ExpectedFreeEnergy`), hidden states, observations,
  policies — have proved decompositions and nonnegativity results on the
  fep_lean side, so an executed GNN model can be checked against witnessed
  properties rather than only against itself.
- Blanket structure is native: conditional independence between internal and
  external states given the blanket holds as `CondIndepFun` in Mathlib's
  measure/kernel layer, not just as a modeling convention.
- The two GNN model kinds map one-to-one onto the two fep_lean carrier
  families, which is why the collaboration program scopes exactly those two
  families first.

## Where the collaboration lives

- fep_lean side: `../fep_lean/docs/design/gnn-bridge/` (design program,
  canonical bridge contract, both direction programs).
- GNN side: this folder, with the program in
  [fep_lean_gnn.md](fep_lean_gnn.md) and the contract mirror in
  [bridge-contract.md](bridge-contract.md).
