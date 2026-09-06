# ACTIVEINFERENCE_JL Rendering Results

Generated from GNN POMDP Model: **Dirichlet Likelihood Learning Agent**

## Model Information

- **Model Name**: Dirichlet Likelihood Learning Agent
- **Model Description**: This model describes a discrete POMDP agent that learns its observation model:
- 3 hidden states, 3 observation outcomes, 2 actions (cycle, stay).
- The likelihood matrix A is NOT fixed: it is a latent DirichletCollection
variable with prior pseudo-counts declared in dirichlet_A.
- The A values under InitialParameterization are the GROUND-TRUTH likelihood
used by the environment to simulate observations; the agent never sees them
directly and must recover them in q(A).
- The Dirichlet prior is identity-biased (diagonal 3.0, off-diagonal 1.0):
the agent starts believing observations weakly track states. A fully
uniform prior leaves the column-permutation symmetry unbroken and
variational inference converges to a label-switched optimum.
- Transitions B are near-deterministic and known, so states are
well-determined by actions and likelihood learning is well-conditioned.
- Inference: structured VMP with mean-field cut q(s, A) = q(s)q(A),
q(A) initialized at the prior counts, q(s) initialized uniform.
- **Generation Date**: 2026-09-05 20:25:28

## POMDP Dimensions

- **Number of States**: 3
- **Number of Observations**: 3
- **Number of Actions**: 2

## Active Inference Matrices

### Available Matrices/Vectors:
- **A Matrix (Likelihood)**: 3×3 - Maps hidden states to observations
- **B Matrix (Transition)**: 3×3×2 - State transitions given actions
- **C Vector (Preferences)**: Length 3 - Preferences over observations
- **D Vector (Prior)**: Length 3 - Prior beliefs over states


## Generated Files

- `Dirichlet_Likelihood_Learning_Agent_activeinference.jl` - activeinference_jl simulation script


## Usage

Refer to the main activeinference_jl documentation for information on how to run the generated simulation scripts.

## Framework-Specific Information

- **Framework**: activeinference_jl
- **File Extension**: .jl
- **Multi-Modality Support**: ✅
- **Multi-Factor Support**: ✅
