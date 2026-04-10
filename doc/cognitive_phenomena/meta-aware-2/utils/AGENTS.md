# Utils - Agent Scaffolding

## Module Overview

**Purpose**: Responsible for `Utils` operations within the GNN pipeline architecture.
**Category**: Generated Pipeline Component
**Status**: Development

---

## Core Functionality

### Primary Responsibilities
Mathematical Utilities for Meta-Awareness Active Inference Model

Generic, dimensionally-flexible mathematical operations for hierarchical active inference.
Supports arbitrary state space dimensions while maintaining numerical stability.

Part of the meta-aware-2 "golden spike" GNN-specified executa

### Extracted Code Entities

- **Classes**: MathUtils
- **Functions**: bayesian_model_average, check_matrix_dimensions, compute_attentional_charge, compute_entropy, compute_entropy_terms, compute_kl_divergence, create_identity_matrix, create_likelihood_matrix, create_transition_matrix, create_uniform_distribution, discrete_choice, entropy, expected_free_energy, generate_oddball_sequence, kl_div, log_softmax, normalise, normalize, policy_posterior, precision_weighted_likelihood, setup_likelihood_matrices, setup_transition_matrices, softmax, softmax_dim2, test_math_utils, update_precision_beliefs, validate_probability_matrix, variational_free_energy

## Implementation Details

This module follows the Thin Orchestrator Pattern. It is governed by the Zero-Mock testing policy.
