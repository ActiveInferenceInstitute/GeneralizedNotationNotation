# Meta-Awareness - Agent Scaffolding

## Overview

**Purpose**: Responsible for `Meta-Awareness` operations within the GNN pipeline architecture.
**Category**: Generated Pipeline Component
**Status**: Development

---

## Core Functionality

### Primary Responsibilities
Complete reproduction of Sandved-Smith et al. (2021) computational phenomenology results

This script reproduces all figures and results from:
"Towards a computational phenomenology of mental action: modelling meta-awareness 
and attentional control with deep parametric active inference"

Neuroscien

### Extracted Code Entities

- **Classes**: SandvedSmithModel
- **Functions**: bayesian_model_average, compare_models, compute_attentional_charge, compute_entropy_terms, demo_simulation_results, discrete_choice, display_results_summary, expected_free_energy, generate_oddball_sequence, main, normalise, plot_figure_10, plot_figure_11, plot_figure_7, plot_free_energy_dynamics, plot_precision_dynamics, policy_posterior, precision_weighted_likelihood, reproduce_figure_10, reproduce_figure_11, reproduce_figure_7, run_comprehensive_test, run_figure_10_simulation, run_figure_11_simulation, run_figure_7_simulation, run_simulation, save_all_figures, setup_likelihood_matrices, setup_matplotlib_style, setup_transition_matrices, softmax, softmax_dim2, test_figure_modes, test_mind_wandering_dynamics, test_model_consistency, test_precision_dynamics, test_three_level_vs_two_level, test_utility_functions, update_precision_beliefs, validate_against_paper, variational_free_energy, verify_behavioral_patterns, verify_consistency, verify_figure_reproduction, verify_implementation, verify_mathematical_operations, verify_parameters

## Implementation Details

This module follows the Thin Orchestrator Pattern. It is governed by the Zero-Mock testing policy.
