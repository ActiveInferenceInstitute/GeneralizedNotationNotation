# Modules - Agent Scaffolding

## Overview

**Purpose**: Responsible for `Modules` operations within the GNN pipeline architecture.
**Category**: Generated Pipeline Component
**Status**: Development

---

## Core Functionality

### Primary Responsibilities
Recurrent Mixture Model (rMM) - Interaction and control module.

Implements the rMM from the AXIOM architecture, modeling dependencies
between objects, actions, and rewards for sparse interaction modeling. Transition Mixture Model (tMM) - Object dynamics module.

Implements the tMM from the AXIOM ar

### Extracted Code Entities

- **Classes**: ActiveInferencePlanning, IdentityMixtureModel, RecurrentMixtureModel, SlotMixtureModel, StructureLearning, TransitionMixtureModel
- **Functions**: apply_bmr, check_expansion, count_parameters, expand_contexts, expand_dynamics, expand_identities, expand_slots, get_complexity_metrics, get_planning_metrics, get_state_dict, get_summary, inference, load_state_dict, plan, reset_planning

## Implementation Details

This module follows the Thin Orchestrator Pattern. It is governed by the Zero-Mock testing policy.
