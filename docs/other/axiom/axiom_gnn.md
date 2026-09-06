# AXIOM-GNN: Comprehensive Specification for Representing AXIOM Models in Generalized Notation Notation

## Executive Summary

This document provides a comprehensive specification for representing all [AXIOM (Active eXpanding Inference with Object-centric Models)](https://arxiv.org/abs/2505.24784) architectures using Generalized Notation Notation (GNN). AXIOM represents a revolutionary approach to AI learning that combines Active Inference principles with object-centric modeling, achieving human-like learning efficiency through Bayesian mixture models rather than gradient-based optimization.

The full AXIOM codebase is available at: https://github.com/VersesTech/axiom/tree/main/axiom

## Technical Context

Based on comprehensive analysis of the AXIOM research and implementation, this specification addresses:

- **Revolutionary Performance**: AXIOM achieves 60% better performance, 7x faster learning, 39x computational efficiency, and 440x smaller model size compared to state-of-the-art deep reinforcement learning
- **Gradient-Free Learning**: Eliminates the need for backpropagation through variational Bayesian inference
- **Object-Centric Cognition**: Implements core cognitive priors about objects and their interactions
- **Expanding Architecture**: Dynamic model growth and Bayesian Model Reduction for optimal complexity
- **Active Inference Planning**: Expected free energy minimization with epistemic and pragmatic value

## 1. Introduction

### 1.1 AXIOM Architecture Overview

AXIOM employs four core Bayesian mixture models that work together to create an expanding, object-centric world model:

1. **Slot Mixture Model (sMM)**: Parses visual input into object-centric representations
2. **Identity Mixture Model (iMM)**: Assigns discrete identity codes to objects based on visual features
3. **Transition Mixture Model (tMM)**: Models object dynamics as piecewise linear trajectories
4. **Recurrent Mixture Model (rMM)**: Captures sparse object-object interactions and control dependencies

### 1.2 GNN Suitability for AXIOM

GNN's strengths align perfectly with AXIOM's architecture:

- **Bayesian Model Specification**: GNN's mathematical notation naturally represents priors, likelihoods, and posteriors
- **Mixture Model Support**: GNN can elegantly express mixture distributions and component expansions
- **Temporal Dynamics**: GNN's time specification handles both discrete-time updates and continuous trajectories
- **Object-Centric Variables**: GNN's variable indexing system (e.g., `s_f0`, `s_f1`) maps directly to AXIOM's slot structure
- **Hierarchical Structure**: GNN can represent the modular architecture of AXIOM's four mixture models

## 2. Core GNN-AXIOM Mapping Framework

### 2.1 Variable Naming Conventions

Following GNN syntax and Active Inference ontology:

```
# Object-centric slot variables
s_slot_k[K,continuous]     # Continuous slot features for K slots
z_slot_k[K,discrete]       # Discrete slot assignments
w_slot_k[K,binary]         # Slot presence indicators

# Mixture model assignments
z_smm[N,K]                 # Slot assignment for N pixels to K slots
z_imm[K,V]                 # Identity assignment for K slots to V types
s_tmm[K,L]                 # Transition mode for K slots across L dynamics
s_rmm[K,M]                 # Recurrent mode for K slots across M contexts

# Observations and actions
o_pixels[H,W,5]            # Pixel observations (RGB + coordinates)
u_action[1,discrete]       # Control actions
r_reward[1,continuous]     # Reward signal

# Temporal indices
t_current[1,discrete]      # Current timestep
h_horizon[1,discrete]      # Planning horizon
```

### 2.2 Model Parameters in GNN

```
# sMM Parameters
Theta_smm_A[5,7]           # Projection matrix for slot features to pixels
Theta_smm_B[2,7]           # Shape projection matrix
Theta_smm_sigma[K,3]       # Color variance parameters

# iMM Parameters  
Theta_imm_mu[V,5]          # Identity type means for color+shape
Theta_imm_Sigma[V,5,5]     # Identity type covariances
Theta_imm_pi[V]            # Identity mixing weights

# tMM Parameters
Theta_tmm_D[L,7,7]         # Linear dynamics matrices for L modes
Theta_tmm_b[L,7]           # Linear dynamics bias terms
Theta_tmm_pi[L]            # Transition mode mixing weights

# rMM Parameters
Theta_rmm_mu[M,F]          # Recurrent context means for M modes
Theta_rmm_Sigma[M,F,F]     # Recurrent context covariances
Theta_rmm_alpha[M,D]       # Categorical parameters for discrete features
Theta_rmm_pi[M]            # Recurrent mode mixing weights
```

## 3. Core AXIOM Models in GNN

### 3.1 Slot Mixture Model (sMM) Specification

```gnn
# GNN Specification: AXIOM Slot Mixture Model
GNNVersionAndFlags: 1.4

ModelName: AXIOM_SlotMixtureModel

ModelAnnotation: |
  Object-centric visual perception module that decomposes pixel observations
  into K competing object slots using Gaussian mixture modeling. Each slot
  represents continuous object features (position, color, shape) that generate
  pixel likelihoods through linear projections.

StateSpaceBlock:
  # Input observations
  o_pixels[N,5,continuous]     ### N pixels with RGB+XY coordinates
  
  # Slot representations  
  s_slot[K,7,continuous]       ### K slots with position(2) + color(3) + shape(2)
  z_slot_assign[N,K,binary]    ### Pixel-to-slot assignment variables
  
  # Model parameters
  Theta_smm_pi[K,continuous]   ### Slot mixing weights with stick-breaking prior

Connections:
  s_slot -> o_pixels           ### Slots generate pixel observations
  z_slot_assign -> o_pixels    ### Assignment determines which slot explains pixel
  Theta_smm_pi -> z_slot_assign ### Mixing weights determine assignment probabilities

InitialParameterization:
  # Slot features (position, color, shape)
  s_slot ~ N(mu_slot_prior, Sigma_slot_prior)
  
  # Pixel assignment probabilities  
  z_slot_assign[n,k] ~ Categorical(Theta_smm_pi)
  
  # Mixing weights with stick-breaking prior
  Theta_smm_pi ~ StickBreaking(alpha_smm=1.0)

Equations:
  # Generative model for pixels
  p(o_pixels[n] | s_slot, z_slot_assign[n]) = 
    ∏_{k=1}^K N(A·s_slot[k], diag([B·s_slot[k], σ_c[k]]))^{z_slot_assign[n,k]}
  
  # Where A selects position+color, B selects shape
  A = [I_2, I_3, 0_{3×2}]  # Select position(2) + color(3)  
  B = [0_{2×5}, I_2]       # Select shape(2)

Time:
  ModelTimeHorizon: T_max
  DiscreteTime: true
  Dynamic: true

ActInfOntologyAnnotation:
  s_slot: "object_state_representation"
  o_pixels: "sensory_observation" 
  z_slot_assign: "perceptual_binding"
  Theta_smm_pi: "prior_beliefs_about_objects"
```

### 3.2 Identity Mixture Model (iMM) Specification

```gnn
# GNN Specification: AXIOM Identity Mixture Model
GNNVersionAndFlags: 1.4

ModelName: AXIOM_IdentityMixtureModel

ModelAnnotation: |
  Object identity classification module that assigns discrete type labels
  to object slots based on their color and shape features. Enables
  type-specific rather than instance-specific learning of dynamics.

StateSpaceBlock:
  # Input from slots (color + shape features only)
  s_appearance[K,5,continuous]     ### Color(3) + shape(2) features from slots
  
  # Identity assignments
  z_identity[K,V,binary]           ### K slots assigned to V identity types
  
  # Identity type parameters
  Theta_imm_mu[V,5,continuous]     ### Type means for appearance features
  Theta_imm_Sigma[V,5,5,continuous] ### Type covariance matrices
  Theta_imm_pi[V,continuous]       ### Identity type mixing weights

Connections:
  s_appearance -> z_identity       ### Appearance determines identity
  z_identity -> Theta_imm_mu       ### Identity types have characteristic appearances
  Theta_imm_pi -> z_identity       ### Prior over identity types

InitialParameterization:
  # Identity assignments  
  z_identity[k,v] ~ Categorical(Theta_imm_pi)
  
  # Type parameters with conjugate NIW priors
  (Theta_imm_mu[v], Theta_imm_Sigma[v]) ~ NIW(m_v, κ_v, U_v, n_v)
  
  # Mixing weights with stick-breaking
  Theta_imm_pi ~ StickBreaking(alpha_imm=1.0)

Equations:
  # Likelihood of appearance given identity
  p(s_appearance[k] | z_identity[k], Theta_imm) = 
    ∏_{v=1}^V N(Theta_imm_mu[v], Theta_imm_Sigma[v])^{z_identity[k,v]}
  
  # Prior over type parameters  
  p(Theta_imm_mu[v], Theta_imm_Sigma[v]^{-1}) = NIW(m_v, κ_v, U_v, n_v)

Time:
  ModelTimeHorizon: T_max
  DiscreteTime: true  
  Dynamic: true

ActInfOntologyAnnotation:
  s_appearance: "object_feature_representation"
  z_identity: "object_categorization"
  Theta_imm_mu: "categorical_prototypes"
  Theta_imm_Sigma: "categorical_uncertainty"
```

### 3.3 Transition Mixture Model (tMM) Specification

```gnn
# GNN Specification: AXIOM Transition Mixture Model  
GNNVersionAndFlags: 1.4

ModelName: AXIOM_TransitionMixtureModel

ModelAnnotation: |
  Object dynamics module modeling each slot's evolution as piecewise linear
  trajectories. Represents a switching linear dynamical system (SLDS) where
  different linear modes capture distinct motion patterns (falling, bouncing, etc).

StateSpaceBlock:
  # Slot states across time
  s_slot_t[K,7,continuous]        ### Current slot states
  s_slot_t1[K,7,continuous]       ### Next slot states
  
  # Dynamics mode assignments
  s_tmm_mode[K,L,binary]          ### K slots assigned to L dynamics modes
  
  # Linear dynamics parameters
  Theta_tmm_D[L,7,7,continuous]   ### Linear transition matrices
  Theta_tmm_b[L,7,continuous]     ### Linear bias terms
  Theta_tmm_pi[L,continuous]      ### Mode mixing weights

Connections:
  s_slot_t -> s_slot_t1           ### Current state influences next state
  s_tmm_mode -> s_slot_t1         ### Mode selection determines dynamics
  Theta_tmm_D -> s_slot_t1        ### Linear dynamics transform states
  Theta_tmm_b -> s_slot_t1        ### Bias terms shift dynamics

InitialParameterization:
  # Mode assignments
  s_tmm_mode[k,l] ~ Categorical(Theta_tmm_pi)
  
  # Linear dynamics with uniform priors
  Theta_tmm_D[l] ~ Uniform(-1, 1)
  Theta_tmm_b[l] ~ Uniform(-1, 1)
  
  # Mode mixing weights
  Theta_tmm_pi ~ StickBreaking(alpha_tmm=1.0)

Equations:
  # Linear dynamics likelihood
  p(s_slot_t1[k] | s_slot_t[k], s_tmm_mode[k]) = 
    ∏_{l=1}^L N(Theta_tmm_D[l]·s_slot_t[k] + Theta_tmm_b[l], 2I)^{s_tmm_mode[k,l]}
  
  # Mode probability
  p(s_tmm_mode[k]) = Categorical(Theta_tmm_pi)

Time:
  ModelTimeHorizon: T_max
  DiscreteTime: true
  Dynamic: true

ActInfOntologyAnnotation:
  s_slot_t: "hidden_state_dynamics"
  s_tmm_mode: "dynamical_regime_selection"  
  Theta_tmm_D: "transition_model_parameters"
  Theta_tmm_b: "dynamical_bias_terms"
```

### 3.4 Recurrent Mixture Model (rMM) Specification

```gnn
# GNN Specification: AXIOM Recurrent Mixture Model
GNNVersionAndFlags: 1.4

ModelName: AXIOM_RecurrentMixtureModel

ModelAnnotation: |
  Interaction and control module that models dependencies between objects,
  actions, and rewards. Predicts next transition modes and rewards based on
  multi-object features, enabling sparse interaction modeling and planning.

StateSpaceBlock:
  # Multi-object context features
  f_continuous[K,F_c,continuous]   ### Continuous context (positions, distances)
  d_discrete[K,F_d,discrete]       ### Discrete context (identities, actions, rewards)
  
  # Context assignments to mixture components
  s_rmm_context[K,M,binary]        ### K slots assigned to M context modes
  
  # Output predictions
  s_tmm_next[K,L,binary]           ### Next transition mode predictions
  r_reward_next[1,continuous]      ### Next reward prediction
  
  # Model parameters
  Theta_rmm_mu[M,F_c,continuous]   ### Context means for continuous features
  Theta_rmm_Sigma[M,F_c,F_c,continuous] ### Context covariances
  Theta_rmm_alpha[M,F_d,continuous] ### Categorical parameters for discrete features
  Theta_rmm_pi[M,continuous]       ### Context mode mixing weights

Connections:
  f_continuous -> s_rmm_context    ### Continuous features determine context
  d_discrete -> s_rmm_context      ### Discrete features determine context  
  s_rmm_context -> s_tmm_next      ### Context predicts dynamics mode
  s_rmm_context -> r_reward_next   ### Context predicts reward
  
InitialParameterization:
  # Context assignments
  s_rmm_context[k,m] ~ Categorical(Theta_rmm_pi)
  
  # Continuous feature parameters
  (Theta_rmm_mu[m], Theta_rmm_Sigma[m]) ~ NIW(m_rmm, κ_rmm, U_rmm, n_rmm)
  
  # Discrete feature parameters  
  Theta_rmm_alpha[m,d] ~ Dirichlet(α_rmm)
  
  # Context mixing weights
  Theta_rmm_pi ~ StickBreaking(alpha_rmm=1.0)

Equations:
  # Joint context likelihood
  p(f_continuous[k], d_discrete[k] | s_rmm_context[k]) = 
    ∏_{m=1}^M [N(f_continuous[k]; Theta_rmm_mu[m], Theta_rmm_Sigma[m]) · 
               ∏_i Cat(d_discrete[k,i]; Theta_rmm_alpha[m,i])]^{s_rmm_context[k,m]}
  
  # Predictive distributions
  p(s_tmm_next[k] | s_rmm_context[k]) = Context-dependent categorical
  p(r_reward_next | s_rmm_context) = Context-dependent continuous

Time:
  ModelTimeHorizon: T_max
  DiscreteTime: true
  Dynamic: true

ActInfOntologyAnnotation:
  f_continuous: "spatial_interaction_features"
  d_discrete: "symbolic_interaction_features"
  s_rmm_context: "interaction_context_classification"
  s_tmm_next: "predicted_dynamics_regime"
  r_reward_next: "expected_utility"
```

## 4. Integrated AXIOM System Specification

### 4.1 Full System Integration

```gnn
# GNN Specification: Complete AXIOM Architecture
GNNVersionAndFlags: 1.4

ModelName: AXIOM_Complete_System

ModelAnnotation: |
  Integrated AXIOM agent combining all four mixture models for object-centric
  world modeling, learning, and planning. Implements online Bayesian structure
  learning with model expansion and reduction.

StateSpaceBlock:
  # === INPUTS ===
  o_pixels[N,5,continuous]         ### Pixel observations (RGB + XY)
  u_action[1,discrete]             ### Control actions  
  r_reward[1,continuous]           ### Reward signals
  
  # === OBJECT SLOTS ===
  s_slot[K,7,continuous]           ### Object slot features
  z_slot_present[K,binary]         ### Slot presence indicators
  z_slot_moving[K,binary]          ### Slot movement indicators
  
  # === MIXTURE ASSIGNMENTS ===
  z_smm[N,K,binary]                ### Pixel-to-slot assignments
  z_imm[K,V,binary]                ### Slot-to-identity assignments  
  s_tmm[K,L,binary]                ### Slot-to-dynamics assignments
  s_rmm[K,M,binary]                ### Slot-to-context assignments
  
  # === MODEL PARAMETERS ===
  Theta_smm[*]                     ### sMM parameters
  Theta_imm[*]                     ### iMM parameters
  Theta_tmm[*]                     ### tMM parameters
  Theta_rmm[*]                     ### rMM parameters
  
  # === PLANNING VARIABLES ===
  pi_policy[H,A,continuous]        ### Policy distributions over horizon H
  G_expected_free_energy[H,continuous] ### Expected free energy per timestep

Connections:
  # Perception pathway
  o_pixels > s_slot               ### Pixels to slots via sMM
  s_slot > z_imm                  ### Slots to identities via iMM
  
  # Dynamics pathway  
  s_slot > s_tmm                  ### Slots to dynamics via context
  s_rmm > s_tmm                   ### Context determines dynamics
  s_tmm > s_slot                  ### Dynamics evolve slots
  
  # Interaction pathway
  s_slot > s_rmm                  ### Slots to context via features
  u_action > s_rmm                ### Actions influence context
  r_reward > s_rmm                ### Rewards influence context
  s_rmm > r_reward                ### Context predicts rewards
  
  # Planning pathway
  s_slot > pi_policy              ### Current state influences policy
  G_expected_free_energy > pi_policy ### Free energy determines policy
  pi_policy > u_action            ### Policy determines actions

InitialParameterization:
  # Hierarchical Bayesian priors for all mixture models
  Theta_smm ~ PriorDistribution_sMM()
  Theta_imm ~ PriorDistribution_iMM()  
  Theta_tmm ~ PriorDistribution_tMM()
  Theta_rmm ~ PriorDistribution_rMM()
  
  # Initial slot configurations
  s_slot ~ N(μ_slot_init, Σ_slot_init)
  
  # Initial policy prior
  pi_policy ~ Uniform(action_space)

Equations:
  # === GENERATIVE MODEL ===
  p(o_pixels, s_slot, assignments, rewards | Theta, actions) = 
    ∏_t [p(o_pixels_t | s_slot_t, z_smm_t, Theta_smm) ·
          p(z_imm_t | s_slot_t, Theta_imm) ·
          p(s_slot_{t+1} | s_slot_t, s_tmm_t, Theta_tmm) ·
          p(s_tmm_t, r_reward_t | s_slot_t, u_action_t, s_rmm_t, Theta_rmm)]
  
  # === VARIATIONAL INFERENCE ===
  q(all_latents, all_parameters) = ∏_modules q(latents_module, params_module)
  
  Free_Energy = D_KL[q || p] ≥ -log p(observations)
  
  # === ACTIVE INFERENCE PLANNING ===
  G_expected_free_energy[τ] = -E_q[log p(r_τ | o_τ, π)] + 
                               D_KL[q(Theta_rmm | o_τ, π) || q(Theta_rmm)]
  
  π* = argmin_π ∑_{τ=0}^H G_expected_free_energy[τ]

Time:
  ModelTimeHorizon: T_planning
  DiscreteTime: true
  Dynamic: true

ActInfOntologyAnnotation:
  o_pixels: "sensory_observation"
  s_slot: "hidden_state_object_centric"
  u_action: "control_action"
  r_reward: "utility_signal"
  pi_policy: "policy_distribution"
  G_expected_free_energy: "expected_free_energy"
  z_smm: "perceptual_binding"
  z_imm: "object_categorization"
  s_tmm: "dynamical_regime"
  s_rmm: "interaction_context"
```

## 5. Advanced AXIOM Features in GNN

### 5.1 Structure Learning and Model Expansion

```gnn
# GNN Specification: AXIOM Structure Learning
ModelName: AXIOM_Structure_Learning

ModelAnnotation: |
  Online Bayesian structure learning implementing fast component addition
  and Bayesian Model Reduction (BMR) for mixture model optimization.

StateSpaceBlock:
  # Component counts (dynamic)
  K_slots[1,discrete]              ### Number of active slots
  V_identities[1,discrete]         ### Number of identity types
  L_dynamics[1,discrete]           ### Number of dynamics modes  
  M_contexts[1,discrete]           ### Number of context modes
  
  # Expansion thresholds
  tau_smm[1,continuous]            ### sMM expansion threshold
  tau_imm[1,continuous]            ### iMM expansion threshold
  tau_tmm[1,continuous]            ### tMM expansion threshold
  tau_rmm[1,continuous]            ### rMM expansion threshold
  
  # BMR schedule
  T_bmr[1,discrete]                ### BMR application interval
  n_bmr_pairs[1,discrete]          ### Number of merge candidates

Equations:
  # Component addition criterion
  AddComponent(module) = max_c ℓ_{t,c} < τ_module + log α_module
  
  # BMR merge criterion  
  MergeComponents(c1, c2) = F_merged < F_separate
  
  # Where F is variational free energy
  F = -∑_data log p(data | merged_params) + D_KL[q(merged_params) || p(merged_params)]

Time:
  ModelTimeHorizon: ∞
  DiscreteTime: true
  Dynamic: true

ActInfOntologyAnnotation:
  K_slots: "model_complexity_slots"
  tau_smm: "expansion_threshold_perception"
  T_bmr: "model_reduction_schedule"
```

### 5.2 Planning and Expected Free Energy

```gnn
# GNN Specification: AXIOM Active Inference Planning
ModelName: AXIOM_Planning

ModelAnnotation: |
  Active inference planning module implementing expected free energy
  minimization with utility maximization and information gain.

StateSpaceBlock:
  # Planning horizon
  H_planning[1,discrete]           ### Planning horizon length
  
  # Policy space
  pi_actions[H,A,continuous]       ### Action probabilities over horizon
  
  # Predicted trajectories  
  s_predicted[H,K,7,continuous]    ### Predicted slot trajectories
  r_predicted[H,continuous]        ### Predicted rewards
  
  # Information gain
  IG_epistemic[H,continuous]       ### Epistemic information gain
  U_pragmatic[H,continuous]        ### Pragmatic utility

Connections:
  s_slot > s_predicted             ### Current slots to predicted trajectories
  pi_actions > s_predicted         ### Actions influence predictions
  s_predicted > r_predicted        ### Trajectories determine rewards
  s_predicted > IG_epistemic       ### Trajectories provide information
  r_predicted > U_pragmatic        ### Rewards provide utility

Equations:
  # Expected free energy decomposition
  G[τ] = -E_q[log p(r_τ | s_τ, π)] - D_KL[q(Theta_rmm | s_τ, π) || q(Theta_rmm)]
       = -U_pragmatic[τ] - IG_epistemic[τ]
  
  # Optimal policy
  π*[τ] ∝ exp(-γ · G[τ])
  
  # Where γ is precision parameter
  
  # Utility expectation
  U_pragmatic[τ] = E_q(s_τ|π)[log p(r_τ | s_τ, π)]
  
  # Information gain
  IG_epistemic[τ] = D_KL[q(Theta_rmm | s_τ, π) || q(Theta_rmm)]

Time:
  ModelTimeHorizon: H_planning
  DiscreteTime: true
  Dynamic: true

ActInfOntologyAnnotation:
  H_planning: "planning_horizon"
  pi_actions: "policy_distribution"
  G: "expected_free_energy"
  U_pragmatic: "expected_utility"
  IG_epistemic: "expected_information_gain"
```

## 6. Implementation Guidelines

### 6.1 GNN File Organization Strategy

For a complete AXIOM implementation, we recommend creating separate GNN files organized as follows:

1. **`axiom_core_architecture.md`** - Main integrated system specification
2. **`axiom_slot_mixture_model.md`** - Detailed sMM specification
3. **`axiom_identity_mixture_model.md`** - Detailed iMM specification  
4. **`axiom_transition_mixture_model.md`** - Detailed tMM specification
5. **`axiom_recurrent_mixture_model.md`** - Detailed rMM specification
6. **`axiom_structure_learning.md`** - Online learning and BMR specification
7. **`axiom_planning.md`** - Active inference planning specification
8. **`axiom_gameworld_environments.md`** - Environment-specific adaptations
9. **`axiom_variational_inference.md`** - Coordinate ascent variational inference
10. **`axiom_bayesian_model_reduction.md`** - BMR algorithms and heuristics

### 6.2 Variable Indexing Conventions

Follow these GNN indexing patterns for AXIOM:

```
# Slot indexing
s_slot_k[K,D,type]     # K slots, D dimensions, specific type
k ∈ {1, 2, ..., K_max} # Slot indices

# Time indexing  
s_slot_t[K,D,type]     # Current timestep
s_slot_t1[K,D,type]    # Next timestep
t ∈ {0, 1, ..., T_max} # Time indices

# Mixture component indexing
z_component_m[M,type]  # M mixture components
m ∈ {1, 2, ..., M_max} # Component indices

# Pixel indexing
o_pixel_n[N,type]      # N pixels
n ∈ {1, 2, ..., N_max} # Pixel indices
```

### 6.3 Parameter Evolution Specification

```gnn
# Dynamic parameter updates in GNN
ModelName: AXIOM_Parameter_Evolution

StateSpaceBlock:
  # Parameter sufficient statistics
  SS_smm[K,*,continuous]           ### sMM sufficient statistics
  SS_imm[V,*,continuous]           ### iMM sufficient statistics
  SS_tmm[L,*,continuous]           ### tMM sufficient statistics
  SS_rmm[M,*,continuous]           ### rMM sufficient statistics
  
  # Update timestamps
  T_last_update[4,discrete]        ### Last update time per module

Equations:
  # Variational parameter updates (coordinate ascent)
  Theta_new = Update_VB(Theta_old, SS_current, prior_params)
  
  # Where Update_VB implements module-specific natural parameter updates
  # for exponential family distributions

Time:
  ModelTimeHorizon: ∞
  DiscreteTime: true
  Dynamic: true
```

## 7. Environmental Adaptations

### 7.1 Gameworld 10k Environment Specifications

```gnn
# GNN Specification: AXIOM Gameworld Environment Interface
ModelName: AXIOM_Gameworld_Interface

ModelAnnotation: |
  Environment-specific adaptations for Gameworld 10k benchmark suite,
  including game-specific object types and interaction patterns.

StateSpaceBlock:
  # Environment-specific constants
  H_screen[1,discrete]             ### Screen height (210)
  W_screen[1,discrete]             ### Screen width (160)
  N_games[1,discrete]              ### Number of games (10)
  
  # Game-specific object types
  V_game_objects[10,*,discrete]    ### Object types per game
  
  # Interaction patterns
  I_collision_types[*,discrete]    ### Collision interaction types
  I_reward_zones[*,continuous]     ### Spatial reward regions

Equations:
  # Game-specific adaptation functions
  AdaptObjectTypes(game_id) = V_game_objects[game_id]
  AdaptRewardFunction(game_id) = R_game_specific[game_id]
  AdaptPhysics(game_id) = Physics_rules[game_id]

ActInfOntologyAnnotation:
  V_game_objects: "environment_specific_object_types"
  I_collision_types: "interaction_affordances"
  I_reward_zones: "utility_landscape"
```

## 8. Performance and Scalability Considerations

### 8.1 Computational Complexity Specification

```gnn
# GNN Specification: AXIOM Computational Complexity
ModelName: AXIOM_Complexity_Analysis

ModelAnnotation: |
  Computational complexity analysis for AXIOM components,
  including parameter counts and inference costs.

StateSpaceBlock:
  # Model sizes
  P_smm[1,discrete]                ### sMM parameter count
  P_imm[1,discrete]                ### iMM parameter count  
  P_tmm[1,discrete]                ### tMM parameter count
  P_rmm[1,discrete]                ### rMM parameter count
  P_total[1,discrete]              ### Total parameter count
  
  # Computational costs
  C_inference[1,continuous]        ### Inference cost per timestep
  C_planning[1,continuous]         ### Planning cost per action
  C_structure_learning[1,continuous] ### Structure learning cost

Equations:
  # Parameter scaling
  P_smm = O(K × D_slot)
  P_imm = O(V × D_appearance^2)  
  P_tmm = O(L × D_slot^2)
  P_rmm = O(M × (D_continuous^2 + D_discrete))
  P_total = P_smm + P_imm + P_tmm + P_rmm
  
  # Computational scaling
  C_inference = O(N × K + K × V + K × L + K × M)
  C_planning = O(H × A × C_inference)
  C_structure_learning = O(BMR_frequency × M^2)

ActInfOntologyAnnotation:
  P_total: "model_complexity"
  C_inference: "computational_cost_inference"
  C_planning: "computational_cost_planning"
```

## 9. Extensions and Future Directions

### 9.1 Hierarchical AXIOM Specification

```gnn
# GNN Specification: Hierarchical AXIOM Extension
ModelName: AXIOM_Hierarchical

ModelAnnotation: |
  Hierarchical extension of AXIOM with multiple spatial and temporal scales,
  enabling compositional object understanding and multi-level planning.

StateSpaceBlock:
  # Hierarchical slot structure
  s_slot_level[L,K,D,continuous]   ### L levels, K slots per level
  
  # Cross-level connections
  z_composition[L,K,K,binary]      ### Compositional relationships
  
  # Multi-scale dynamics
  s_tmm_scale[L,K,L_dyn,binary]    ### Scale-specific dynamics

Connections:
  s_slot_level[l] > s_slot_level[l+1] ### Bottom-up composition
  s_slot_level[l+1] > s_slot_level[l] ### Top-down attention

ActInfOntologyAnnotation:
  s_slot_level: "hierarchical_object_representation"
  z_composition: "compositional_binding"
```

### 9.2 Multi-Agent AXIOM Specification

```gnn
# GNN Specification: Multi-Agent AXIOM
ModelName: AXIOM_MultiAgent

ModelAnnotation: |
  Multi-agent extension where multiple AXIOM agents interact,
  requiring Theory of Mind and collaborative planning.

StateSpaceBlock:
  # Agent-specific states
  s_agent_slot[N_agents,K,D,continuous] ### Slots per agent
  
  # Theory of Mind
  z_other_beliefs[N_agents,*,continuous] ### Beliefs about other agents
  
  # Communication
  c_messages[N_agents,N_agents,D_comm,discrete] ### Inter-agent messages

ActInfOntologyAnnotation:
  s_agent_slot: "multi_agent_state"
  z_other_beliefs: "theory_of_mind"
  c_messages: "agent_communication"
```

## 10. Validation and Testing Framework

### 10.1 AXIOM-GNN Validation Specification

```gnn
# GNN Specification: AXIOM Validation Framework
ModelName: AXIOM_Validation

ModelAnnotation: |
  Comprehensive validation framework for AXIOM-GNN specifications,
  including unit tests, integration tests, and performance benchmarks.

StateSpaceBlock:
  # Test metrics
  M_perceptual_accuracy[1,continuous]  ### Slot assignment accuracy
  M_dynamics_prediction[1,continuous]  ### Trajectory prediction error
  M_reward_prediction[1,continuous]    ### Reward prediction accuracy
  M_sample_efficiency[1,continuous]    ### Learning speed metric
  
  # Benchmark comparisons
  B_axiom_vs_drl[1,continuous]        ### AXIOM vs DRL performance
  B_computational_cost[1,continuous]   ### Relative computational cost
  
  # Robustness tests
  R_perturbation_recovery[1,continuous] ### Recovery from perturbations
  R_domain_transfer[1,continuous]      ### Cross-domain generalization

Equations:
  # Validation criteria
  Valid_AXIOM_GNN = (M_perceptual_accuracy > 0.9) ∧ 
                    (M_sample_efficiency > baseline) ∧
                    (B_computational_cost < DRL_baseline)

ActInfOntologyAnnotation:
  M_perceptual_accuracy: "perception_validation_metric"
  M_sample_efficiency: "learning_efficiency_metric"
  R_perturbation_recovery: "robustness_validation"
```

## 11. Code Generation and Practical Implementation

### 11.1 GNN-to-Code Translation Framework

The GNN specifications can be systematically translated to executable code using the following mapping:

```python
# Example translation pattern for sMM specification
class AxiomSlotMixtureModel:
    def __init__(self, K_slots=8, N_pixels=33600, alpha_smm=1.0):
        self.K = K_slots  # From GNN: K in s_slot[K,7,continuous]
        self.N = N_pixels  # From GNN: N in o_pixels[N,5,continuous]
        self.alpha_smm = alpha_smm  # From GNN: StickBreaking(alpha_smm=1.0)
        
        # Initialize parameters from GNN InitialParameterization
        self.init_parameters()
    
    def generative_model(self, s_slot, z_assignment):
        """Implements GNN Equations section"""
        # p(o_pixels[n] | s_slot, z_slot_assign[n])
        return self._pixel_likelihood(s_slot, z_assignment)
    
    def variational_inference(self, observations):
        """Implements variational E-M updates"""
        # Coordinate ascent as specified in GNN
        return self._update_posteriors(observations)
```

### 11.2 Integration with GNN Pipeline

AXIOM-GNN specifications integrate seamlessly with the existing GNN processing pipeline:

1. **Parsing**: GNN parser extracts AXIOM model specifications
2. **Validation**: Type checking ensures mathematical consistency  
3. **Code Generation**: PyMDP/RxInfer.jl renderers generate executable simulation code
4. **Execution**: AXIOM agents run in Gameworld 10k environments
5. **Analysis**: LLM modules provide interpretable explanations of learned behaviors

### 11.3 Multi-File GNN Coordination System

The AXIOM mega-theory requires careful coordination between multiple GNN files. Here's the proposed coordination framework:

```gnn
# GNN Specification: AXIOM Meta-Coordinator
GNNVersionAndFlags: 1.4

ModelName: AXIOM_Meta_Coordinator

ModelAnnotation: |
  Master coordination specification that orchestrates all AXIOM component
  GNN files and manages their interdependencies, shared variables, and
  communication protocols.

StateSpaceBlock:
  # Coordination variables
  sync_timestamp[1,discrete]       ### Global synchronization timestamp
  component_status[4,discrete]     ### Status of sMM, iMM, tMM, rMM
  
  # Shared variable registry
  shared_slots[K,7,continuous]     ### Globally shared slot representations
  shared_assignments[N,K,binary]   ### Globally shared assignments
  
  # Module communication
  msg_smm_to_imm[K,*,continuous]   ### sMM to iMM messages
  msg_imm_to_rmm[K,*,discrete]     ### iMM to rMM messages  
  msg_rmm_to_tmm[K,*,binary]       ### rMM to tMM messages
  msg_tmm_to_smm[K,*,continuous]   ### tMM to sMM messages

Connections:
  # Inter-module dependencies
  axiom_slot_mixture_model.s_slot -> shared_slots
  shared_slots -> axiom_identity_mixture_model.s_appearance
  axiom_identity_mixture_model.z_identity -> msg_imm_to_rmm
  msg_rmm_to_tmm -> axiom_transition_mixture_model.s_tmm_mode
  
  # Coordination flow
  sync_timestamp -> component_status
  component_status -> shared_variable_updates

InitialParameterization:
  # Global coordination parameters
  sync_timestamp ~ 0
  component_status ~ [READY, READY, READY, READY]
  
  # Communication protocols
  msg_protocols ~ StandardAxiomProtocol()

Equations:
  # Coordination invariants
  ∀t: ConsistentSlotRepresentation(shared_slots[t])
  ∀k: ConservedSlotIdentity(k, across_modules)
  
  # Synchronization protocol
  UpdateSync() = WaitForAll(component_status == READY) → 
                 BroadcastUpdate() → 
                 AdvanceTimestamp()

Time:
  ModelTimeHorizon: ∞
  DiscreteTime: true
  Dynamic: true

ActInfOntologyAnnotation:
  sync_timestamp: "temporal_coordination"
  shared_slots: "global_object_state"
  component_status: "modular_system_health"
```

### 11.4 Deployment Configuration

```toml
# axiom_deployment_config.toml
[axiom_core]
max_slots = 16
max_identities = 10
max_dynamics_modes = 20
max_context_modes = 100
coordination_mode = "synchronized"
meta_coordinator_enabled = true

[structure_learning]
bmr_interval = 500
expansion_threshold_smm = 0.1
expansion_threshold_imm = 0.1
expansion_threshold_tmm = 0.1
expansion_threshold_rmm = 0.1
global_bmr_coordination = true

[planning]
horizon = 16
action_space_size = 5
rollout_samples = 64
precision_gamma = 1.0
planning_coordination = "centralized"

[environment]
screen_height = 210
screen_width = 160
gameworld_suite = "gameworld10k"
perturbation_robustness = true

[gnn_coordination]
file_discovery_pattern = "axiom_*.md"
dependency_resolution = "topological_sort"
shared_variable_namespace = "axiom_global"
inter_module_communication = "message_passing"
```

## 12. Conclusion

This comprehensive specification demonstrates how GNN can elegantly represent the complete AXIOM architecture through its mathematical notation system. The modular approach allows for:

1. **Clear Separation of Concerns**: Each mixture model has its own GNN specification
2. **Mathematical Rigor**: All Bayesian relationships are explicitly represented
3. **Temporal Dynamics**: Time evolution and planning horizons are clearly specified
4. **Extensibility**: New modules and hierarchical extensions can be easily added
5. **Validation**: Comprehensive testing frameworks ensure correctness

The GNN representation of AXIOM provides a bridge between the theoretical foundations of Active Inference and practical implementation, enabling researchers to:

- **Understand** the complete AXIOM architecture through standardized notation
- **Implement** AXIOM systems using GNN as a specification language  
- **Extend** AXIOM with new capabilities following GNN conventions
- **Validate** implementations against mathematical specifications
- **Communicate** AXIOM concepts across different research communities

This specification serves as the foundation for generating multiple GNN files that collectively capture the full AXIOM system, supporting the development of next-generation AI systems that combine human-like learning efficiency with mathematical rigor.

## 13. Immediate Next Steps

To realize the full potential of AXIOM-GNN specifications:

1. **Generate Individual Module GNN Files**: Create the 10 separate GNN files outlined in Section 6.1
2. **Implement Meta-Coordinator**: Build the coordination system for multi-file GNN orchestration
3. **Develop Code Generators**: Extend existing GNN renderers to support AXIOM-specific constructs
4. **Create Validation Suite**: Implement the testing framework specified in Section 10.1
5. **Deploy Gameworld Integration**: Connect AXIOM-GNN specifications to Gameworld 10k environments
6. **Benchmark Performance**: Validate the claimed 60% performance improvement and 7x learning acceleration
7. **Scale to Complex Domains**: Extend beyond Gameworld to real-world applications

This comprehensive AXIOM-GNN specification represents a crucial bridge between cutting-edge AI research and practical implementation, enabling the democratization of human-level learning efficiency through principled mathematical specification.

---

**References:**
- [AXIOM Paper](https://arxiv.org/abs/2505.24784) - Heins et al., "AXIOM: Learning to Play Games in Minutes with Expanding Object-Centric Models"
- [AXIOM Code Repository](https://github.com/VersesTech/axiom/tree/main/axiom) - Full implementation reference
- [GNN Specification](https://zenodo.org/records/7803328) - Smékal & Friedman, "Generalized Notation Notation for Active Inference Models"
- [Active Inference Institute](https://www.activeinference.org/) - Community resources and documentation
- [Gameworld 10k](https://github.com/VersesTech/gameworld) - Benchmark environment suite
- [VERSES AI Research](https://www.verses.ai/) - AXIOM development team and additional resources