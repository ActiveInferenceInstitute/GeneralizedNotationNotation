# GNN Example: Factorized Posterior Model

# GNN Version: 1.0

# Demonstrates explicit mean-field factorization of the posterior
# Q(s) = Q(s_1) * Q(s_2) — a key computational simplification in
# variational inference when exact joint inference is intractable.

## GNNSection

ActInfFactorized

## GNNVersionAndFlags

GNN v1

## ModelName

Factorized Posterior Agent

## ModelAnnotation

A mean-field factorized POMDP agent. The joint posterior over two
independent state factors `s_1` (location) and `s_2` (goal identity) is
approximated as the product of marginals Q(s_1, s_2) = Q(s_1) * Q(s_2).
This is the canonical simplification used in variational inference when
exact joint posteriors are computationally intractable.

- Two state factors: location (4 states), goal (2 states)
- Two observation modalities: visual (3 obs), proprioceptive (2 obs)
- Separate transition matrices B_1 (location × action) and B_2 (goal is static)
- Explicit factorization declared in ## Equations
- Tests multi-factor / multi-modality handling in the parser

## StateSpaceBlock

# Two factorized state dimensions

s_f0[4,1,type=float]      # Factor 0: agent location (4 possible positions)
s_f1[2,1,type=float]      # Factor 1: goal identity (2 possible goals)

# Two observation modalities

o_m0[3,1,type=int]        # Modality 0: visual observation (3 visual cues)
o_m1[2,1,type=int]        # Modality 1: proprioceptive observation (2 body states)

# Actions

u[3,1,type=int]           # 3 possible actions: stay, forward, backward

# Likelihood matrices (one per modality)

A_m0[3,4,2,type=float]    # Visual likelihood: depends on both factors
A_m1[2,4,type=float]      # Proprioceptive likelihood: depends only on location

# Transition matrices (one per factor)

B_f0[4,4,3,type=float]    # Location transitions (depends on action)
B_f1[2,2,type=float]      # Goal transitions (typically identity — goal is static)

# Priors and preferences

D_f0[4,1,type=float]      # Prior over locations (uniform)
D_f1[2,1,type=float]      # Prior over goals
C_m0[3,1,type=float]      # Visual preferences
C_m1[2,1,type=float]      # Proprioceptive preferences

## Connections

# Factor-wise priors
D_f0>s_f0
D_f1>s_f1

# Transitions per factor
(s_f0, u)>B_f0
B_f0>s_f0
s_f1>B_f1
B_f1>s_f1

# Observations couple modalities to state factors
(s_f0, s_f1)>A_m0
A_m0>o_m0
s_f0>A_m1
A_m1>o_m1

# Preferences
C_m0-o_m0
C_m1-o_m1

## InitialParameterization

# A_m0: visual observation depends on (location, goal) jointly
# Simplified uniform mapping for each goal

A_m0={
  (
    (0.7, 0.1, 0.1, 0.1),
    (0.1, 0.7, 0.1, 0.1),
    (0.2, 0.2, 0.8, 0.8)
  ),
  (
    (0.1, 0.7, 0.1, 0.1),
    (0.7, 0.1, 0.1, 0.1),
    (0.2, 0.2, 0.8, 0.8)
  )
}

# A_m1: proprioceptive depends only on location

A_m1={
  (0.9, 0.1, 0.1, 0.1),
  (0.1, 0.9, 0.9, 0.9)
}

# D_f0: uniform over 4 locations

D_f0={(0.25, 0.25, 0.25, 0.25)}

# D_f1: slight bias toward goal 0

D_f1={(0.6, 0.4)}

# C_m0: prefer the third visual observation

C_m0={(0.0, 0.0, 1.0)}

# C_m1: neutral

C_m1={(0.5, 0.5)}

## Equations

# Mean-field factorization assumption:
# Q(s_f0, s_f1) = Q(s_f0) * Q(s_f1)
#
# Per-factor updates (coordinate ascent VI):
# Q(s_f0) = softmax(ln(D_f0) + Σ_m ln(Σ_{s_f1} A_m[s_f0, s_f1] * Q(s_f1)^T * o_m))
# Q(s_f1) = softmax(ln(D_f1) + Σ_m ln(Σ_{s_f0} A_m[s_f0, s_f1] * Q(s_f0)^T * o_m))
#
# Factorized policy posterior:
# Q(π) = softmax(-G(π))
# G(π) = Σ_m [E_Q[ln Q(o_m|π) - ln C_m] - H[Q(s_f0|π)] - H[Q(s_f1|π)]]

## Time

Dynamic
DiscreteTime=t
ModelTimeHorizon=15

## ActInfOntologyAnnotation

s_f0=HiddenStateFactor0
s_f1=HiddenStateFactor1
o_m0=ObservationModality0
o_m1=ObservationModality1
u=Action
A_m0=LikelihoodMatrixModality0
A_m1=LikelihoodMatrixModality1
B_f0=TransitionMatrixFactor0
B_f1=TransitionMatrixFactor1
D_f0=PriorFactor0
D_f1=PriorFactor1
C_m0=PreferenceModality0
C_m1=PreferenceModality1

## ModelParameters

num_hidden_states_factor0: 4
num_hidden_states_factor1: 2
num_obs_modality0: 3
num_obs_modality1: 2
num_actions: 3
num_factors: 2
num_modalities: 2
num_timesteps: 15

## Footer

Factorized Posterior Agent v1.0 — explicit mean-field Q(s) = Q(s_0) Q(s_1)
factorization. Two state factors × two observation modalities.

## Signature

Cryptographic signature goes here
