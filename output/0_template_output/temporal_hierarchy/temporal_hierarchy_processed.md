
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/hierarchical/temporal_hierarchy.md
# Processed on: 2026-04-15T12:24:33.758881
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Three-Level Temporal Hierarchy Agent

# GNN Version: 1.0

# Hierarchical Active Inference with three temporal scales

## GNNSection

ActInfPOMDP_Hierarchical

## GNNVersionAndFlags

GNN v1

## ModelName

Three-Level Temporal Hierarchy Agent

## ModelAnnotation

A three-level hierarchical Active Inference agent with distinct temporal scales:

- Level 0 (fast, 100ms): Sensorimotor control — immediate reflexive responses
- Level 1 (medium, 1s): Tactical planning — goal-directed behavior sequences
- Level 2 (slow, 10s): Strategic planning — long-term objective management
- Top-down flow: Strategy sets tactical goals, tactics set sensorimotor preferences
- Bottom-up flow: Sensorimotor observations inform tactical beliefs, tactical outcomes inform strategy
- Each level maintains its own generative model with A, B, C, D matrices
- Timescale separation encoded via update ratios (Level 2 updates every 10 Level 0 steps)
- Demonstrates deep temporal models from Friston et al. hierarchical Active Inference

## StateSpaceBlock

# Level 0: Fast sensorimotor (4 states, 3 obs, 3 actions)

A0[3,4,type=float]         # Level 0 likelihood: P(fast_obs | fast_state)
B0[4,4,3,type=float]       # Level 0 transitions: P(fast_state' | fast_state, fast_action)
C0[3,type=float]           # Level 0 preferences (modulated by Level 1)
D0[4,type=float]           # Level 0 prior over initial states
s0[4,1,type=float]         # Level 0 hidden state belief
o0[3,1,type=int]           # Level 0 observation
pi0[3,type=float]          # Level 0 policy
u0[1,type=int]             # Level 0 action
G0[pi0,type=float]         # Level 0 Expected Free Energy

# Level 1: Medium tactical (3 states, 4 obs, 3 actions)

A1[4,3,type=float]         # Level 1 likelihood: P(tactic_obs | tactic_state)
B1[3,3,3,type=float]       # Level 1 transitions
C1[4,type=float]           # Level 1 preferences (modulated by Level 2)
D1[3,type=float]           # Level 1 prior (modulated by Level 2 predictions)
s1[3,1,type=float]         # Level 1 hidden state belief
o1[4,1,type=float]         # Level 1 observation (= summary of Level 0 state trajectory)
pi1[3,type=float]          # Level 1 policy
u1[1,type=int]             # Level 1 action
G1[pi1,type=float]         # Level 1 Expected Free Energy

# Level 2: Slow strategic (2 states, 3 obs, 2 actions)

A2[3,2,type=float]         # Level 2 likelihood: P(strategy_obs | strategy_state)
B2[2,2,2,type=float]       # Level 2 transitions
C2[3,type=float]           # Level 2 preferences (fixed strategic goals)
D2[2,type=float]           # Level 2 prior over strategies
s2[2,1,type=float]         # Level 2 hidden state belief
o2[3,1,type=float]         # Level 2 observation (= summary of Level 1 outcomes)
pi2[2,type=float]          # Level 2 policy
u2[1,type=int]             # Level 2 action
G2[pi2,type=float]         # Level 2 Expected Free Energy

# Timescale parameters

tau0[1,type=float]         # Level 0 time constant (0.1s)
tau1[1,type=float]         # Level 1 time constant (1.0s)
tau2[1,type=float]         # Level 2 time constant (10.0s)

# Time

t[1,type=int]              # Global discrete time counter

## Connections

# Level 0 (fast) internal loop

D0>s0
s0-A0
A0-o0
C0>G0
G0>pi0
pi0>u0
B0>u0

# Level 1 (medium) internal loop

D1>s1
s1-A1
A1-o1
C1>G1
G1>pi1
pi1>u1
B1>u1

# Level 2 (slow) internal loop

D2>s2
s2-A2
A2-o2
C2>G2
G2>pi2
pi2>u2
B2>u2

# Top-down causal flow (context modulates subordinate levels)

s2>C1
s1>C0
s2>D1

# Bottom-up evidential flow (observations inform superior levels)

s0>o1
s1>o2

## InitialParameterization

# Level 0: Sensorimotor (fast, reflexive)

A0={
  (0.85, 0.05, 0.05, 0.05),
  (0.05, 0.85, 0.05, 0.05),
  (0.05, 0.05, 0.85, 0.05)
}

C0={(0.0, -1.0, 1.0)}
D0={(0.25, 0.25, 0.25, 0.25)}

# Level 1: Tactical

A1={
  (0.8, 0.1, 0.1),
  (0.1, 0.8, 0.1),
  (0.1, 0.1, 0.8),
  (0.1, 0.1, 0.1)
}

C1={(-0.5, 1.0, 1.5, -1.0)}
D1={(0.33, 0.33, 0.34)}

# Level 2: Strategic

A2={
  (0.9, 0.1),
  (0.1, 0.9),
  (0.1, 0.1)
}

C2={(-1.0, 2.0, 0.5)}
D2={(0.5, 0.5)}

# Timescale constants

tau0={(0.1)}
tau1={(1.0)}
tau2={(10.0)}

## Equations

# Each level runs standard Active Inference

# Perception: Q(s) = softmax(ln D + ln A^T o)

# Policy: pi = softmax(-G + ln E)

# EFE: G(pi) = epistemic + instrumental

# Cross-level interactions

# Top-down: C_lower = f(s_higher) — higher beliefs set lower preferences

# Bottom-up: o_higher = h(s_lower) — lower state trajectory summarized as higher observation

# Timescale separation: Level k updates every (tau_k / tau_0) Level 0 steps

## Time

Time=t
Dynamic
Discrete
ModelTimeHorizon=100

## ActInfOntologyAnnotation

A0=FastLikelihoodMatrix
B0=FastTransitionMatrix
C0=FastPreferenceVector
D0=FastPrior
s0=FastHiddenState
o0=FastObservation
pi0=FastPolicyVector
u0=FastAction
G0=FastExpectedFreeEnergy
A1=TacticalLikelihoodMatrix
B1=TacticalTransitionMatrix
C1=TacticalPreferenceVector
D1=TacticalPrior
s1=TacticalHiddenState
o1=TacticalObservation
pi1=TacticalPolicyVector
u1=TacticalAction
G1=TacticalExpectedFreeEnergy
A2=StrategicLikelihoodMatrix
B2=StrategicTransitionMatrix
C2=StrategicPreferenceVector
D2=StrategicPrior
s2=StrategicHiddenState
o2=StrategicObservation
pi2=StrategicPolicyVector
u2=StrategicAction
G2=StrategicExpectedFreeEnergy
tau0=FastTimeConstant
tau1=TacticalTimeConstant
tau2=StrategicTimeConstant
t=Time

## ModelParameters

num_levels: 3
num_states_l0: 4
num_obs_l0: 3
num_actions_l0: 3
num_states_l1: 3
num_obs_l1: 4
num_actions_l1: 3
num_states_l2: 2
num_obs_l2: 3
num_actions_l2: 2
timescale_ratio_1_0: 10
timescale_ratio_2_1: 10
num_timesteps: 100

## Footer

Three-Level Temporal Hierarchy Agent v1 - GNN Representation.
Fast (100ms sensorimotor), Medium (1s tactical), Slow (10s strategic).
Top-down: strategy → tactics → sensorimotor preferences.
Bottom-up: sensory evidence → tactical summaries → strategic outcomes.

## Signature

Cryptographic signature goes here

