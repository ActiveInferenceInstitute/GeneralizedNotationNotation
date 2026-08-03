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

A_level0[3,4,type=float]         # Level 0 likelihood: P(fast_obs | fast_state)
B_level0[4,4,3,type=float]       # Level 0 transitions: P(fast_state' | fast_state, fast_action)
C_level0[3,type=float]           # Level 0 preferences (modulated by Level 1)
D_level0[4,type=float]           # Level 0 prior over initial states
s_level0[4,1,type=float]         # Level 0 hidden state belief
o_level0[3,1,type=int]           # Level 0 observation
pi0[3,type=float]          # Level 0 policy
u_level0[1,type=int]             # Level 0 action
G0[pi0,type=float]         # Level 0 Expected Free Energy

# Level 1: Medium tactical (3 states, 4 obs, 3 actions)

A_level1[4,3,type=float]         # Level 1 likelihood: P(tactic_obs | tactic_state)
B_level1[3,3,3,type=float]       # Level 1 transitions
C_level1[4,type=float]           # Level 1 preferences (modulated by Level 2)
D_level1[3,type=float]           # Level 1 prior (modulated by Level 2 predictions)
s_level1[3,1,type=float]         # Level 1 hidden state belief
o_level1[4,1,type=float]         # Level 1 observation (= summary of Level 0 state trajectory)
pi1[3,type=float]          # Level 1 policy
u_level1[1,type=int]             # Level 1 action
G1[pi1,type=float]         # Level 1 Expected Free Energy

# Level 2: Slow strategic (2 states, 3 obs, 2 actions)

A_level2[3,2,type=float]         # Level 2 likelihood: P(strategy_obs | strategy_state)
B_level2[2,2,2,type=float]       # Level 2 transitions
C_level2[3,type=float]           # Level 2 preferences (fixed strategic goals)
D_level2[2,type=float]           # Level 2 prior over strategies
s_level2[2,1,type=float]         # Level 2 hidden state belief
o_level2[3,1,type=float]         # Level 2 observation (= summary of Level 1 outcomes)
pi2[2,type=float]          # Level 2 policy
u_level2[1,type=int]             # Level 2 action
G2[pi2,type=float]         # Level 2 Expected Free Energy

# Timescale parameters

tau_level0[1,type=float]         # Level 0 time constant (0.1s)
tau_level1[1,type=float]         # Level 1 time constant (1.0s)
tau_level2[1,type=float]         # Level 2 time constant (10.0s)

# Time

t[1,type=int]              # Global discrete time counter

## Connections

# Level 0 (fast) internal loop

D_level0>s_level0
s_level0-A_level0
A_level0-o_level0
C_level0>G0
G0>pi0
pi0>u_level0
B_level0>u_level0

# Level 1 (medium) internal loop

D_level1>s_level1
s_level1-A_level1
A_level1-o_level1
C_level1>G1
G1>pi1
pi1>u_level1
B_level1>u_level1

# Level 2 (slow) internal loop

D_level2>s_level2
s_level2-A_level2
A_level2-o_level2
C_level2>G2
G2>pi2
pi2>u_level2
B_level2>u_level2

# Top-down causal flow (context modulates subordinate levels)

s_level2>C_level1
s_level1>C_level0
s_level2>D_level1

# Bottom-up evidential flow (observations inform superior levels)

s_level0>o_level1
s_level1>o_level2

## InitialParameterization

# Level 0: Sensorimotor (fast, reflexive)

A_level0={
  (0.85, 0.05, 0.05, 0.05),
  (0.05, 0.85, 0.05, 0.05),
  (0.05, 0.05, 0.85, 0.05)
}

C_level0={(0.0, -1.0, 1.0)}
D_level0={(0.25, 0.25, 0.25, 0.25)}

# Level 1: Tactical

A_level1={
  (0.8, 0.1, 0.1),
  (0.1, 0.8, 0.1),
  (0.1, 0.1, 0.8),
  (0.1, 0.1, 0.1)
}

C_level1={(-0.5, 1.0, 1.5, -1.0)}
D_level1={(0.33, 0.33, 0.34)}

# Level 2: Strategic

A_level2={
  (0.9, 0.1),
  (0.1, 0.9),
  (0.1, 0.1)
}

C_level2={(-1.0, 2.0, 0.5)}
D_level2={(0.5, 0.5)}

# Timescale constants

tau_level0={(0.1)}
tau_level1={(1.0)}
tau_level2={(10.0)}

# B: per-level transition matrices (added for POMDP/pymdp rendering)

B_level0={
  ( (0.9,0.05,0.05,0.0), (0.05,0.9,0.05,0.0), (0.05,0.05,0.9,0.0), (0.0,0.0,0.0,1.0) ),
  ( (0.05,0.9,0.05,0.0), (0.9,0.05,0.05,0.0), (0.05,0.05,0.9,0.0), (0.0,0.0,0.0,1.0) ),
  ( (0.9,0.05,0.05,0.0), (0.05,0.9,0.05,0.0), (0.05,0.05,0.9,0.0), (0.0,0.0,0.0,1.0) )
}

B_level1={
  ( (0.9,0.05,0.05), (0.05,0.9,0.05), (0.05,0.05,0.9) ),
  ( (0.05,0.9,0.05), (0.9,0.05,0.05), (0.05,0.05,0.9) ),
  ( (0.9,0.05,0.05), (0.05,0.9,0.05), (0.05,0.05,0.9) )
}

B_level2={
  ( (0.95, 0.05), (0.05, 0.95) )
}

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

A_level0=FastLikelihoodMatrix
B_level0=FastTransitionMatrix
C_level0=FastPreferenceVector
D_level0=FastPrior
s_level0=FastHiddenState
o_level0=FastObservation
pi0=FastPolicyVector
u_level0=FastAction
G0=FastExpectedFreeEnergy
A_level1=TacticalLikelihoodMatrix
B_level1=TacticalTransitionMatrix
C_level1=TacticalPreferenceVector
D_level1=TacticalPrior
s_level1=TacticalHiddenState
o_level1=TacticalObservation
pi1=TacticalPolicyVector
u_level1=TacticalAction
G1=TacticalExpectedFreeEnergy
A_level2=StrategicLikelihoodMatrix
B_level2=StrategicTransitionMatrix
C_level2=StrategicPreferenceVector
D_level2=StrategicPrior
s_level2=StrategicHiddenState
o_level2=StrategicObservation
pi2=StrategicPolicyVector
u_level2=StrategicAction
G2=StrategicExpectedFreeEnergy
tau_level0=FastTimeConstant
tau_level1=TacticalTimeConstant
tau_level2=StrategicTimeConstant
t=Time

## ModelParameters

num_hidden_states: 24
num_obs: 36
num_actions: 3
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
