
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/hierarchical/hierarchical_pomdp.md
# Processed on: 2026-03-18T09:18:36.301333
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Hierarchical Active Inference POMDP
# GNN Version: 1.0
# Two-level hierarchical POMDP with slow higher-level and fast lower-level dynamics.

## GNNSection
ActInfPOMDP_Hierarchical

## GNNVersionAndFlags
GNN v1

## ModelName
Hierarchical Active Inference POMDP

## ModelAnnotation
A two-level hierarchical POMDP where:
- Level 1 (fast): 4 observations, 4 hidden states, 3 actions
- Level 2 (slow): 2 contextual states that modulate Level 1 likelihood
- Higher-level beliefs are updated at a slower timescale
- Top-down predictions constrain bottom-up inference at Level 1

## StateSpaceBlock
# Level 1 (fast dynamics)
A1[4,4,type=float]     # Level 1 likelihood: observations x hidden states
B1[4,4,3,type=float]   # Level 1 transitions: next x prev x actions
C1[4,type=float]       # Level 1 preferences over observations
D1[4,type=float]       # Level 1 prior over hidden states
s1[4,1,type=float]     # Level 1 hidden state distribution
s1_prime[4,1,type=float] # Level 1 next hidden state
o1[4,1,type=int]       # Level 1 observations
π1[3,type=float]       # Level 1 policy (actions)
u1[1,type=int]         # Level 1 action
G1[π1,type=float]      # Level 1 Expected Free Energy

# Level 2 (slow dynamics)
A2[4,2,type=float]     # Level 2 likelihood: maps context to Level 1 hidden state prior
B2[2,2,1,type=float]   # Level 2 transitions (context switches)
C2[2,type=float]       # Level 2 preferences over context
D2[2,type=float]       # Level 2 prior over contextual states
s2[2,1,type=float]     # Level 2 contextual hidden state
o2[4,1,type=float]     # Level 2 observation (= Level 1 hidden state distribution)
G2[1,type=float]       # Level 2 Expected Free Energy

# Time
t1[1,type=int]         # Fast timescale counter
t2[1,type=int]         # Slow timescale counter

## Connections
D1>s1
s1-A1
s1>s1_prime
A1-o1
C1>G1
G1>π1
π1>u1
B1>u1
u1>s1_prime
s1>o2
D2>s2
s2-A2
A2>D1
s2-B2
C2>G2
G2>s2

## InitialParameterization
A1={
  (0.85, 0.05, 0.05, 0.05),
  (0.05, 0.85, 0.05, 0.05),
  (0.05, 0.05, 0.85, 0.05),
  (0.05, 0.05, 0.05, 0.85)
}

B1={
  ( (1.0,0.0,0.0,0.0), (0.0,1.0,0.0,0.0), (0.0,0.0,1.0,0.0), (0.0,0.0,0.0,1.0) ),
  ( (0.0,1.0,0.0,0.0), (1.0,0.0,0.0,0.0), (0.0,0.0,0.0,1.0), (0.0,0.0,1.0,0.0) ),
  ( (0.0,0.0,1.0,0.0), (0.0,0.0,0.0,1.0), (1.0,0.0,0.0,0.0), (0.0,1.0,0.0,0.0) )
}

C1={(0.1, 0.1, 0.1, 1.0)}
D1={(0.25, 0.25, 0.25, 0.25)}

A2={
  (0.9, 0.1, 0.0, 0.0),
  (0.1, 0.9, 0.0, 0.0),
  (0.0, 0.0, 0.9, 0.1),
  (0.0, 0.0, 0.1, 0.9)
}

B2={
  ( (0.9, 0.1), (0.1, 0.9) )
}

C2={(0.1, 1.0)}
D2={(0.5, 0.5)}

## Equations
# Level 1: Standard Active Inference POMDP update equations
# Level 2: Slower Bayesian context inference
# Cross-level: A2 maps context s2 to modulated prior D1
# Hierarchical message passing: top-down (s2→D1), bottom-up (s1→o2)

## Time
Time=t1
Dynamic
Discrete
ModelTimeHorizon=Unbounded

## ActInfOntologyAnnotation
A1=LikelihoodMatrix
B1=TransitionMatrix
C1=LogPreferenceVector
D1=PriorOverHiddenStates
s1=HiddenState
o1=Observation
π1=PolicyVector
u1=Action
G1=ExpectedFreeEnergy
A2=HigherLevelLikelihoodMatrix
B2=ContextTransitionMatrix
s2=ContextualHiddenState
o2=HigherLevelObservation
G2=HigherLevelExpectedFreeEnergy

## ModelParameters
num_hidden_states_l1: 4
num_obs_l1: 4
num_actions_l1: 3
num_context_states_l2: 2
num_timesteps: 20
timescale_ratio: 5

## Footer
Hierarchical Active Inference POMDP v1 - GNN Representation.
Level 2 updates every 5 Level 1 timesteps (timescale_ratio=5).
Demonstrates context-dependent behavior modulation.

## Signature
Cryptographic signature goes here

