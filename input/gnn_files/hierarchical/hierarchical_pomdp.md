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
A_level1[4,4,type=float]     # Level 1 likelihood: observations x hidden states
B_level1[4,4,3,type=float]   # Level 1 transitions: next x prev x actions
C_level1[4,type=float]       # Level 1 preferences over observations
D_level1[4,type=float]       # Level 1 prior over hidden states
s_level1[4,1,type=float]     # Level 1 hidden state distribution
x_next1[4,1,type=float] # Level 1 next hidden state
o_level1[4,1,type=int]       # Level 1 observations
π1[3,type=float]       # Level 1 policy (actions)
u_level1[1,type=int]         # Level 1 action
G1[π1,type=float]      # Level 1 Expected Free Energy

# Level 2 (slow dynamics)
A_level2[4,2,type=float]     # Level 2 likelihood: maps context to Level 1 hidden state prior
B_level2[2,2,1,type=float]   # Level 2 transitions (context switches)
C_level2[2,type=float]       # Level 2 preferences over context
D_level2[2,type=float]       # Level 2 prior over contextual states
s_level2[2,1,type=float]     # Level 2 contextual hidden state
o_level2[4,1,type=float]     # Level 2 observation (= Level 1 hidden state distribution)
G2[1,type=float]       # Level 2 Expected Free Energy

# Time
t1[1,type=int]         # Fast timescale counter
t2[1,type=int]         # Slow timescale counter

## Connections
D_level1>s_level1
s_level1-A_level1
s_level1>x_next1
A_level1-o_level1
C_level1>G1
G1>π1
π1>u_level1
B_level1>u_level1
u_level1>x_next1
s_level1>o_level2
D_level2>s_level2
s_level2-A_level2
A_level2>D_level1
s_level2-B_level2
C_level2>G2
G2>s_level2

## InitialParameterization
A_level1={
  (0.85, 0.05, 0.05, 0.05),
  (0.05, 0.85, 0.05, 0.05),
  (0.05, 0.05, 0.85, 0.05),
  (0.05, 0.05, 0.05, 0.85)
}

B_level1={
  ( (1.0,0.0,0.0,0.0), (0.0,1.0,0.0,0.0), (0.0,0.0,1.0,0.0), (0.0,0.0,0.0,1.0) ),
  ( (0.0,1.0,0.0,0.0), (1.0,0.0,0.0,0.0), (0.0,0.0,0.0,1.0), (0.0,0.0,1.0,0.0) ),
  ( (0.0,0.0,1.0,0.0), (0.0,0.0,0.0,1.0), (1.0,0.0,0.0,0.0), (0.0,1.0,0.0,0.0) )
}

C_level1={(0.1, 0.1, 0.1, 1.0)}
D_level1={(0.25, 0.25, 0.25, 0.25)}

A_level2={
  (0.9, 0.1),
  (0.1, 0.9),
  (0.5, 0.5),
  (0.5, 0.5)
}

B_level2={
  ( (0.9, 0.1), (0.1, 0.9) )
}

C_level2={(0.0, 0.5, 0.0, 0.5)}
D_level2={(0.5, 0.5)}

## Equations
# Level 1: Standard Active Inference POMDP update equations
# Level 2: Slower Bayesian context inference
# Cross-level: A_level2 maps context s_level2 to modulated prior D_level1
# Hierarchical message passing: top-down (s_level2→D_level1), bottom-up (s_level1→o_level2)

## Time
Time=t1
Dynamic
Discrete
ModelTimeHorizon=Unbounded

## ActInfOntologyAnnotation
A_level1=LikelihoodMatrix
B_level1=TransitionMatrix
C_level1=LogPreferenceVector
D_level1=PriorOverHiddenStates
s_level1=HiddenState
o_level1=Observation
π1=PolicyVector
u_level1=Action
G1=ExpectedFreeEnergy
A_level2=HigherLevelLikelihoodMatrix
B_level2=ContextTransitionMatrix
s_level2=ContextualHiddenState
o_level2=HigherLevelObservation
G2=HigherLevelExpectedFreeEnergy

## ModelParameters
num_hidden_states: 8
num_obs: 16
num_actions: 3
num_timesteps: 20
num_hidden_states_l1: 4
num_obs_l1: 4
num_actions_l1: 3
num_context_states_l2: 2
timescale_ratio: 5

## Footer
Hierarchical Active Inference POMDP v1 - GNN Representation.
Level 2 updates every 5 Level 1 timesteps (timescale_ratio=5).
Demonstrates context-dependent behavior modulation.

## Signature
Cryptographic signature goes here
