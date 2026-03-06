
# Processed by GNN Pipeline Template
# Original file: /Users/4d/Documents/GitHub/generalizednotationnotation/input/gnn_files/discrete/deep_planning_horizon.md
# Processed on: 2026-03-06T15:48:32.098436
# Options: {'verbose': True, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Deep Planning Horizon POMDP
# GNN Version: 1.0
# POMDP with T=5 planning horizon for multi-step policy evaluation.

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Deep Planning Horizon POMDP

## ModelAnnotation
An Active Inference POMDP with deep (T=5) planning horizon:
- Evaluates policies over 5 future timesteps before acting
- Uses rollout Expected Free Energy accumulation
- 4 hidden states, 4 observations, 4 actions
- Each action policy is a sequence of T actions: π = [a_1, a_2, ..., a_T]
- Enables sophisticated multi-step reasoning and delayed reward attribution

## StateSpaceBlock
# Generative model
A[4,4,type=float]      # Likelihood matrix
B[4,4,4,type=float]    # Transition matrix (4 actions)
C[4,type=float]        # Preferences (per observation)
D[4,type=float]        # Prior over initial states
E[64,type=float]       # Habit prior over policies (4^3 = 64 short policies)

# Current state
s[4,1,type=float]      # Current hidden state belief
o[4,1,type=int]        # Current observation

# Policy representation
π[64,type=float]       # Policy distribution (over T-step action sequences)
u[1,type=int]          # Selected first action from best policy

# Planning horizon rollouts (T=5 steps)
s_tau1[4,1,type=float] # Predicted state at tau=1
s_tau2[4,1,type=float] # Predicted state at tau=2
s_tau3[4,1,type=float] # Predicted state at tau=3
s_tau4[4,1,type=float] # Predicted state at tau=4
s_tau5[4,1,type=float] # Predicted state at tau=5

# EFE per timestep
G_tau1[64,type=float]  # EFE contribution at tau=1
G_tau2[64,type=float]  # EFE contribution at tau=2
G_tau3[64,type=float]  # EFE contribution at tau=3
G_tau4[64,type=float]  # EFE contribution at tau=4
G_tau5[64,type=float]  # EFE contribution at tau=5
G[64,type=float]       # Cumulative EFE (sum over horizon)

# Free Energy
F[π,type=float]        # Variational Free Energy for current state

# Time
t[1,type=int]          # Discrete time step (action timestep)

## Connections
D>s
s-A
A-o
s-F
o-F
E>π
G>π
s>s_tau1
B>s_tau1
s_tau1>s_tau2
B>s_tau2
s_tau2>s_tau3
B>s_tau3
s_tau3>s_tau4
B>s_tau4
s_tau4>s_tau5
A-s_tau1
A-s_tau2
A-s_tau3
A-s_tau4
A-s_tau5
C>G_tau1
C>G_tau2
C>G_tau3
C>G_tau4
C>G_tau5
G_tau1>G
G_tau2>G
G_tau3>G
G_tau4>G
G_tau5>G
G>π
π>u

## InitialParameterization
A={
  (0.9, 0.05, 0.025, 0.025),
  (0.05, 0.9, 0.025, 0.025),
  (0.025, 0.025, 0.9, 0.05),
  (0.025, 0.025, 0.05, 0.9)
}

B={
  ( (0.9,0.1,0.0,0.0), (0.0,0.9,0.1,0.0), (0.0,0.0,0.9,0.1), (0.1,0.0,0.0,0.9) ),
  ( (0.9,0.0,0.0,0.1), (0.1,0.9,0.0,0.0), (0.0,0.1,0.9,0.0), (0.0,0.0,0.1,0.9) ),
  ( (0.8,0.1,0.1,0.0), (0.0,0.8,0.1,0.1), (0.1,0.0,0.8,0.1), (0.1,0.1,0.0,0.8) ),
  ( (0.7,0.1,0.1,0.1), (0.1,0.7,0.1,0.1), (0.1,0.1,0.7,0.1), (0.1,0.1,0.1,0.7) )
}

C={(-1.0, -0.5, -0.5, 2.0)}
D={(0.25, 0.25, 0.25, 0.25)}

## Equations
# Policy selection over sequences π = [a_1,...,a_T]
# G(π) = sum_{τ=1}^{T} G_τ(π) where:
# G_τ(π) = -E_Q[log P(C|o_τ)] - E_Q[H[P(o_τ|s_τ)]]
# = instrumental value + epistemic value at step τ
# State rollouts: s_{τ+1} = B[:,:,a_τ] * s_τ

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=Unbounded

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
E=PolicyPrior
s=HiddenState
o=Observation
π=PolicySequenceDistribution
u=Action
G=CumulativeExpectedFreeEnergy
F=VariationalFreeEnergy
t=Time

## ModelParameters
num_hidden_states: 4
num_obs: 4
num_actions: 4
planning_horizon: 5
num_policies: 64
num_timesteps: 30

## Footer
Deep Planning Horizon POMDP v1 - GNN Representation.
T=5 horizon enables multi-step consequence reasoning.
Policy space grows exponentially with T: |actions|^T = 4^3 = 64 policies (pruned).

## Signature
Cryptographic signature goes here

