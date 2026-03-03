
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/discrete/multi_armed_bandit.md
# Processed on: 2026-03-03T08:15:07.779054
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Multi-Armed Bandit

# GNN Version: 1.0

# A classic multi-armed bandit formulated as a degenerate POMDP

# Single hidden state context, multiple actions yield stochastic rewards

# Tests the case where state dynamics are trivial but action-observation mapping matters

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

Multi Armed Bandit Agent

## ModelAnnotation

This model describes a 3-armed bandit as a degenerate POMDP:

- 3 hidden states representing the "reward context" (which arm is currently best).
- 3 observations representing reward signals (no-reward, small-reward, big-reward).
- 3 actions: pull arm 0, pull arm 1, or pull arm 2.
- Context switches slowly (sticky transitions), testing exploration vs exploitation.
- The agent prefers big-reward observations (observation 2).
- Tests the bandit structure: meaningful actions despite nearly-static state dynamics.

## StateSpaceBlock

# Likelihood matrix: A[observations, hidden_states]

A[3,3,type=float]     # Reward likelihood given context state

# Transition matrix: B[states_next, states_previous, actions]

B[3,3,3,type=float]   # Context transitions (mostly identity — context is sticky)

# Preference vector: C[observations]

C[3,type=float]       # Prefer big rewards

# Prior vector: D[states]

D[3,type=float]       # Prior over reward context

# Hidden State

s[3,1,type=float]     # Current reward context belief
s_prime[3,1,type=float] # Next context belief

# Observation

o[3,1,type=int]       # Reward observation

# Policy and Control

π[3,type=float]       # Policy over arms
u[1,type=int]         # Arm pulled
G[π,type=float]       # Expected Free Energy

# Time

t[1,type=int]         # Discrete time step

## Connections

D>s
s-A
A-o
s>s_prime
s-B
C>G
G>π
π>u
B>u
u>s_prime

## InitialParameterization

# A: Reward likelihood. Context 0 → arm 0 best, context 1 → arm 1 best, etc

# Columns are contexts, rows are reward observations (none/small/big)

A={
  (0.1, 0.5, 0.5),
  (0.3, 0.4, 0.3),
  (0.6, 0.1, 0.2)
}

# B: Context is sticky regardless of action (arms don't change the world)

B={
  ( (0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9) ),
  ( (0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9) ),
  ( (0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9) )
}

# C: Strongly prefer big reward (obs 2), mildly prefer small (obs 1)

C={(0.0, 1.0, 3.0)}

# D: Uniform prior — don't know which arm is best

D={(0.33333, 0.33333, 0.33333)}

## Equations

# Active Inference bandit

# qs = infer_states(reward_obs) — learn which context/arm is active

# G(pi) = EFE(pi) — trade off pragmatic (reward) and epistemic (info gain) value

# u ~ softmax(-G) — select arm balancing exploration vs exploitation

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
G=ExpectedFreeEnergy
s=HiddenState
s_prime=NextHiddenState
o=Observation
π=PolicyVector
u=Action
t=Time

## ModelParameters

num_hidden_states: 3
num_obs: 3
num_actions: 3
num_timesteps: 30

## Footer

Multi Armed Bandit Agent v1 - GNN Representation.
3-armed bandit as degenerate POMDP. Tests exploration vs exploitation with sticky context.

## Signature

Cryptographic signature goes here

