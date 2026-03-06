
# Processed by GNN Pipeline Template
# Original file: /Users/4d/Documents/GitHub/generalizednotationnotation/input/gnn_files/precision/curiosity_driven_agent.md
# Processed on: 2026-03-06T15:48:32.102692
# Options: {'verbose': True, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Curiosity-Driven Active Inference Agent
# GNN Version: 1.0
# Agent with explicit epistemic value (information gain) driving exploration.

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Curiosity-Driven Active Inference Agent

## ModelAnnotation
An Active Inference agent with:
- Explicit epistemic value (information gain / Bayesian surprise) component in G
- Separate instrumental value (preference satisfaction) component
- Precision parameter γ weighting epistemic vs instrumental contributions
- 5 hidden states, 5 observations, 4 actions in a navigation context
- Agent is rewarded for reducing posterior uncertainty

## StateSpaceBlock
# Core generative model
A[5,5,type=float]      # Likelihood matrix
B[5,5,4,type=float]    # Transition matrix (4 actions: up/down/left/right)
C[5,type=float]        # Instrumental preference vector (goal observations)
D[5,type=float]        # Prior over hidden states (uniform = no preference)
E[4,type=float]        # Habit vector over actions

# State and observation
s[5,1,type=float]      # Hidden state belief
s_prime[5,1,type=float] # Next hidden state belief
o[5,1,type=int]        # Current observation

# Policy and free energy
π[4,type=float]        # Policy distribution over actions
u[1,type=int]          # Selected action
G[π,type=float]        # Total Expected Free Energy (epistemic + instrumental)
G_epi[π,type=float]    # Epistemic value component (information gain)
G_ins[π,type=float]    # Instrumental value component (preference satisfaction)

# Precision parameters
γ[1,type=float]        # Precision weighting epistemic vs instrumental value
F[π,type=float]        # Variational Free Energy for state inference

# Time
t[1,type=int]          # Discrete time step

## Connections
D>s
s-A
s>s_prime
A-o
C>G_ins
G_epi>G
G_ins>G
γ>G
E>π
G>π
π>u
B>u
u>s_prime
s-F
o-F

## InitialParameterization
A={
  (0.9, 0.025, 0.025, 0.025, 0.025),
  (0.025, 0.9, 0.025, 0.025, 0.025),
  (0.025, 0.025, 0.9, 0.025, 0.025),
  (0.025, 0.025, 0.025, 0.9, 0.025),
  (0.025, 0.025, 0.025, 0.025, 0.9)
}

B={
  ( (0.9,0.1,0.0,0.0,0.0), (0.1,0.8,0.1,0.0,0.0), (0.0,0.1,0.8,0.1,0.0), (0.0,0.0,0.1,0.8,0.1), (0.0,0.0,0.0,0.1,0.9) ),
  ( (0.9,0.1,0.0,0.0,0.0), (0.0,0.9,0.1,0.0,0.0), (0.0,0.0,0.9,0.1,0.0), (0.0,0.0,0.0,0.9,0.1), (0.0,0.0,0.0,0.0,1.0) ),
  ( (1.0,0.0,0.0,0.0,0.0), (0.1,0.9,0.0,0.0,0.0), (0.0,0.1,0.9,0.0,0.0), (0.0,0.0,0.1,0.9,0.0), (0.0,0.0,0.0,0.1,0.9) ),
  ( (0.9,0.0,0.0,0.0,0.1), (0.0,0.9,0.0,0.0,0.1), (0.0,0.0,0.9,0.0,0.1), (0.0,0.0,0.0,0.9,0.1), (0.0,0.0,0.0,0.0,1.0) )
}

# Prefer goal state (state 5 = index 4) with high log-probability
C={(-2.0, -2.0, -2.0, -2.0, 2.0)}

D={(0.2, 0.2, 0.2, 0.2, 0.2)}
E={(0.25, 0.25, 0.25, 0.25)}

# Precision: balanced epistemic/instrumental weighting
γ={(1.0)}

## Equations
# G = G_epi + γ * G_ins
# G_epi[π] = -E_Q[H[P(o|s)]] = expected information gain (Bayesian surprise)
# G_ins[π] = -E_Q[log P(C|o)] = expected preference satisfaction
# State inference: F = D_KL[Q(s)||P(s)] - E_Q[log P(o|s)]

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=30

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
E=Habit
s=HiddenState
s_prime=NextHiddenState
o=Observation
π=PolicyVector
u=Action
G=ExpectedFreeEnergy
G_epi=EpistemicValue
G_ins=InstrumentalValue
γ=PrecisionParameter
F=VariationalFreeEnergy
t=Time

## ModelParameters
num_hidden_states: 5
num_obs: 5
num_actions: 4
num_timesteps: 30
epistemic_weight: 1.0
instrumental_weight: 1.0

## Footer
Curiosity-Driven Active Inference Agent v1 - GNN Representation.
Demonstrates explicit epistemic value (information gain) alongside instrumental value.
Useful for studying exploration-exploitation in Active Inference.

## Signature
Cryptographic signature goes here

