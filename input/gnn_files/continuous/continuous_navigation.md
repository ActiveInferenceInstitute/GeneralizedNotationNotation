# GNN Example: Continuous State Navigation Agent
# GNN Version: 1.0
# Active Inference navigation agent (discrete POMDP equivalent of the continuous 2D navigator)

## GNNSection
ActInfPOMDP

## GNNVersionAndFlags
GNN v1

## ModelName
Continuous State Navigation Agent

## ModelAnnotation
An Active Inference navigation agent: the original continuous (Gaussian)
2D-navigation model is represented here as a discretized POMDP equivalent so
it renders and executes with pymdp, RxInfer, and the general simulation
frameworks:
- Hidden states: 3 discrete locations (start L0, corridor L1, goal L2)
- Observations: 3 noisy location readings
- Actions: 4 discrete actions (stay, +x, +y, -x)
- Preferences favor the goal location L2

## StateSpaceBlock
# Likelihood matrix: A[observation_outcomes, hidden_states]
A[3,3,type=float]    # Observation model (noisy location readings)

# Transition matrix: B[states_next, states_previous, actions]
B[3,3,4,type=float]  # State transitions given state and action

# Preference vector: C[observation_outcomes]
C[3,type=float]      # Log-preferences over observations (goal = L2)

# Prior vector: D[states]
D[3,type=float]      # Prior over initial hidden states (start at L0)

# Hidden state
s[3,1,type=float]    # Current hidden state distribution
s_prime[3,1,type=float] # Next hidden state distribution

# Observation
o[3,1,type=int]      # Current observation

# Action
u[1,type=int]        # Discrete action (0=stay, 1=+x, 2=+y, 3=-x)

# Free Energy quantities
F[1,type=float]      # Variational Free Energy (scalar)
G[1,type=float]      # Expected Free Energy (scalar)

# Time
t[1,type=float]      # Discrete time step

## Connections
D>s
s-A
A-o
s-B
B>u
u>s_prime
s>s_prime
C>G
G>u

## InitialParameterization
# A: noisy location readings — each location maps mostly to its own reading
A={
  (0.9, 0.05, 0.05),
  (0.05, 0.9, 0.05),
  (0.05, 0.05, 0.9)
}

# B: 4 actions. Action 0=stay, 1=+x (L0->L1), 2=+y, 3=-x (L1->L0)
B={
  ( (0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9) ),
  ( (0.05, 0.9, 0.05), (0.9, 0.05, 0.05), (0.05, 0.05, 0.9) ),
  ( (0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9) ),
  ( (0.05, 0.9, 0.05), (0.9, 0.05, 0.05), (0.05, 0.05, 0.9) )
}

# C: preference for the goal observation (L2)
C={(-1.0, -1.0, 3.0)}

# D: start at L0
D={(0.9, 0.05, 0.05)}


# Continuous (linear-Gaussian) parameterization — native RxInfer LGSSM rendering
# State x = (x, y) position of the continuous 2D Gaussian navigator.
# F: position persists between steps; movement enters as the control input u.
F={
  (1.0, 0.0),
  (0.0, 1.0)
}

# H: noisy location readings observe the 2D position directly (identity readout)
H={
  (1.0, 0.0),
  (0.0, 1.0)
}

# Q: process (motion) noise — modest slip per step (discrete B rows keep 0.9 mass)
Q={
  (0.05, 0.0),
  (0.0, 0.05)
}

# R: observation noise on the location readings (discrete A keeps 0.9 diagonal)
R={
  (0.1, 0.0),
  (0.0, 0.1)
}

# Gaussian prior over the initial position: start at L0 = origin, fairly certain (D puts 0.9 on L0)
prior_mean={(0.0, 0.0)}
prior_cov={
  (0.5, 0.0),
  (0.0, 0.5)
}

## Equations
# Standard Active Inference update equations
# State inference: qs = softmax(ln(A[o,:]) + ln(B[s_prev] @ pi))
# Policy inference: G(pi) = sum_t EFE(pi,t)
# Action selection: u ~ Categorical(softmax(-G))
# Discretized equivalent of the continuous (Gaussian) 2D navigator.

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=15

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrix
C=LogPreferenceVector
D=PriorOverHiddenStates
s=HiddenState
s_prime=NextHiddenState
o=Observation
u=Action
F=VariationalFreeEnergy
G=ExpectedFreeEnergy
t=Time

## ModelParameters
num_hidden_states: 3
num_obs: 3
num_actions: 4
num_timesteps: 15

## Footer
Continuous State Navigation Agent v1 - GNN Representation.
Discrete POMDP equivalent of the continuous (Gaussian) 2D navigator.

## Signature
Cryptographic signature goes here
