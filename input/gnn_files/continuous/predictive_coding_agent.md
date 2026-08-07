# GNN Example: Predictive Coding Active Inference Agent

# GNN Version: 1.0

# Predictive coding agent (discrete POMDP equivalent of the continuous
# predictive-processing agent, for POMDP/pymdp and general framework rendering)

## GNNSection

ActInfPOMDP

## GNNVersionAndFlags

GNN v1

## ModelName

Predictive Coding Active Inference Agent

## ModelAnnotation

A predictive-coding Active Inference agent represented as a discrete POMDP
equivalent (the original continuous predictive-processing formulation is
preserved in the Equations prose):

- 3 hidden states (prediction-error regimes)
- 4 observation outcomes (sensory readouts)
- 2 actions (attend / explore)
- Precision-weighted likelihood and transition structure

## StateSpaceBlock

# Likelihood matrix: A[observation_outcomes, hidden_states]

A[4,3,type=float]    # Observation model (precision-weighted readouts)

# Transition matrix: B[states_next, states_previous, actions]

B[3,3,2,type=float]  # State transitions given state and action

# Preference vector: C[observation_outcomes]

C[4,type=float]      # Log-preferences over observations

# Prior vector: D[states]

D[3,type=float]      # Prior over initial hidden states

# Hidden state

s[3,1,type=float]    # Current hidden state distribution

# Observation

o[4,1,type=int]      # Current observation (one-hot encoded)

# Action

u[1,type=int]        # Action taken (0=attend, 1=explore)

# Free energy quantities

F[1,type=float]      # Variational Free Energy (scalar)

# Time

t[1,type=float]      # Discrete time step

## Connections

D>s
s-A
A-o
s-B
B>u
u>s
C>F
F>u

## InitialParameterization

# A: precision-weighted observation mapping (each state -> distinct readout)

A={
  (0.9, 0.05, 0.05),
  (0.05, 0.9, 0.05),
  (0.05, 0.05, 0.9),
  (0.05, 0.9, 0.05)
}

# B: 2 actions. Action 0=attend (self-persistent), 1=explore (state shift)

B={
  ( (0.9, 0.05, 0.05), (0.05, 0.9, 0.05), (0.05, 0.05, 0.9) ),
  ( (0.05, 0.9, 0.05), (0.9, 0.05, 0.05), (0.05, 0.05, 0.9) )
}

# C: neutral-to-positive preferences over readouts

C={(0.0, 0.5, 1.0, 0.5)}

# D: uniform prior over prediction-error regimes

D={(0.4, 0.4, 0.2)}


# Continuous (linear-Gaussian) parameterization — native RxInfer LGSSM rendering
# State mu = (mu, mu_dot): generalized coordinates of the predictive-coding belief.
# F: linearized dynamics f(mu) — mu integrates mu_dot (dt = 0.1); mu_dot leaks toward the flow
F={
  (1.0, 0.1),
  (0.0, 0.8)
}

# H: linearized sensory mapping g(mu) — identity readout of both generalized coordinates
H={
  (1.0, 0.0),
  (0.0, 1.0)
}

# Q: dynamics-error covariance Pi_d^-1 (dynamics precision Pi_d = 10)
Q={
  (0.1, 0.0),
  (0.0, 0.1)
}

# R: sensory-error covariance Pi_s^-1 (sensory precision Pi_s = 4)
R={
  (0.25, 0.0),
  (0.0, 0.25)
}

# Gaussian prior over initial beliefs: broad/uninformative (D is near-uniform)
prior_mean={(0.0, 0.0)}
prior_cov={
  (1.0, 0.0),
  (0.0, 1.0)
}

## Equations

# Predictive coding formulation (preserved from the continuous agent):
# Sensory prediction error: e_s = o - g(mu)
# Dynamics prediction error: e_d = mu_dot - f(mu)
# Variational Free Energy: F = (1/2) e_s^T Pi_s e_s + (1/2) e_d^T Pi_d e_d
# Belief update: d(mu)/dt = mu_dot - kappa_mu * dF/d(mu)
# Action: d(u)/dt = -kappa_u * dF/d(u)
#
# Discrete POMDP equivalent used for rendering/execution:
# State inference: qs = softmax(ln(A[o,:]) + ln(B[s_prev] @ pi))
# Action selection: u ~ Categorical(softmax(-G))

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
o=Observation
u=Action
F=VariationalFreeEnergy
t=Time

## ModelParameters

num_hidden_states: 3
num_obs: 4
num_actions: 2
num_timesteps: 15

## Footer

Predictive Coding Active Inference Agent v1 - GNN Representation.
Discrete POMDP equivalent of the continuous predictive-coding agent.

## Signature

Cryptographic signature goes here
