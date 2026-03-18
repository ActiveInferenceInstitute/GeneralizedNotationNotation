
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/continuous/predictive_coding_agent.md
# Processed on: 2026-03-17T16:41:05.757982
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Predictive Coding Active Inference Agent

# GNN Version: 1.0

# Continuous-state agent using hierarchical prediction error minimization

## GNNSection

ActInfContinuous

## GNNVersionAndFlags

GNN v1

## ModelName

Predictive Coding Active Inference Agent

## ModelAnnotation

A continuous-state Active Inference agent implementing predictive coding:

- Two-level predictive hierarchy: sensory predictions and dynamics predictions
- Prediction errors drive belief updating via gradient descent on free energy
- Precision-weighted prediction errors enable attentional modulation
- Sensory level: predicts observations from hidden causes
- Dynamics level: predicts state evolution from generative dynamics
- Action minimizes expected free energy by changing sensory input
- Uses generalized coordinates of motion (position, velocity, acceleration)
- Demonstrates the core predictive processing framework underlying Active Inference

## StateSpaceBlock

# Generalized coordinates of motion (hidden causes)

mu[3,1,type=float]         # State belief: mean of hidden cause (3D)
mu_dot[3,1,type=float]     # Velocity of hidden cause (first temporal derivative)
Sigma[3,3,type=float]      # Covariance of hidden cause belief

# Sensory prediction error hierarchy

e_s[4,1,type=float]        # Sensory prediction error: o - g(mu)
e_d[3,1,type=float]        # Dynamics prediction error: mu_dot - f(mu)

# Generative model parameters

g_params[12,type=float]    # Sensory mapping parameters: o = g(mu, g_params) + noise
f_params[9,type=float]     # Dynamics parameters: mu_dot = f(mu, f_params) + noise

# Precision matrices (inverse covariance of noise)

Pi_s[4,4,type=float]       # Sensory precision: confidence in observations
Pi_d[3,3,type=float]       # Dynamics precision: confidence in dynamics model

# Observations

o[4,1,type=float]          # Continuous observation vector (4D sensory input)

# Action

u[2,1,type=float]          # Continuous action (2D motor command)

# Free energy quantities

F[1,type=float]            # Variational Free Energy (scalar)
F_s[1,type=float]          # Sensory contribution to VFE
F_d[1,type=float]          # Dynamics contribution to VFE

# Preferences (target attractor)

mu_star[3,1,type=float]    # Desired state (prior expectation / set-point)

# Time

t[1,type=float]            # Continuous time

## Connections

mu-g_params
g_params-o
o-e_s
mu-e_s
Pi_s-e_s
e_s-F_s
mu-f_params
f_params-mu_dot
mu_dot-e_d
Pi_d-e_d
e_d-F_d
F_s>F
F_d>F
F>mu
F>u
mu_star-F
mu-Sigma
Sigma-Pi_s

## InitialParameterization

# Initial state belief: centered at origin

mu={(0.0), (0.0), (0.0)}
mu_dot={(0.0), (0.0), (0.0)}

# Initial covariance: moderate uncertainty

Sigma={
  (1.0, 0.0, 0.0),
  (0.0, 1.0, 0.0),
  (0.0, 0.0, 1.0)
}

# Sensory precision: high (trusts observations)

Pi_s={
  (8.0, 0.0, 0.0, 0.0),
  (0.0, 8.0, 0.0, 0.0),
  (0.0, 0.0, 8.0, 0.0),
  (0.0, 0.0, 0.0, 8.0)
}

# Dynamics precision: moderate

Pi_d={
  (4.0, 0.0, 0.0),
  (0.0, 4.0, 0.0),
  (0.0, 0.0, 4.0)
}

# Desired state: attractor at (1.0, 1.0, 0.5)

mu_star={(1.0), (1.0), (0.5)}

## Equations

# Sensory prediction error: e_s = o - g(mu)

# Dynamics prediction error: e_d = mu_dot - f(mu)

# Variational Free Energy

# F = (1/2) *e_s^T Pi_s e_s + (1/2)* e_d^T Pi_d e_d

# Belief update (gradient descent on F)

# d(mu)/dt = mu_dot - kappa_mu * dF/d(mu)

# Action (gradient descent on F w.r.t. action)

# d(u)/dt = -kappa_u *dF/d(u) = -kappa_u* dg/du^T Pi_s e_s

# This implements active inference: action changes sensory input to minimize prediction error

## Time

Time=t
Dynamic
Continuous
ModelTimeHorizon=5.0

## ActInfOntologyAnnotation

mu=BeliefMean
mu_dot=BeliefVelocity
Sigma=BeliefCovariance
e_s=SensoryPredictionError
e_d=DynamicPredictionError
g_params=SensoryMappingParameters
f_params=DynamicsParameters
Pi_s=SensoryPrecision
Pi_d=DynamicPrecision
o=Observation
u=ContinuousAction
F=VariationalFreeEnergy
F_s=SensoryFreeEnergy
F_d=DynamicFreeEnergy
mu_star=PriorExpectation
t=ContinuousTime

## ModelParameters

state_dim: 3
obs_dim: 4
action_dim: 2
dt: 0.01
simulation_time: 5.0
learning_rate_belief: 1.0
learning_rate_action: 0.1

## Footer

Predictive Coding Active Inference Agent v1 - GNN Representation.
Implements the core predictive processing architecture:
  perception = minimizing sensory prediction errors,
  action = minimizing expected sensory prediction errors.
Precision parameters enable attentional modulation.

## Signature

Cryptographic signature goes here

