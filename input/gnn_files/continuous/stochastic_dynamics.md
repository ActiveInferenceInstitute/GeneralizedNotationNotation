# GNN Example: Stochastic Continuous Dynamics

# GNN Version: 1.0

# Demonstrates explicit stochastic noise terms in Equations — the model
# is a stochastic differential equation (SDE) style dynamical system.

## GNNSection

ActInfContinuous

## GNNVersionAndFlags

GNN v1

## ModelName

Stochastic Continuous Dynamics Agent

## ModelAnnotation

A continuous-state Active Inference agent whose dynamics include
explicit process and observation noise. This sample exercises the
continuous-time path of the language: Time=Continuous, state as a
multi-dimensional vector, and Equations containing noise terms ε_state
and ε_obs that downstream renderers must handle.

- 4-dimensional continuous hidden state (e.g., position + velocity in 2D)
- 2-dimensional continuous observation (noisy position readout)
- Linear Gaussian dynamics: ẋ = Fx + Gu + ε_state
- Linear Gaussian observation: o = Hx + ε_obs
- Precision parameters γ_state (process) and γ_obs (observation)

Used to validate that downstream backends either generate SDE solvers
(Stan via latent-variable sampling, NumPyro) or raise a clean
"unsupported feature" warning (PyMDP, which is discrete-only).

## StateSpaceBlock

# Continuous 4-D hidden state (e.g., [x, y, ẋ, ẏ])

x[4,1,type=float]         # Continuous hidden state vector
o[2,1,type=float]         # Continuous observation vector
u[2,1,type=float]         # Continuous control input

# Linear dynamics parameters

F[4,4,type=float]         # State transition matrix (drift)
G[4,2,type=float]         # Control gain matrix
H[2,4,type=float]         # Observation matrix

# Noise / precision parameters (scalars)

gamma_state[1,type=float] # Process noise precision
gamma_obs[1,type=float]   # Observation noise precision

# Explicit noise terms (zero-mean Gaussian)

epsilon_state[4,1,type=float]  # Process noise ε ~ N(0, γ_state^-1 * I)
epsilon_obs[2,1,type=float]    # Observation noise ε ~ N(0, γ_obs^-1 * I)

## Connections

# Dynamics: next state depends on current state, control, and noise
(x, u, epsilon_state)>F
F>x

# Observation: current state + noise
(x, epsilon_obs)>H
H>o

# Precisions gate the noise distributions
gamma_state-epsilon_state
gamma_obs-epsilon_obs

## InitialParameterization

# F: near-identity drift with small velocity coupling
# (represents x_{t+1} = x_t + dt * [ẋ, ẏ, 0, 0])

F={
  (1.0, 0.0, 0.1, 0.0),
  (0.0, 1.0, 0.0, 0.1),
  (0.0, 0.0, 0.95, 0.0),
  (0.0, 0.0, 0.0, 0.95)
}

# G: control directly affects velocity

G={
  (0.0, 0.0),
  (0.0, 0.0),
  (1.0, 0.0),
  (0.0, 1.0)
}

# H: observe position only

H={
  (1.0, 0.0, 0.0, 0.0),
  (0.0, 1.0, 0.0, 0.0)
}

# Precisions (high precision = low noise)

gamma_state={(10.0)}
gamma_obs={(5.0)}

## Equations

# State evolution (SDE form):
# dx/dt = F * x + G * u + ε_state
# ε_state ~ N(0, γ_state^-1 * I_4)
#
# Observation model:
# o = H * x + ε_obs
# ε_obs ~ N(0, γ_obs^-1 * I_2)
#
# Variational free energy (Gaussian):
# F = 0.5 * γ_state * ||x_{t+1} - F*x_t - G*u_t||^2
#   + 0.5 * γ_obs * ||o - H*x||^2
#   - 0.5 * ln(γ_state) - 0.5 * ln(γ_obs)
#
# Expected free energy for policy π:
# G(π) = E_Q[0.5 * γ_obs * ||o - H*x||^2] + H[Q(x|π)]

## Time

Continuous
ContinuousTime=τ
ModelTimeHorizon=5.0

## ActInfOntologyAnnotation

x=ContinuousHiddenState
o=ContinuousObservation
u=ContinuousAction
F=DriftMatrix
G=ControlMatrix
H=ObservationMatrix
gamma_state=ProcessNoisePrecision
gamma_obs=ObservationNoisePrecision
epsilon_state=ProcessNoise
epsilon_obs=ObservationNoise

## ModelParameters

state_dim: 4
obs_dim: 2
control_dim: 2
num_timesteps: 50
time_horizon: 5.0
integration_step: 0.1

## Footer

Stochastic Continuous Dynamics Agent v1.0 — linear Gaussian SDE-style
model with explicit process and observation noise. Tests continuous-time
+ noise-term handling across backends.

## Signature

Cryptographic signature goes here
