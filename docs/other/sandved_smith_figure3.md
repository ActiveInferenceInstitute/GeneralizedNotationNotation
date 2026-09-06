# GNN Example: Sandved-Smith Figure 3 - Deep Generative Model for Policy Selection
# Format: Markdown representation of a Deep Active Inference model for meta-awareness and attentional control
# Version: 1.0
# This file represents the computational architecture from Sandved-Smith et al. (2021) Figure 3

## GNNSection
SandvedSmithFigure3

## GNNVersionAndFlags
GNN v1

## ModelName
Deep Generative Model for Policy Selection with Meta-Awareness

## ModelAnnotation
This model represents a deep parametric active inference architecture for modeling meta-awareness and attentional control.
Based on Sandved-Smith et al. (2021) "Towards a computational phenomenology of mental action: modelling meta-awareness and attentional control with deep parametric active inference."

The model includes:
- Single hidden state factor relating to observable modality
- Policy selection over cognitive states through hierarchical inference
- Precision control for attentional modulation
- Meta-awareness states that modulate confidence in lower-level mappings
- Dynamic perception and action in a unified framework

This architecture enables the emergence of opacity-transparency dynamics characteristic of conscious experience and deliberate cognitive control.

## StateSpaceBlock
# Likelihood matrices - A matrices (observation models)
A[2,2,type=float]                    # Likelihood mapping from hidden states to observations P(o|s)

# Transition matrices - B matrices (dynamics models)  
B[2,2,1,type=float]                  # Transition dynamics P(s'|s,π) conditioned on policy
B_pi[2,2,2,type=float]               # Policy-dependent transition matrices

# Preference vectors - C matrices (prior preferences)
C[2,type=float]                      # Prior preferences over observations P(o)

# Prior state distributions - D matrices
D[2,type=float]                      # Prior beliefs over initial states P(s_0)

# Policy priors - E matrices
E[2,type=float]                      # Prior beliefs over policies P(π)

# Expected free energy - G
G[2,type=float]                      # Expected free energy for each policy G_π

# Hidden states
s[2,1,type=continuous]               # Hidden state beliefs at current time
s_prev[2,1,type=continuous]          # Hidden state beliefs at previous time  
s_next[2,1,type=continuous]          # Hidden state beliefs at next time
s_bar[2,1,type=continuous]           # Posterior hidden state beliefs

# Observations
o[2,1,type=discrete]                 # Current observations
o_bar[2,1,type=continuous]           # Posterior observation beliefs
o_pred[2,1,type=continuous]          # Predicted observations

# Policies and actions
π[2,type=continuous]                 # Policy beliefs (posterior over policies)
π_bar[2,type=continuous]             # Updated policy posterior
u[1,type=int]                        # Selected action

# Precision parameters
γ_A[1,type=float]                    # Precision (inverse variance) of likelihood mapping
β_A[1,type=float]                    # Inverse precision parameter (β_A = 1/γ_A)
β_A_bar[1,type=float]                # Updated inverse precision

# Free energy components
F_π[2,type=float]                    # Variational free energy per policy
E_π[2,type=float]                    # Complexity term per policy

# Time indexing
t[1,type=int]                        # Current time step

## Connections
# Prior connections
D > s
E > π

# Observation model connections  
(s, γ_A) > A
A > o_pred
(A, s_bar) > o_bar

# Transition model connections
(s, π) > B_pi
B_pi > s_next
(s_prev, B_pi) > s

# Policy selection connections
(G, E_π, F_π) > π_bar
π_bar > u

# Precision control connections
γ_A > A
β_A > γ_A
β_A_bar > β_A

# State inference connections
(A, o, B_pi) > s_bar
(s_bar, C) > F_π
F_π > π

# Expected free energy computation
(C, o_pred, s_bar) > G
G > π

# Temporal connections
s > s_next
s_prev > s
t > (s, o, π)

## InitialParameterization
# Likelihood matrix A: P(o|s)
# Simple 2x2 mapping between states and observations
A={
  ((0.9, 0.1), (0.1, 0.9))  # High confidence mapping with some noise
}

# Transition matrix B: P(s'|s) for baseline (no action)
B={
  ((0.8, 0.2), (0.2, 0.8))  # Tendency to stay in current state
}

# Policy-dependent transitions B_π
B_pi={
  ((0.9, 0.1), (0.1, 0.9)),  # Policy 0: stay in current state
  ((0.3, 0.7), (0.7, 0.3))   # Policy 1: switch states
}

# Prior preferences C
C={(0.0, 1.0)}  # Preference for observing outcome 1

# Prior state beliefs D  
D={(0.5, 0.5)}  # Uniform prior over initial states

# Prior policy beliefs E
E={(0.5, 0.5)}  # Uniform prior over policies

# Precision parameters
γ_A=2.0         # Moderate precision on likelihood mapping
β_A=0.5         # Inverse precision (1/γ_A)

# Initial states
s={(0.5, 0.5)}           # Uncertain initial state
o={(1.0, 0.0)}           # Initial observation (outcome 0)
π={(0.5, 0.5)}           # Initial policy uncertainty

# Time
t=0

## Equations
# Policy posterior (softmax of negative expected free energy):
# π̄ = σ(-E_π - G_π)
# where σ is the softmax function

# State transition dynamics:
# s_{t+1} = B_π * s_t
# where B_π is the policy-dependent transition matrix

# Precision relationship:
# γ_A = 1/β_A

# Expected free energy per policy:
# G_π = Σ_t [o_{π,t} · (ln(o_{π,t}) - C) - diag(A · ln(A)) · s̄_{π,t}]
# where · denotes element-wise multiplication

# Variational free energy per policy:
# F_π = Σ_t [s̄_{π,t} · (ln(s̄_{π,t}) - ln(A) · o_t - 0.5 * ln(B_{t-1} * s̄_{π,t-1}) - 0.5 * ln(B_{t+1} * s̄_{π,t+1}))]

# Complete policy posterior:
# π̄ = σ(-E_π - G_π - F_π)

# State inference (posterior beliefs):
# s̄_{π,t} = σ(ln(B_{π,t-1} * s_{t-1}) + ln(Ā · ō_t) + ln(B_{π,t} * s_{t+1}))

# Precision update:
# β̄_A = β_A - Σ_t(ln(A) · (ō_t * s_t - Ā * s_t))

# Observation posterior:
# ō_t = δ(o_t)
# where δ is the Kronecker delta function

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=Unbounded
TemporalDepth=3  # Past, present, future dependencies

## ActInfOntologyAnnotation
A=LikelihoodMapping
B=TransitionDynamics
B_pi=PolicyDependentTransitions
C=PriorPreferences
D=PriorStateBeliefs
E=PriorPolicyBeliefs
G=ExpectedFreeEnergy
F_π=VariationalFreeEnergy
s=HiddenStates
s_bar=PosteriorHiddenStates
o=Observations
o_bar=PosteriorObservations
π=PolicyBeliefs
π_bar=PosteriorPolicyBeliefs
u=SelectedAction
γ_A=LikelihoodPrecision
β_A=InversePrecision
t=TimeStep

## ModelParameters
num_states: 2              # Binary state space
num_observations: 2        # Binary observation space  
num_policies: 2           # Two available policies
temporal_horizon: 3        # Past, present, future
precision_dynamics: true   # Enable precision learning
policy_selection: true     # Enable active policy selection

## Footer
Deep Generative Model for Policy Selection - Sandved-Smith et al. (2021) Figure 3
Computational Phenomenology of Mental Action and Meta-Awareness

## Signature
Creator: AI Assistant for GNN
Date: 2024-12-28
Source: Sandved-Smith, L., Hesp, C., Mattout, J., Friston, K., Lutz, A., & Ramstead, M. J. D. (2021)
Paper: "Towards a computational phenomenology of mental action: modelling meta-awareness and attentional control with deep parametric active inference"
Journal: Neuroscience of Consciousness, 2021(1), niab018
Status: Research implementation of deep parametric active inference for cognitive control 