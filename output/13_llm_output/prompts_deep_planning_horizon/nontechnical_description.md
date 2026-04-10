# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

GNN Section
Deep Planning Horizon POMDP v1 - GNN Representation

# GNN Example: Deep Planning Horizon POMDP

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
- Each action policy is a sequence of T actions: π = [a_1,...,a_T]
- Enables sophisticated multi-step reasoning and delayed reward attribution

## StateSpaceBlock
# Generative model
A[4,4,type=float]      # Likelihood matrix
B[4,4,4,type=float]    # Transition matrix (4 actions)
C[4,type=float]        # Preferences (per observation)
D[4,type=float]        # Prior over initial states
E[64,type=float]       # Habit prior over policies (4^3 = 64 short policies)
G_tau1[64,type=float]  # EFE contribution at tau=1
G_tau2[64,type=float]  # EFE contribution at tau=2
G_tau3[64,type=float]  # EFE contribution at tau=3
G_tau4[64,type=float]  # EFE contribution at tau=4
G_tau5[64,type=float]  # EFE contribution at tau=5
G[64,type=float]       # Cumulative EFE (sum over horizon)

# Current state
s[4,1,type=float]      # Current hidden state belief
o[4,1,type=int]        # Current observation

# Policy representation
π[64,type=float]       # Policy distribution (over T-step action sequences)
u[1,type=int]          # Selected first action from best policy
G_tau1[64,type=float]  # EFE contribution at tau=1
G_tau2