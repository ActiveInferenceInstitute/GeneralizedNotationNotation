# GNN Example: Dirichlet Likelihood Learning Agent

# GNN Version: 1.0

# A POMDP agent that LEARNS its likelihood matrix A from observations

# The A matrix is a latent variable with a Dirichlet prior (dirichlet_A pseudo-counts);
# hidden-state inference and likelihood learning run jointly via variational message passing.

## GNNSection

ActInfPOMDP_Learning

## GNNVersionAndFlags

GNN v1

## ModelName

Dirichlet Likelihood Learning Agent

## ModelAnnotation

This model describes a discrete POMDP agent that learns its observation model:

- 3 hidden states, 3 observation outcomes, 2 actions (cycle, stay).
- The likelihood matrix A is NOT fixed: it is a latent DirichletCollection
  variable with prior pseudo-counts declared in dirichlet_A.
- The A values under InitialParameterization are the GROUND-TRUTH likelihood
  used by the environment to simulate observations; the agent never sees them
  directly and must recover them in q(A).
- The Dirichlet prior is identity-biased (diagonal 3.0, off-diagonal 1.0):
  the agent starts believing observations weakly track states. A fully
  uniform prior leaves the column-permutation symmetry unbroken and
  variational inference converges to a label-switched optimum.
- Transitions B are near-deterministic and known, so states are
  well-determined by actions and likelihood learning is well-conditioned.
- Inference: structured VMP with mean-field cut q(s, A) = q(s)q(A),
  q(A) initialized at the prior counts, q(s) initialized uniform.

## StateSpaceBlock

# Likelihood matrix: A[observation_outcomes, hidden_states] — LEARNED (latent)

A[3,3,type=float]    # Ground-truth observation model P(o|s); latent in the agent

# Transition matrix: B[next_state, previous_state, actions] — known

B[3,3,2,type=float]  # State transitions given previous state and action

# Preference vector: C[observation_outcomes]

C[3,type=float]      # Log-preferences over observations

# Prior vector: D[states]

D[3,type=float]      # Prior over initial hidden states (known start)

# Dirichlet prior pseudo-counts over the likelihood: dirichlet_A[observations, states]

dirichlet_A[3,3,type=float] # Prior counts for q(A) ~ DirichletCollection

# Hidden State

s[3,1,type=float]    # Current hidden state distribution
s_prime[3,1,type=float] # Next hidden state distribution

# Observation

o[3,1,type=int]      # Current observation index

# Policy and Control

π[2,type=float]      # Policy (distribution over actions)
u[1,type=int]        # Action taken
G[π,type=float]      # Expected Free Energy (per policy)

# Time

t[1,type=int]        # Discrete time step

## Connections

D>s
s-A
s>s_prime
A-o
s-B
C>G
G>π
π>u
B>u
u>s_prime

## InitialParameterization

# A: GROUND-TRUTH likelihood (columns = P(o|s)); the agent learns q(A) toward this

A={
  (0.85, 0.05, 0.10),
  (0.10, 0.90, 0.05),
  (0.05, 0.05, 0.85)
}

# B: 2 actions. Action 0 = cycle (1->2->3->1), action 1 = stay
# The transition tensor B is stored as (next_state, previous_state, action):
# the outer axis is the next state; within each slice, rows are previous
# states and columns are actions.

B={
  ( (0.1, 0.9), (0.0, 0.05), (0.9, 0.05) ),
  ( (0.9, 0.05), (0.1, 0.9), (0.0, 0.05) ),
  ( (0.0, 0.05), (0.9, 0.05), (0.1, 0.9) )
}

# C: Mild preference for observation 3

C={(0.0, 0.0, 1.0)}

# D: Known start state (state 1)

D={(1.0, 0.0, 0.0)}

# dirichlet_A: identity-biased Dirichlet prior pseudo-counts for the likelihood
# (rows = observations, columns = states; each column is one Dirichlet)

dirichlet_A={
  (3.0, 1.0, 1.0),
  (1.0, 3.0, 1.0),
  (1.0, 1.0, 3.0)
}

## Equations

# Joint state inference and likelihood learning via structured VMP:

# Generative model: A ~ DirichletCollection(dirichlet_A); s_1 ~ Cat(D);
#   s_t ~ Cat(B[:,:,u_{t-1}] s_{t-1}); o_t ~ Cat(A s_t)

# Variational posterior: q(s_{1:T}, A) = q(s_{1:T}) q(A)  (mean-field cut)

# Likelihood update (conjugate): q(A) = DirichletCollection(dirichlet_A + sum_t o_t q(s_t)^T)

# State update: q(s_t) proportional to exp( E_q(A)[ln A]^T o_t ) x chain messages

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
dirichlet_A=LikelihoodMatrixConcentrationParameters
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
num_actions: 2
num_timesteps: 15
# Joint state + likelihood learning needs more VMP sweeps than fixed-A
# inference; 40 iterations converges cleanly even at T=100 batch runs.
inference_iterations: 40

## Footer

Dirichlet Likelihood Learning Agent v1 - GNN Representation.
Likelihood learning (q(A) from DirichletCollection prior) jointly with state inference.

## Signature

Cryptographic signature goes here
