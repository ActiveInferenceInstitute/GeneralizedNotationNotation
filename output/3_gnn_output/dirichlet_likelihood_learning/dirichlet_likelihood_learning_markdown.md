## GNNVersionAndFlags
Version: 1.0

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
A[3,3],float
B[3,3,2],float
C[3],float
D[3],float
dirichlet_A[3,3],float
s[3,1],float
s_prime[3,1],float
o[3,1],integer
π[2],float
u[1],integer
G[1],float
t[1],integer

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
A = [[0.85, 0.05, 0.1], [0.1, 0.9, 0.05], [0.05, 0.05, 0.85]]
B = [[[0.1, 0.9], [0.0, 0.05], [0.9, 0.05]], [[0.9, 0.05], [0.1, 0.9], [0.0, 0.05]], [[0.0, 0.05], [0.9, 0.05], [0.1, 0.9]]]
C = [[0.0, 0.0, 1.0]]
D = [[1.0, 0.0, 0.0]]
dirichlet_A = [[3.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 3.0]]
num_hidden_states = 3
num_obs = 3
num_actions = 2
num_timesteps = 15
inference_iterations = 40

## Time
Dynamic
ModelTimeHorizon = Unbounded

## ActInfOntologyAnnotation
A = LikelihoodMatrix
B = TransitionMatrix
C = LogPreferenceVector
D = PriorOverHiddenStates
dirichlet_A = LikelihoodMatrixConcentrationParameters
G = ExpectedFreeEnergy
s = HiddenState
s_prime = NextHiddenState
o = Observation
π = PolicyVector
u = Action
t = Time

## Footer
Generated: 2026-09-05T20:30:53.630216

## Signature
