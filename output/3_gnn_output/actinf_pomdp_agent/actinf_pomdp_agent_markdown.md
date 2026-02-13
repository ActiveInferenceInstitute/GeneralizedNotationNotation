## GNNVersionAndFlags
Version: 1.0

## ModelName
Active Inference POMDP Agent

## ModelAnnotation
This model describes a classic Active Inference agent for a discrete POMDP:
- One observation modality ("state_observation") with 3 possible outcomes.
- One hidden state factor ("location") with 3 possible states.
- The hidden state is fully controllable via 3 discrete actions.
- The agent's preferences are encoded as log-probabilities over observations.
- The agent has an initial policy prior (habit) encoded as log-probabilities over actions.

## StateSpaceBlock
A[3,3],float
B[3,3,3],float
C[3],float
D[3],float
E[3],float
s[3,1],float
s_prime[3,1],float
F[1],float
o[3,1],integer
π[3],float
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
E>π
G>π
π>u
B>u
u>s_prime

## InitialParameterization
A = [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
B = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]]
C = [[0.1, 0.1, 1.0]]
D = [[0.33333, 0.33333, 0.33333]]
E = [[0.33333, 0.33333, 0.33333]]
num_actions: 3       # B actions_dim = 3 (controlled by π)
num_timesteps: 30    # Number of simulation timesteps for all frameworks

## Time
Dynamic
ModelTimeHorizon = Unbounded # The agent is defined for an unbounded time horizon; simulation runs may specify a finite horizon.

## ActInfOntologyAnnotation
A = LikelihoodMatrix
B = TransitionMatrix
C = LogPreferenceVector
D = PriorOverHiddenStates
E = Habit
F = VariationalFreeEnergy
G = ExpectedFreeEnergy
s = HiddenState
s_prime = NextHiddenState
o = Observation
π = PolicyVector # Distribution over actions
u = Action       # Chosen action
t = Time

## Footer
Generated: 2026-02-13T11:19:11.543121

## Signature
