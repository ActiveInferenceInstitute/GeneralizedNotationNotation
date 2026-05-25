## GNNVersionAndFlags
Version: 1.0

## ModelName
Time-Varying Transition Dynamics Agent

## ModelAnnotation
A POMDP agent operating in a non-stationary environment. The key feature
is that the transition matrix `B` is indexed by time (`B_t`), capturing
dynamics that evolve across the planning horizon — e.g., shifting wind
patterns for a sailing agent, or changing opponent strategy in a
sequential game.

- 3 hidden states, 3 observations, 2 actions
- B_t: 3D transition tensor per timestep (shape: next_state × current_state × action)
- Agent must adapt belief updates each step to the current B_t
- Exercises time-varying matrix handling in renderers

This sample pushes the language extensions around time-indexed tensors
and tests downstream code generation when matrix literals are
timestep-dependent.

## StateSpaceBlock
A[3,3],float
B_t[3,3,2],float
C[3,1],float
D[3,1],float
s_t[3,1],float
s_t+1[3,1],float
o_t[3,1],integer
u_t[2,1],integer

## Connections
D>s_t
s_t>B_t
u_t>B_t
B_t>s_t+1
s_t-A
A-o_t
C-o_t

## InitialParameterization
A = [[0.85, 0.1, 0.05], [0.1, 0.8, 0.1], [0.05, 0.1, 0.85]]
B_t = [[[0.6, 0.1], [0.3, 0.1], [0.1, 0.8]], [[0.3, 0.1], [0.6, 0.6], [0.1, 0.3]], [[0.1, 0.8], [0.1, 0.3], [0.8, 0.1]]]
C = [[0.0, 0.0, 1.0]]
D = [[0.33, 0.33, 0.34]]
num_hidden_states = 3
num_obs = 3
num_actions = 2
num_timesteps = 10

## Time
Dynamic
DiscreteTime
ModelTimeHorizon = 10

## ActInfOntologyAnnotation
A = LikelihoodMatrix
B_t = TimeVaryingTransitionMatrix
C = PreferenceVector
D = Prior
s_t = HiddenState
o_t = Observation
u_t = Action

## Footer
Generated: 2026-05-25T09:51:15.188508

## Signature
