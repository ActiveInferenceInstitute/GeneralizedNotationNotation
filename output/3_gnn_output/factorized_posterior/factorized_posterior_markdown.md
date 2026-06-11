## GNNVersionAndFlags
Version: 1.0

## ModelName
Factorized Posterior Agent

## ModelAnnotation
A mean-field factorized POMDP agent. The joint posterior over two
independent state factors `s_1` (location) and `s_2` (goal identity) is
approximated as the product of marginals Q(s_1, s_2) = Q(s_1) * Q(s_2).
This is the canonical simplification used in variational inference when
exact joint posteriors are computationally intractable.

- Two state factors: location (4 states), goal (2 states)
- Two observation modalities: visual (3 obs), proprioceptive (2 obs)
- Separate transition matrices B_1 (location × action) and B_2 (goal is static)
- Explicit factorization declared in ## Equations
- Tests multi-factor / multi-modality handling in the parser

## StateSpaceBlock
s_f0[4,1],float
s_f1[2,1],float
o_m0[3,1],integer
o_m1[2,1],integer
u[3,1],integer
A_m0[3,4,2],float
A_m1[2,4],float
B_f0[4,4,3],float
B_f1[2,2],float
D_f0[4,1],float
D_f1[2,1],float
C_m0[3,1],float
C_m1[2,1],float

## Connections
D_f0>s_f0
D_f1>s_f1
s_f0>B_f0
u>B_f0
B_f0>s_f0
s_f1>B_f1
B_f1>s_f1
s_f0>A_m0
s_f1>A_m0
A_m0>o_m0
s_f0>A_m1
A_m1>o_m1
C_m0-o_m0
C_m1-o_m1

## InitialParameterization
A_m0 = [[[0.7, 0.1, 0.1, 0.1], [0.1, 0.7, 0.1, 0.1], [0.2, 0.2, 0.8, 0.8]], [[0.1, 0.7, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1], [0.2, 0.2, 0.8, 0.8]]]
A_m1 = [[0.9, 0.1, 0.1, 0.1], [0.1, 0.9, 0.9, 0.9]]
D_f0 = [[0.25, 0.25, 0.25, 0.25]]
D_f1 = [[0.6, 0.4]]
C_m0 = [[0.0, 0.0, 1.0]]
C_m1 = [[0.5, 0.5]]
num_hidden_states_factor0 = 4
num_hidden_states_factor1 = 2
num_obs_modality0 = 3
num_obs_modality1 = 2
num_actions = 3
num_factors = 2
num_modalities = 2
num_timesteps = 15

## Time
Dynamic
DiscreteTime
ModelTimeHorizon = 15

## ActInfOntologyAnnotation
s_f0 = HiddenStateFactor0
s_f1 = HiddenStateFactor1
o_m0 = ObservationModality0
o_m1 = ObservationModality1
u = Action
A_m0 = LikelihoodMatrixModality0
A_m1 = LikelihoodMatrixModality1
B_f0 = TransitionMatrixFactor0
B_f1 = TransitionMatrixFactor1
D_f0 = PriorFactor0
D_f1 = PriorFactor1
C_m0 = PreferenceModality0
C_m1 = PreferenceModality1

## Footer
Generated: 2026-05-25T09:51:15.343102

## Signature
