## GNNVersionAndFlags
Version: 1.0

## ModelName
Three-Level Temporal Hierarchy Agent

## ModelAnnotation
A three-level hierarchical Active Inference agent with distinct temporal scales:

- Level 0 (fast, 100ms): Sensorimotor control — immediate reflexive responses
- Level 1 (medium, 1s): Tactical planning — goal-directed behavior sequences
- Level 2 (slow, 10s): Strategic planning — long-term objective management
- Top-down flow: Strategy sets tactical goals, tactics set sensorimotor preferences
- Bottom-up flow: Sensorimotor observations inform tactical beliefs, tactical outcomes inform strategy
- Each level maintains its own generative model with A, B, C, D matrices
- Timescale separation encoded via update ratios (Level 2 updates every 10 Level 0 steps)
- Demonstrates deep temporal models from Friston et al. hierarchical Active Inference

## StateSpaceBlock
A0[3,4],float
B0[4,4,3],float
C0[3],float
D0[4],float
s0[4,1],float
o0[3,1],integer
pi0[3],float
u0[1],integer
G0[1],float
A1[4,3],float
B1[3,3,3],float
C1[4],float
D1[3],float
s1[3,1],float
o1[4,1],float
pi1[3],float
u1[1],integer
G1[1],float
A2[3,2],float
B2[2,2,2],float
C2[3],float
D2[2],float
s2[2,1],float
o2[3,1],float
pi2[2],float
u2[1],integer
G2[1],float
tau0[1],float
tau1[1],float
tau2[1],float
t[1],integer

## Connections
D0>s0
s0-A0
A0-o0
C0>G0
G0>pi0
pi0>u0
B0>u0
D1>s1
s1-A1
A1-o1
C1>G1
G1>pi1
pi1>u1
B1>u1
D2>s2
s2-A2
A2-o2
C2>G2
G2>pi2
pi2>u2
B2>u2
s2>C1
s1>C0
s2>D1
s0>o1
s1>o2

## InitialParameterization
A0 = [[0.85, 0.05, 0.05, 0.05], [0.05, 0.85, 0.05, 0.05], [0.05, 0.05, 0.85, 0.05]]
C0 = [[0.0, -1.0, 1.0]]
D0 = [[0.25, 0.25, 0.25, 0.25]]
A1 = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.1]]
C1 = [[-0.5, 1.0, 1.5, -1.0]]
D1 = [[0.33, 0.33, 0.34]]
A2 = [[0.9, 0.1], [0.1, 0.9], [0.1, 0.1]]
C2 = [[-1.0, 2.0, 0.5]]
D2 = [[0.5, 0.5]]
tau0 = [[0.1]]
tau1 = [[1.0]]
tau2 = [[10.0]]

## Time
Dynamic
ModelTimeHorizon = 100

## ActInfOntologyAnnotation
A0 = FastLikelihoodMatrix
B0 = FastTransitionMatrix
C0 = FastPreferenceVector
D0 = FastPrior
s0 = FastHiddenState
o0 = FastObservation
pi0 = FastPolicyVector
u0 = FastAction
G0 = FastExpectedFreeEnergy
A1 = TacticalLikelihoodMatrix
B1 = TacticalTransitionMatrix
C1 = TacticalPreferenceVector
D1 = TacticalPrior
s1 = TacticalHiddenState
o1 = TacticalObservation
pi1 = TacticalPolicyVector
u1 = TacticalAction
G1 = TacticalExpectedFreeEnergy
A2 = StrategicLikelihoodMatrix
B2 = StrategicTransitionMatrix
C2 = StrategicPreferenceVector
D2 = StrategicPrior
s2 = StrategicHiddenState
o2 = StrategicObservation
pi2 = StrategicPolicyVector
u2 = StrategicAction
G2 = StrategicExpectedFreeEnergy
tau0 = FastTimeConstant
tau1 = TacticalTimeConstant
tau2 = StrategicTimeConstant
t = Time

## Footer
Generated: 2026-04-10T10:24:32.075442

## Signature
