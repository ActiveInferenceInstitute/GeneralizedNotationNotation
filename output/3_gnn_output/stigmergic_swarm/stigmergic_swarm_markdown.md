## GNNVersionAndFlags
Version: 1.0

## ModelName
Stigmergic Swarm Active Inference

## ModelAnnotation
Three Active Inference agents coordinating via stigmergy (environmental traces):

- No direct communication between agents — coordination emerges from environment
- Agents deposit and sense environmental signals (pheromone analogy)
- Shared 3x3 grid environment with signal intensity at each cell
- Each agent navigates independently while responding to accumulated signals
- Signal deposition: actions leave traces that other agents can observe
- Signal decay: environmental signals decay over time (volatility)
- Demonstrates emergent collective behavior from individual free energy minimization
- Models ant colony foraging, distributed robotics, and decentralized coordination

## StateSpaceBlock
A1[4,9],float
B1[9,9,4],float
C1[4],float
D1[9],float
s1[9,1],float
o1[4,1],integer
pi1[4],float
u1[1],integer
G1[1],float
A2[4,9],float
B2[9,9,4],float
C2[4],float
D2[9],float
s2[9,1],float
o2[4,1],integer
pi2[4],float
u2[1],integer
G2[1],float
A3[4,9],float
B3[9,9,4],float
C3[4],float
D3[9],float
s3[9,1],float
o3[4,1],integer
pi3[4],float
u3[1],integer
G3[1],float
env_signal[9,1],float
signal_decay[1],float
t[1],integer

## Connections
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
D3>s3
s3-A3
A3-o3
C3>G3
G3>pi3
pi3>u3
B3>u3
u1>env_signal
u2>env_signal
u3>env_signal
env_signal-A1
env_signal-A2
env_signal-A3
signal_decay>env_signal

## InitialParameterization
A1 = [[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]]
A2 = [[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]]
A3 = [[0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.1], [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.1], [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.7]]
C1 = [[-0.5, 0.5, 1.5, 3.0]]
C2 = [[-0.5, 0.5, 1.5, 3.0]]
C3 = [[-0.5, 0.5, 1.5, 3.0]]
D1 = [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
D2 = [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
D3 = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]]
env_signal = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
signal_decay = [[0.9]]

## Time
Dynamic
ModelTimeHorizon = 30

## ActInfOntologyAnnotation
A1 = Agent1LikelihoodMatrix
B1 = Agent1TransitionMatrix
C1 = Agent1PreferenceVector
D1 = Agent1PositionPrior
s1 = Agent1PositionState
o1 = Agent1Observation
pi1 = Agent1PolicyVector
u1 = Agent1Action
G1 = Agent1ExpectedFreeEnergy
A2 = Agent2LikelihoodMatrix
B2 = Agent2TransitionMatrix
C2 = Agent2PreferenceVector
D2 = Agent2PositionPrior
s2 = Agent2PositionState
o2 = Agent2Observation
pi2 = Agent2PolicyVector
u2 = Agent2Action
G2 = Agent2ExpectedFreeEnergy
A3 = Agent3LikelihoodMatrix
B3 = Agent3TransitionMatrix
C3 = Agent3PreferenceVector
D3 = Agent3PositionPrior
s3 = Agent3PositionState
o3 = Agent3Observation
pi3 = Agent3PolicyVector
u3 = Agent3Action
G3 = Agent3ExpectedFreeEnergy
env_signal = EnvironmentalSignal
signal_decay = SignalDecayRate
t = Time

## Footer
Generated: 2026-03-18T10:10:54.811227

## Signature
