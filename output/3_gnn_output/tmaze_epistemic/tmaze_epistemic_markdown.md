## GNNVersionAndFlags
Version: 1.0

## ModelName
T-Maze Epistemic Foraging Agent

## ModelAnnotation
The classic T-maze task from Active Inference literature (Friston et al.):

- Agent navigates a T-shaped maze with 4 locations: center, left arm, right arm, cue location
- Two observation modalities: location (where am I?) and reward/cue (what do I see?)
- Reward is hidden behind one of the two arms (left or right), determined by context
- Cue location provides partial information about which arm holds the reward
- Agent must decide: go directly to an arm (exploit) or visit cue location first (explore)
- Demonstrates epistemic foraging: Active Inference naturally balances exploration vs exploitation
- The Expected Free Energy decomposes into epistemic (information gain) + instrumental (reward) value

## StateSpaceBlock
s_loc[4,1],float
s_ctx[2,1],float
o_loc[4,1],integer
o_rew[3,1],integer
A_loc[4,4],float
A_rew[3,4,2],float
B_loc[4,4,4],float
B_ctx[2,2,1],float
C_loc[4],float
C_rew[3],float
D_loc[4],float
D_ctx[2],float
π[4],float
u[1],integer
G[1],float
G_epi[1],float
G_ins[1],float
F[1],float
t[1],integer

## Connections
D_loc>s_loc
D_ctx>s_ctx
s_loc-A_loc
A_loc-o_loc
s_loc-A_rew
s_ctx-A_rew
A_rew-o_rew
s_loc-B_loc
s_ctx-B_ctx
C_rew>G_ins
G_epi>G
G_ins>G
G>π
π>u
B_loc>u
s_loc-F
s_ctx-F
o_loc-F
o_rew-F

## InitialParameterization
A_loc = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
B_loc = [[[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0]]]
B_ctx = [[[1.0, 0.0], [0.0, 1.0]]]
C_loc = [[0.0, 0.0, 0.0, 0.0]]
C_rew = [[-1.0, 3.0, 0.0]]
D_loc = [[1.0, 0.0, 0.0, 0.0]]
D_ctx = [[0.5, 0.5]]

## Time
Dynamic
ModelTimeHorizon = 3

## ActInfOntologyAnnotation
A_loc = LocationLikelihoodMatrix
A_rew = RewardLikelihoodMatrix
B_loc = LocationTransitionMatrix
B_ctx = ContextTransitionMatrix
C_loc = LocationPreferenceVector
C_rew = RewardPreferenceVector
D_loc = LocationPrior
D_ctx = ContextPrior
s_loc = LocationHiddenState
s_ctx = ContextHiddenState
o_loc = LocationObservation
o_rew = RewardObservation
π = PolicyVector
u = Action
G = ExpectedFreeEnergy
G_epi = EpistemicValue
G_ins = InstrumentalValue
F = VariationalFreeEnergy
t = Time

## Footer
Generated: 2026-03-13T14:15:03.928697

## Signature
