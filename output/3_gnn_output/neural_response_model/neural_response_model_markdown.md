## GNNVersionAndFlags
Version: 1.0

## ModelName
Active Inference Neural Response Model v1

## ModelAnnotation
This model describes how a neuron responds to stimuli using Active Inference principles:
- One primary observation modality (firing_rate) with 4 possible activity levels
- Two auxiliary observation modalities (postsynaptic_potential, calcium_signal) for comprehensive monitoring
- Five hidden state factors representing different aspects of neural computation
- Three control factors for plasticity, channel modulation, and metabolic allocation
- The model captures key neural phenomena: membrane potential dynamics, synaptic plasticity (STDP-like), activity-dependent adaptation, homeostatic regulation, and metabolic constraints
- Preferences encode biologically realistic goals: stable firing rates, energy efficiency, and synaptic balance

## StateSpaceBlock
A[12,405],float
B[405,405,27],float
C[12],float
D[405],float
E[27],float
V_m[5,1],float
W[4,1],float
A[3,1],float
H[3,1],float
M[3,1],float
FR[4,1],float
PSP[3,1],float
Ca[3,1],float
P[3,1],float
C_mod[3,1],float
M_alloc[3,1],float
F[1],float
G[1],float
t[1],integer

## Connections
D>V_m
V_m>B
W>B
A>B
H>B
M>B
V_m>A
W>A
V_m>A
P>B
C_mod>B
M_alloc>B
C>G
E>P
G>P
P>C_mod
C_mod>M_alloc

## InitialParameterization
A = [[0.05, 0.15, 0.25, 0.55, 0.4, 0.4, 0.2, 0.1, 0.35, 0.55, 0.3, 0.45], [0.1, 0.2, 0.3, 0.4, 0.35, 0.45, 0.2, 0.15, 0.4, 0.45, 0.25, 0.4], [0.15, 0.25, 0.35, 0.25, 0.3, 0.5, 0.2, 0.2, 0.45, 0.35, 0.2, 0.35]]
B = []
C = [[0.1, 0.2, 0.4, 0.3, 0.15, 0.35, 0.5, 0.25, 0.35, 0.4, 0.25, 0.2]]
D = [[0.05, 0.15, 0.35, 0.35, 0.1, 0.2, 0.4, 0.3, 0.1, 0.4, 0.4, 0.2, 0.2, 0.6, 0.2, 0.15, 0.7, 0.15]]
E = [[0.2, 0.3, 0.5, 0.25, 0.5, 0.25, 0.25, 0.5, 0.25, 0.3, 0.4, 0.3, 0.25, 0.5, 0.25, 0.3, 0.4, 0.3, 0.35, 0.4, 0.25, 0.3, 0.45, 0.25, 0.35, 0.4, 0.25]]

## Time
Dynamic
ModelTimeHorizon = Unbounded # Neural model defined for continuous operation; simulations may specify finite duration.

## ActInfOntologyAnnotation
A = LikelihoodMatrix
B = TransitionMatrices
C = LogPreferenceVector
D = PriorOverHiddenStates
E = HabitVector
F = VariationalFreeEnergy
G = ExpectedFreeEnergy
V_m = MembranePotentialState
W = SynapticWeightFactor
A = AdaptationState
H = HomeostaticSetPoint
M = MetabolicState
FR = FiringRateObservation
PSP = PostsynapticPotentialObservation
Ca = CalciumSignalObservation
P = PlasticityControl
C_mod = ChannelModulation
M_alloc = MetabolicAllocation
t = TimeStep

## Footer
Generated: 2025-10-10T10:34:18.522806

## Signature
