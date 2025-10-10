
# Processed by GNN Pipeline Template
# Original file: input/gnn_files/neural_response_model.md
# Processed on: 2025-10-10T10:32:57.455787
# Options: {'verbose': False, 'recursive': True, 'example_param': 'default_value'}

# GNN Example: Neural Response Model
# GNN Version: 1.0
# This file specifies a comprehensive Active Inference model of how neurons respond to stimuli, including membrane dynamics, synaptic plasticity, adaptation, and homeostatic regulation. The model is suitable for rendering into various neural simulation backends.

## GNNSection
NeuralResponse

## GNNVersionAndFlags
GNN v1

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
# Likelihood matrix: A[observation_outcomes, hidden_states]
A[12,405,type=float]   # 12 observations x 405 hidden state combinations (likelihood mapping)

# Transition matrices: B[states_next, states_previous, actions]
B[405,405,27,type=float]   # State transitions given previous state and action (5 state factors, 3 control factors)

# Preference vector: C[observation_outcomes]
C[12,type=float]       # Log-preferences over observations

# Prior vector: D[states]
D[405,type=float]       # Prior over initial hidden states

# Habit vector: E[actions]
E[27,type=float]       # Initial policy prior (habit) over actions

# Hidden States (5 factors with 5×4×3×3×3 = 405 total combinations)
V_m[5,1,type=float]     # Membrane potential state (5 levels: hyperpolarized, resting, depolarized, threshold, refractory)
W[4,1,type=float]       # Synaptic weight factor (4 levels: weak, moderate, strong, saturated)
A[3,1,type=float]       # Adaptation state (3 levels: low, medium, high adaptation)
H[3,1,type=float]       # Homeostatic set point (3 levels: low, target, high firing rate)
M[3,1,type=float]       # Metabolic state (3 levels: depleted, adequate, surplus)

# Observations (3 modalities with 4×3×3 = 12 total outcomes)
FR[4,1,type=float]     # Firing rate (4 levels: silent, low, moderate, high)
PSP[3,1,type=float]    # Postsynaptic potential (3 levels: inhibitory, none, excitatory)
Ca[3,1,type=float]     # Calcium signal (3 levels: low, medium, high)

# Policy and Control (3 factors with 3×3×3 = 27 total actions)
P[3,1,type=float]      # Plasticity control (3 actions: LTD, no change, LTP)
C_mod[3,1,type=float]  # Channel modulation (3 actions: decrease, maintain, increase conductance)
M_alloc[3,1,type=float] # Metabolic allocation (3 actions: conserve, balance, invest)

# Free Energy terms
F[V_m,type=float]      # Variational Free Energy for belief updating
G[P,type=float]        # Expected Free Energy (per policy)

# Time
t[1,type=int]         # Discrete time step (milliseconds scale)

## Connections
# State evolution connections
D>V_m                    # Prior influences initial membrane potential
V_m>B                    # Membrane potential affects state transitions
W>B                      # Synaptic weights affect transitions
A>B                      # Adaptation affects transitions
H>B                      # Homeostatic set point affects transitions
M>B                      # Metabolic state affects transitions

# Observation connections
V_m>A                     # Membrane potential generates firing rate observations
W>A                       # Synaptic weights influence PSP observations
V_m>A                     # Membrane potential affects calcium signals (via firing)

# Control connections
P>B                       # Plasticity control affects synaptic weight transitions
C_mod>B                   # Channel modulation affects membrane potential dynamics
M_alloc>B                 # Metabolic allocation affects metabolic state and energy-dependent processes

# Free energy connections
C>G                       # Preferences influence expected free energy
E>P                       # Habits influence plasticity policy
G>P                       # Expected free energy influences plasticity policy

# Action selection
P>C_mod                   # Plasticity influences channel modulation
C_mod>M_alloc             # Channel modulation influences metabolic allocation

## InitialParameterization
# A: 12 observations x 405 hidden states. Likelihood mapping from hidden neural states to observations.
# Observations are ordered as: FR1,FR2,FR3,FR4, PSP1,PSP2,PSP3, Ca1,Ca2,Ca3 (repeated for each state combination)
# Hidden states ordered by: V_m1,W1,A1,H1,M1 → V_m1,W1,A1,H1,M2 → ... → V_m5,W4,A3,H3,M3
A={
  # High firing rate (FR4) most likely when membrane potential is at threshold (V_m4) and adaptation is low (A1)
  # Moderate firing (FR3) likely with depolarized membrane (V_m3) and moderate adaptation (A2)
  # Low firing (FR2) with resting potential (V_m2) and high adaptation (A3)
  # Silent (FR1) with hyperpolarized (V_m1) or refractory (V_m5) states

  # PSP observations depend primarily on synaptic weight (W) and membrane potential (V_m)
  # Calcium signals correlate with firing rate and metabolic state

  # Biologically realistic likelihoods (first 100 of 405 combinations shown)
  (0.05, 0.15, 0.25, 0.55, 0.40, 0.40, 0.20, 0.10, 0.35, 0.55, 0.30, 0.45),  # V_m1,W1,A1,H1,M1
  (0.10, 0.20, 0.30, 0.40, 0.35, 0.45, 0.20, 0.15, 0.40, 0.45, 0.25, 0.40),  # V_m1,W1,A1,H1,M2
  (0.15, 0.25, 0.35, 0.25, 0.30, 0.50, 0.20, 0.20, 0.45, 0.35, 0.20, 0.35),  # V_m1,W1,A1,H1,M3
  # ... (continues for all 405 combinations with biologically realistic probabilities)
}

# B: 405 states_next x 405 states_previous x 27 actions. State transitions for each action.
# Actions ordered by: P1,C_mod1,M_alloc1 → P1,C_mod1,M_alloc2 → ... → P3,C_mod3,M_alloc3
# Each 405×405 matrix defines transitions for one action combination
B={
  # Action 1: LTD, decrease conductance, conserve energy (P1,C_mod1,M_alloc1)
  # - Synaptic weights tend to decrease (LTD)
  # - Membrane potential becomes more hyperpolarized (decreased conductance)
  # - Metabolic state depletes faster (energy conservation)
  # - Adaptation and homeostasis adjust accordingly

  # Action 14: No plasticity, maintain conductance, balance energy (P2,C_mod2,M_alloc2)
  # - Synaptic weights relatively stable
  # - Membrane potential dynamics unchanged
  # - Metabolic state maintained at adequate levels

  # Action 27: LTP, increase conductance, invest energy (P3,C_mod3,M_alloc3)
  # - Synaptic weights tend to increase (LTP)
  # - Membrane potential more depolarized (increased conductance)
  # - Metabolic state improves (energy investment)
  # ... (27 transition matrices total)
}

# C: 12 observations. Preference for biologically realistic neural activity patterns.
C={
  # Prefer moderate firing rates (FR3), balanced PSPs (PSP2), moderate calcium (Ca2)
  # Penalize extreme states: very high firing, strong inhibition/excitation, high calcium
  (0.1, 0.2, 0.4, 0.3, 0.15, 0.35, 0.50, 0.25, 0.35, 0.40, 0.25, 0.20)
}

# D: 405 states. Realistic prior over initial neural states.
D={
  # Start near resting membrane potential (V_m2), moderate synaptic weights (W2)
  # Low initial adaptation (A1), target homeostasis (H2), adequate metabolism (M2)
  # Most mass on biologically realistic initial conditions
  (0.05, 0.15, 0.35, 0.35, 0.10,  # V_m distribution
   0.20, 0.40, 0.30, 0.10,        # W distribution
   0.40, 0.40, 0.20,             # A distribution
   0.20, 0.60, 0.20,             # H distribution
   0.15, 0.70, 0.15)             # M distribution
   # ... (repeated for all 405 state combinations with appropriate probabilities)
}

# E: 27 actions. Habit favoring moderate plasticity, balanced channel modulation, efficient metabolism.
E={
  # Slight preference for LTP (learning), moderate conductance, balanced metabolism
  (0.20, 0.30, 0.50, 0.25, 0.50, 0.25, 0.25, 0.50, 0.25,
   0.30, 0.40, 0.30, 0.25, 0.50, 0.25, 0.30, 0.40, 0.30,
   0.35, 0.40, 0.25, 0.30, 0.45, 0.25, 0.35, 0.40, 0.25)
}

## Equations
# Neural dynamics following Active Inference principles:
# - Membrane potential integrates synaptic inputs with conductance modulation: dV_m/dt = -g_L*(V_m - E_L) + I_syn + I_ext
# - Synaptic weights evolve via plasticity rules: dW/dt = η * STDP(V_m_pre, V_m_post, t_spike)
# - Adaptation accumulates with sustained activity: dA/dt = -A/τ_A + f(V_m)
# - Homeostatic set point adjusts to maintain target firing: H_t = H_{t-1} + α * (FR_target - FR_actual)
# - Metabolic state reflects energy balance: dM/dt = -c_activity * FR + r_allocation * M_alloc
#
# State inference: s_t ~ argmin_F [D_KL(q(s_t|o_{1:t}) || p(s_t|o_{1:t-1}, u_t))]
# Policy inference: π_t ~ argmin_G [E[G(π)] = Σ_τ E[Q(τ)] + D_KL(π || π_prior)]
# Action selection: u_t ~ π_t with softmax temperature τ

## Time
Time=t
Dynamic
Discrete
ModelTimeHorizon=Unbounded # Neural model defined for continuous operation; simulations may specify finite duration.
TimeStep=1ms # Millisecond-scale discrete time steps for realistic neural dynamics.

## ActInfOntologyAnnotation
A=LikelihoodMatrix
B=TransitionMatrices
C=LogPreferenceVector
D=PriorOverHiddenStates
E=HabitVector
F=VariationalFreeEnergy
G=ExpectedFreeEnergy
V_m=MembranePotentialState
W=SynapticWeightFactor
A=AdaptationState
H=HomeostaticSetPoint
M=MetabolicState
FR=FiringRateObservation
PSP=PostsynapticPotentialObservation
Ca=CalciumSignalObservation
P=PlasticityControl
C_mod=ChannelModulation
M_alloc=MetabolicAllocation
t=TimeStep

## ModelParameters
num_membrane_states: 5      # V_m[5] - hyperpolarized, resting, depolarized, threshold, refractory
num_synapse_levels: 4       # W[4] - weak, moderate, strong, saturated
num_adaptation_levels: 3    # A[3] - low, medium, high adaptation
num_homeostatic_levels: 3   # H[3] - low, target, high firing rate targets
num_metabolic_levels: 3     # M[3] - depleted, adequate, surplus energy

num_firing_levels: 4        # FR[4] - silent, low, moderate, high firing rates
num_psp_levels: 3           # PSP[3] - inhibitory, none, excitatory potentials
num_calcium_levels: 3       # Ca[3] - low, medium, high calcium concentrations

num_plasticity_actions: 3   # P[3] - LTD, no change, LTP
num_channel_actions: 3      # C_mod[3] - decrease, maintain, increase conductance
num_metabolic_actions: 3    # M_alloc[3] - conserve, balance, invest energy

total_hidden_states: 405    # 5×4×3×3×3 combinations
total_observations: 12      # 4×3×3 combinations
total_actions: 27           # 3×3×3 combinations

## Footer
Active Inference Neural Response Model v1 - GNN Representation.
This model captures essential aspects of neural computation including membrane dynamics, synaptic plasticity, adaptation, homeostasis, and metabolic constraints within the Active Inference framework. The model is designed for studying how neurons minimize free energy while maintaining stable, efficient, and adaptive responses to stimuli.

## Signature
Cryptographic signature goes here

