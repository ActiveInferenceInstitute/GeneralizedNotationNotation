# GNN File: src/gnn/examples/gnn_example_pymdp_agent.md\n\n## Raw File Content\n\n```\n# GNN Example: Multifactor PyMDP Agent
# Format: Markdown representation of a Multifactor PyMDP model in Active Inference format
# Version: 1.0
# This file is machine-readable and attempts to represent a PyMDP agent with multiple observation modalities and hidden state factors.

## GNNSection
MultifactorPyMDPAgent

## GNNVersionAndFlags
GNN v1

## ModelName
Multifactor PyMDP Agent v1

## ModelAnnotation
This model represents a PyMDP agent with multiple observation modalities and hidden state factors.
- Observation modalities: "state_observation" (3 outcomes), "reward" (3 outcomes), "decision_proprioceptive" (3 outcomes)
- Hidden state factors: "reward_level" (2 states), "decision_state" (3 states)
- Control: "decision_state" factor is controllable with 3 possible actions.
The parameterization is derived from a PyMDP Python script example.

## StateSpaceBlock
# A_matrices are defined per modality: A_m[observation_outcomes, state_factor0_states, state_factor1_states]
A_m0[3,2,3,type=float]   # Likelihood for modality 0 ("state_observation")
A_m1[3,2,3,type=float]   # Likelihood for modality 1 ("reward")
A_m2[3,2,3,type=float]   # Likelihood for modality 2 ("decision_proprioceptive")

# B_matrices are defined per hidden state factor: B_f[states_next, states_previous, actions]
B_f0[2,2,1,type=float]   # Transitions for factor 0 ("reward_level"), 1 implicit action (uncontrolled)
B_f1[3,3,3,type=float]   # Transitions for factor 1 ("decision_state"), 3 actions

# C_vectors are defined per modality: C_m[observation_outcomes]
C_m0[3,type=float]       # Preferences for modality 0
C_m1[3,type=float]       # Preferences for modality 1
C_m2[3,type=float]       # Preferences for modality 2

# D_vectors are defined per hidden state factor: D_f[states]
D_f0[2,type=float]       # Prior for factor 0
D_f1[3,type=float]       # Prior for factor 1

# Hidden States
s_f0[2,1,type=float]     # Hidden state for factor 0 ("reward_level")
s_f1[3,1,type=float]     # Hidden state for factor 1 ("decision_state")
s_prime_f0[2,1,type=float] # Next hidden state for factor 0
s_prime_f1[3,1,type=float] # Next hidden state for factor 1

# Observations
o_m0[3,1,type=float]     # Observation for modality 0
o_m1[3,1,type=float]     # Observation for modality 1
o_m2[3,1,type=float]     # Observation for modality 2

# Policy and Control
π_f1[3,type=float]       # Policy (distribution over actions) for controllable factor 1
u_f1[1,type=int]         # Action taken for controllable factor 1
G[1,type=float]          # Expected Free Energy (overall, or can be per policy)
t[1,type=int]            # Time step

## Connections
(D_f0,D_f1)-(s_f0,s_f1)
(s_f0,s_f1)-(A_m0,A_m1,A_m2)
(A_m0,A_m1,A_m2)-(o_m0,o_m1,o_m2)
(s_f0,s_f1,u_f1)-(B_f0,B_f1) # u_f1 primarily affects B_f1; B_f0 is uncontrolled
(B_f0,B_f1)-(s_prime_f0,s_prime_f1)
(C_m0,C_m1,C_m2)>G
G>π_f1
π_f1-u_f1
G=ExpectedFreeEnergy
t=Time

## InitialParameterization
# A_m0: num_obs[0]=3, num_states[0]=2, num_states[1]=3. Format: A[obs_idx][state_f0_idx][state_f1_idx]
# A[0][:, :, 0] = np.ones((3,2))/3
# A[0][:, :, 1] = np.ones((3,2))/3
# A[0][:, :, 2] = [[0.8,0.2],[0.0,0.0],[0.2,0.8]] (obs x state_f0 for state_f1=2)
A_m0={
  ( (0.33333,0.33333,0.8), (0.33333,0.33333,0.2) ),  # obs=0; (vals for s_f1 over s_f0=0), (vals for s_f1 over s_f0=1)
  ( (0.33333,0.33333,0.0), (0.33333,0.33333,0.0) ),  # obs=1
  ( (0.33333,0.33333,0.2), (0.33333,0.33333,0.8) )   # obs=2
}

# A_m1: num_obs[1]=3, num_states[0]=2, num_states[1]=3
# A[1][2, :, 0] = [1.0,1.0]
# A[1][0:2, :, 1] = softmax([[1,0],[0,1]]) approx [[0.731,0.269],[0.269,0.731]]
# A[1][2, :, 2] = [1.0,1.0]
# Others are 0.
A_m1={
  ( (0.0,0.731,0.0), (0.0,0.269,0.0) ),  # obs=0
  ( (0.0,0.269,0.0), (0.0,0.731,0.0) ),  # obs=1
  ( (1.0,0.0,1.0), (1.0,0.0,1.0) )      # obs=2
}

# A_m2: num_obs[2]=3, num_states[0]=2, num_states[1]=3
# A[2][0,:,0]=1.0; A[2][1,:,1]=1.0; A[2][2,:,2]=1.0
# Others are 0.
A_m2={
  ( (1.0,0.0,0.0), (1.0,0.0,0.0) ),  # obs=0
  ( (0.0,1.0,0.0), (0.0,1.0,0.0) ),  # obs=1
  ( (0.0,0.0,1.0), (0.0,0.0,1.0) )   # obs=2
}

# B_f0: factor 0 (2 states), uncontrolled (1 action). Format B[s_next, s_prev, action=0]
# B_f0 = eye(2)
B_f0={
  ( (1.0),(0.0) ), # s_next=0; (vals for s_prev over action=0)
  ( (0.0),(1.0) )  # s_next=1
}

# B_f1: factor 1 (3 states), 3 actions. Format B[s_next, s_prev, action_idx]
# B_f1[:,:,action_idx] = eye(3) for each action
B_f1={
  ( (1.0,1.0,1.0), (0.0,0.0,0.0), (0.0,0.0,0.0) ), # s_next=0; (vals for actions over s_prev=0), (vals for actions over s_prev=1), ...
  ( (0.0,0.0,0.0), (1.0,1.0,1.0), (0.0,0.0,0.0) ), # s_next=1
  ( (0.0,0.0,0.0), (0.0,0.0,0.0), (1.0,1.0,1.0) )  # s_next=2
}

# C_m0: num_obs[0]=3. Defaults to zeros.
C_m0={(0.0,0.0,0.0)}

# C_m1: num_obs[1]=3. C[1][0]=1.0, C[1][1]=-2.0
C_m1={(1.0,-2.0,0.0)}

# C_m2: num_obs[2]=3. Defaults to zeros.
C_m2={(0.0,0.0,0.0)}

# D_f0: factor 0 (2 states). Uniform prior.
D_f0={(0.5,0.5)}

# D_f1: factor 1 (3 states). Uniform prior.
D_f1={(0.33333,0.33333,0.33333)}

## Equations
# Standard PyMDP agent equations for state inference (infer_states),
# policy inference (infer_policies), and action sampling (sample_action).
# qs = infer_states(o)
# q_pi, efe = infer_policies()
# action = sample_action()

## Time
Dynamic
DiscreteTime=t
ModelTimeHorizon=Unbounded # Agent definition is generally unbounded, specific simulation runs have a horizon.

## ActInfOntologyAnnotation
A_m0=LikelihoodMatrixModality0
A_m1=LikelihoodMatrixModality1
A_m2=LikelihoodMatrixModality2
B_f0=TransitionMatrixFactor0
B_f1=TransitionMatrixFactor1
C_m0=LogPreferenceVectorModality0
C_m1=LogPreferenceVectorModality1
C_m2=LogPreferenceVectorModality2
D_f0=PriorOverHiddenStatesFactor0
D_f1=PriorOverHiddenStatesFactor1
s_f0=HiddenStateFactor0
s_f1=HiddenStateFactor1
s_prime_f0=NextHiddenStateFactor0
s_prime_f1=NextHiddenStateFactor1
o_m0=ObservationModality0
o_m1=ObservationModality1
o_m2=ObservationModality2
π_f1=PolicyVectorFactor1 # Distribution over actions for factor 1
u_f1=ActionFactor1       # Chosen action for factor 1
G=ExpectedFreeEnergy

## ModelParameters
num_hidden_states_factors: [2, 3]  # s_f0[2], s_f1[3]
num_obs_modalities: [3, 3, 3]     # o_m0[3], o_m1[3], o_m2[3]
num_control_factors: [1, 3]   # B_f0 actions_dim=1 (uncontrolled), B_f1 actions_dim=3 (controlled by pi_f1)

## Footer
Multifactor PyMDP Agent v1 - GNN Representation

## Signature
NA \n```\n\n## Parsed Sections

### _HeaderComments

```
# GNN Example: Multifactor PyMDP Agent
# Format: Markdown representation of a Multifactor PyMDP model in Active Inference format
# Version: 1.0
# This file is machine-readable and attempts to represent a PyMDP agent with multiple observation modalities and hidden state factors.
```

### ModelName

```
Multifactor PyMDP Agent v1
```

### GNNSection

```
MultifactorPyMDPAgent
```

### GNNVersionAndFlags

```
GNN v1
```

### ModelAnnotation

```
This model represents a PyMDP agent with multiple observation modalities and hidden state factors.
- Observation modalities: "state_observation" (3 outcomes), "reward" (3 outcomes), "decision_proprioceptive" (3 outcomes)
- Hidden state factors: "reward_level" (2 states), "decision_state" (3 states)
- Control: "decision_state" factor is controllable with 3 possible actions.
The parameterization is derived from a PyMDP Python script example.
```

### StateSpaceBlock

```
# A_matrices are defined per modality: A_m[observation_outcomes, state_factor0_states, state_factor1_states]
A_m0[3,2,3,type=float]   # Likelihood for modality 0 ("state_observation")
A_m1[3,2,3,type=float]   # Likelihood for modality 1 ("reward")
A_m2[3,2,3,type=float]   # Likelihood for modality 2 ("decision_proprioceptive")

# B_matrices are defined per hidden state factor: B_f[states_next, states_previous, actions]
B_f0[2,2,1,type=float]   # Transitions for factor 0 ("reward_level"), 1 implicit action (uncontrolled)
B_f1[3,3,3,type=float]   # Transitions for factor 1 ("decision_state"), 3 actions

# C_vectors are defined per modality: C_m[observation_outcomes]
C_m0[3,type=float]       # Preferences for modality 0
C_m1[3,type=float]       # Preferences for modality 1
C_m2[3,type=float]       # Preferences for modality 2

# D_vectors are defined per hidden state factor: D_f[states]
D_f0[2,type=float]       # Prior for factor 0
D_f1[3,type=float]       # Prior for factor 1

# Hidden States
s_f0[2,1,type=float]     # Hidden state for factor 0 ("reward_level")
s_f1[3,1,type=float]     # Hidden state for factor 1 ("decision_state")
s_prime_f0[2,1,type=float] # Next hidden state for factor 0
s_prime_f1[3,1,type=float] # Next hidden state for factor 1

# Observations
o_m0[3,1,type=float]     # Observation for modality 0
o_m1[3,1,type=float]     # Observation for modality 1
o_m2[3,1,type=float]     # Observation for modality 2

# Policy and Control
π_f1[3,type=float]       # Policy (distribution over actions) for controllable factor 1
u_f1[1,type=int]         # Action taken for controllable factor 1
G[1,type=float]          # Expected Free Energy (overall, or can be per policy)
t[1,type=int]            # Time step
```

### Connections

```
(D_f0,D_f1)-(s_f0,s_f1)
(s_f0,s_f1)-(A_m0,A_m1,A_m2)
(A_m0,A_m1,A_m2)-(o_m0,o_m1,o_m2)
(s_f0,s_f1,u_f1)-(B_f0,B_f1) # u_f1 primarily affects B_f1; B_f0 is uncontrolled
(B_f0,B_f1)-(s_prime_f0,s_prime_f1)
(C_m0,C_m1,C_m2)>G
G>π_f1
π_f1-u_f1
G=ExpectedFreeEnergy
t=Time
```

### InitialParameterization

```
# A_m0: num_obs[0]=3, num_states[0]=2, num_states[1]=3. Format: A[obs_idx][state_f0_idx][state_f1_idx]
# A[0][:, :, 0] = np.ones((3,2))/3
# A[0][:, :, 1] = np.ones((3,2))/3
# A[0][:, :, 2] = [[0.8,0.2],[0.0,0.0],[0.2,0.8]] (obs x state_f0 for state_f1=2)
A_m0={
  ( (0.33333,0.33333,0.8), (0.33333,0.33333,0.2) ),  # obs=0; (vals for s_f1 over s_f0=0), (vals for s_f1 over s_f0=1)
  ( (0.33333,0.33333,0.0), (0.33333,0.33333,0.0) ),  # obs=1
  ( (0.33333,0.33333,0.2), (0.33333,0.33333,0.8) )   # obs=2
}

# A_m1: num_obs[1]=3, num_states[0]=2, num_states[1]=3
# A[1][2, :, 0] = [1.0,1.0]
# A[1][0:2, :, 1] = softmax([[1,0],[0,1]]) approx [[0.731,0.269],[0.269,0.731]]
# A[1][2, :, 2] = [1.0,1.0]
# Others are 0.
A_m1={
  ( (0.0,0.731,0.0), (0.0,0.269,0.0) ),  # obs=0
  ( (0.0,0.269,0.0), (0.0,0.731,0.0) ),  # obs=1
  ( (1.0,0.0,1.0), (1.0,0.0,1.0) )      # obs=2
}

# A_m2: num_obs[2]=3, num_states[0]=2, num_states[1]=3
# A[2][0,:,0]=1.0; A[2][1,:,1]=1.0; A[2][2,:,2]=1.0
# Others are 0.
A_m2={
  ( (1.0,0.0,0.0), (1.0,0.0,0.0) ),  # obs=0
  ( (0.0,1.0,0.0), (0.0,1.0,0.0) ),  # obs=1
  ( (0.0,0.0,1.0), (0.0,0.0,1.0) )   # obs=2
}

# B_f0: factor 0 (2 states), uncontrolled (1 action). Format B[s_next, s_prev, action=0]
# B_f0 = eye(2)
B_f0={
  ( (1.0),(0.0) ), # s_next=0; (vals for s_prev over action=0)
  ( (0.0),(1.0) )  # s_next=1
}

# B_f1: factor 1 (3 states), 3 actions. Format B[s_next, s_prev, action_idx]
# B_f1[:,:,action_idx] = eye(3) for each action
B_f1={
  ( (1.0,1.0,1.0), (0.0,0.0,0.0), (0.0,0.0,0.0) ), # s_next=0; (vals for actions over s_prev=0), (vals for actions over s_prev=1), ...
  ( (0.0,0.0,0.0), (1.0,1.0,1.0), (0.0,0.0,0.0) ), # s_next=1
  ( (0.0,0.0,0.0), (0.0,0.0,0.0), (1.0,1.0,1.0) )  # s_next=2
}

# C_m0: num_obs[0]=3. Defaults to zeros.
C_m0={(0.0,0.0,0.0)}

# C_m1: num_obs[1]=3. C[1][0]=1.0, C[1][1]=-2.0
C_m1={(1.0,-2.0,0.0)}

# C_m2: num_obs[2]=3. Defaults to zeros.
C_m2={(0.0,0.0,0.0)}

# D_f0: factor 0 (2 states). Uniform prior.
D_f0={(0.5,0.5)}

# D_f1: factor 1 (3 states). Uniform prior.
D_f1={(0.33333,0.33333,0.33333)}
```

### Equations

```
# Standard PyMDP agent equations for state inference (infer_states),
# policy inference (infer_policies), and action sampling (sample_action).
# qs = infer_states(o)
# q_pi, efe = infer_policies()
# action = sample_action()
```

### Time

```
Dynamic
DiscreteTime=t
ModelTimeHorizon=Unbounded # Agent definition is generally unbounded, specific simulation runs have a horizon.
```

### ActInfOntologyAnnotation

```
A_m0=LikelihoodMatrixModality0
A_m1=LikelihoodMatrixModality1
A_m2=LikelihoodMatrixModality2
B_f0=TransitionMatrixFactor0
B_f1=TransitionMatrixFactor1
C_m0=LogPreferenceVectorModality0
C_m1=LogPreferenceVectorModality1
C_m2=LogPreferenceVectorModality2
D_f0=PriorOverHiddenStatesFactor0
D_f1=PriorOverHiddenStatesFactor1
s_f0=HiddenStateFactor0
s_f1=HiddenStateFactor1
s_prime_f0=NextHiddenStateFactor0
s_prime_f1=NextHiddenStateFactor1
o_m0=ObservationModality0
o_m1=ObservationModality1
o_m2=ObservationModality2
π_f1=PolicyVectorFactor1 # Distribution over actions for factor 1
u_f1=ActionFactor1       # Chosen action for factor 1
G=ExpectedFreeEnergy
```

### ModelParameters

```
num_hidden_states_factors: [2, 3]  # s_f0[2], s_f1[3]
num_obs_modalities: [3, 3, 3]     # o_m0[3], o_m1[3], o_m2[3]
num_control_factors: [1, 3]   # B_f0 actions_dim=1 (uncontrolled), B_f1 actions_dim=3 (controlled by pi_f1)
```

### Footer

```
Multifactor PyMDP Agent v1 - GNN Representation
```

### Signature

```
NA
```

