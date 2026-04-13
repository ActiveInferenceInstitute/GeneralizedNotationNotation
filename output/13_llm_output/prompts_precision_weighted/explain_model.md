# EXPLAIN_MODEL

Here is a concise overview of the GNN Section:

**GNNSection:**
ActInfPOMDP

```python
# Define the base POMDP agent with explicit sensory and policy parameters
A=LikelihoodMatrix(
    A, # Likelihood matrix (modulated by ω)
    B = TransitionMatrix # Transition matrix
    C = LogPreferenceVector # Prior over hidden states
    D = PriorOverHiddenStates # Prior over actions
)
```

**GNNVersionAndFlags:**
GNN v1

## ModelName: Precision-Weighted Active Inference Agent

**ModelAnnotation:**
An active inference agent with explicit sensory and policy parameters.

**StateSpaceBlock:**
A = (0.9, 0.05, 0.05) # Likelihood matrix (modulated by ω)
B = ( )   # Transition matrix
C = ( )     # Log-preferences over observations
D = ( )    # Prior over hidden states
E = ( )      # Habit (prior over actions)
s=HiddenState  # Hidden state distribution
s_prime=NextHiddenState # Current observation
o=Observation   # Observation
F=Habit          # Action (action-based policy)
G=ExpectedFreeEnergy    # Expected Free Energy
# GNN Example: Precision-Weighted Active Inference Agent
```