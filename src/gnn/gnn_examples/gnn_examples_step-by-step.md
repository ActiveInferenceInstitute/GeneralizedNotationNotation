# GNN Examples: Step-by-Step Model Development

This document provides step-by-step examples of developing GNN models, following the progression from simple static models to complex dynamic active inference models as described in Smith et al. (2022).

## Model 1: Static Perception Model

### Description
The simplest GNN model demonstrates basic perception without temporal dynamics. It relates hidden states directly to observations through a recognition matrix.

### GNN Specification

```gnn
## GNN v1

# Static Perception Model

## Model annotations
This model demonstrates basic perceptual inference with a single hidden state
and observation modality. It serves as the foundation for understanding GNN
syntax and structure.

## State space block
D[2,1,type=float]     # Prior over hidden states
s[2,1,type=float]     # Hidden state
A[2,2,type=float]     # Recognition matrix (likelihood)
o[2,1,type=float]     # Observation

## Connections
D-s                  # Prior constrains hidden state
s-A                  # Hidden state determines recognition
A-o                  # Recognition matrix generates observations

## Initial Parameterization
D={0.5, 0.5}         # Uniform prior
A={{0.9, 0.1},       # High probability of correct perception
   {0.2, 0.8}}       # Some observation noise
o={1.0, 0.0}         # Clear observation of first state

## Equations
s = softmax(ln(D) + ln(A^T * o))

## Time
Static

## ActInf Ontology Annotation
A=RecognitionMatrix
D=Prior
s=HiddenState
o=Observation

# Static Perception Model
```

### Key Concepts
- **Static models** don't have temporal dynamics
- **Recognition matrix (A)** maps hidden states to observations
- **Prior (D)** represents initial beliefs about states
- **Softmax normalization** ensures probabilities sum to 1

## Model 2: Dynamic Perception Model

### Description
Extends the static model by adding temporal dynamics and state transitions.

### GNN Specification

```gnn
## GNN v1

# Dynamic Perception Model

## Model annotations
This model adds temporal dynamics to the static perception model,
allowing the hidden state to evolve over time through a transition matrix.

## State space block
D[2,1,type=float]     # Initial prior
B[2,2,type=float]     # State transition matrix
s_t[2,1,type=float]   # Hidden state at time t
A[2,2,type=float]     # Recognition matrix
o_t[2,1,type=float]   # Observation at time t
t[1,type=int]         # Time step

## Connections
D-s_t                # Initial prior constrains first state
s_t-B                # Current state influences transitions
B-s_t+1              # Transition matrix generates next state
s_t-A                # State determines recognition
A-o_t                # Recognition generates observations

## Initial Parameterization
D={0.6, 0.4}         # Initial belief slightly favors first state
B={{0.7, 0.3},       # State persistence with some transitions
   {0.2, 0.8}}
A={{0.9, 0.1},       # Same recognition as static model
   {0.1, 0.9}}

## Equations
# For first time step:
s_1 = softmax(ln(D) + ln(A^T * o_1))

# For subsequent time steps:
s_t = softmax(ln(B^T * s_{t-1}) + ln(A^T * o_t))

## Time
Dynamic
DiscreteTime=s_t
ModelTimeHorizon=10

## ActInf Ontology Annotation
A=RecognitionMatrix
B=TransitionMatrix
D=Prior
s=HiddenState
o=Observation
t=Time

# Dynamic Perception Model
```

### Key Concepts
- **Dynamic models** include temporal evolution
- **Transition matrix (B)** defines state dynamics
- **Time indexing** with subscript notation (s_t)
- **Recursive inference** using previous state estimates

## Model 3: Dynamic Perception with Policy Selection

### Description
Adds active inference capabilities by introducing policy selection and expected free energy minimization.

### GNN Specification

```gnn
## GNN v1

# Dynamic Perception with Policy Selection

## Model annotations
This model introduces active inference by adding policy selection mechanisms.
The agent can now choose actions to minimize expected free energy and achieve goals.

## State space block
# Generative model
A[2,2,type=float]       # Recognition matrix
B[2,2,2,type=float]     # Transition matrix (state × state × action)
D[2,1,type=float]       # Initial prior
C[2,1,type=float]       # Preference vector (goal states)

# Inference variables
s_t[2,1,type=float]     # Hidden state at time t
o_t[2,1,type=float]     # Observation at time t

# Action and policy
π[2,type=float]         # Policy (distribution over actions)
G[2,type=float]         # Expected free energy for each policy
u[1,type=int]           # Selected action

t[1,type=int]           # Time step

## Connections
D-s_t                  # Prior constrains initial state
s_t-A                  # State determines recognition
A-o_t                  # Recognition generates observations
s_t-B                  # State influences transitions
B-s_t+1                # Transitions generate next state
C-G                    # Preferences determine expected free energy
G>π                    # Expected free energy determines policy
π-u                    # Policy determines action selection

## Initial Parameterization
D={0.5, 0.5}           # Uniform initial prior
A={{0.9, 0.1},         # High fidelity perception
   {0.1, 0.9}}
C={1.0, 0.0}           # Strong preference for first state

# B matrix (simplified - would be 3D in full implementation)
# B[:,:,0] = {{0.8, 0.2}, {0.3, 0.7}}  # Action 0 transitions
# B[:,:,1] = {{0.6, 0.4}, {0.1, 0.9}}  # Action 1 transitions

## Equations
# State inference
s_t = softmax(ln(D) + ln(B^T * s_{t-1}) + ln(A^T * o_t))

# Expected free energy for each policy π
G_π = E_{s_t,o_t|s_{t-1},π} [ln s_t - ln C + ln A^T * o_t]

# Policy selection (softmax over negative expected free energy)
π = softmax(-G)

# Action selection
u ~ Categorical(π)

## Time
Dynamic
DiscreteTime=s_t
ModelTimeHorizon=5

## ActInf Ontology Annotation
A=RecognitionMatrix
B=TransitionMatrix
C=Preference
D=Prior
G=ExpectedFreeEnergy
π=PolicyVector
s=HiddenState
o=Observation
t=Time
u=Action

# Dynamic Perception with Policy Selection
```

### Key Concepts
- **Policy selection (π)** for choosing actions
- **Expected free energy (G)** for evaluating policies
- **Preferences (C)** defining desired states
- **Active inference** through action selection

## Model 4: Dynamic Perception with Flexible Policy Selection

### Description
The most advanced model adds uncertainty over policies and adaptive behavior through precision parameters.

### GNN Specification

```gnn
## GNN v1

# Dynamic Perception with Flexible Policy Selection

## Model annotations
This advanced model includes meta-control over policy selection precision,
allowing the agent to adapt its exploration-exploitation trade-off.

## State space block
# Generative model
A[2,2,type=float]       # Recognition matrix
B[2,2,2,type=float]     # Transition matrix
D[2,1,type=float]       # Initial prior
C[2,1,type=float]       # Preference vector

# Inference variables
s_t[2,1,type=float]     # Hidden state at time t
o_t[2,1,type=float]     # Observation at time t

# Policy and meta-control
π[2,type=float]         # Policy distribution
G[2,type=float]         # Expected free energy
γ[1,type=float]         # Inverse temperature (precision)
β[1,type=float]         # Beta parameter (precision prior)

u[1,type=int]           # Selected action
t[1,type=int]           # Time step

## Connections
D-s_t                  # Prior constrains initial state
s_t-A                  # State determines recognition
A-o_t                  # Recognition generates observations
s_t-B                  # State influences transitions
B-s_t+1                # Transitions generate next state
C-G                    # Preferences determine expected free energy
G-γ                    # Expected free energy influences precision
γ-π                    # Precision determines policy certainty
π-u                    # Policy determines action

## Initial Parameterization
D={0.5, 0.5}           # Uniform prior
A={{0.9, 0.1},         # High fidelity perception
   {0.1, 0.9}}
C={0.8, 0.2}           # Moderate preference for first state
γ={2.0}                # Initial precision (deterministic)
β={1.0}                # Precision prior

## Equations
# State inference (same as before)
s_t = softmax(ln(D) + ln(B^T * s_{t-1}) + ln(A^T * o_t))

# Expected free energy
G_π = E_{s_t,o_t|s_{t-1},π} [ln s_t - ln C + ln A^T * o_t]

# Policy selection with precision
π = softmax(-γ * G)

# Precision update (adaptive behavior)
γ = γ + β * (π - E[π]) * (-G)

## Time
Dynamic
DiscreteTime=s_t
ModelTimeHorizon=10

## ActInf Ontology Annotation
A=RecognitionMatrix
B=TransitionMatrix
C=Preference
D=Prior
G=ExpectedFreeEnergy
γ=Precision
β=BetaParameter
π=PolicyVector
s=HiddenState
o=Observation
t=Time
u=Action

# Dynamic Perception with Flexible Policy Selection
```

### Key Concepts
- **Precision control (γ)** for adapting policy certainty
- **Adaptive behavior** through precision updates
- **Meta-control** over exploration-exploitation balance
- **Advanced expected free energy** calculations

## Implementation Notes

### Common Implementation Pattern

All these models follow the same basic implementation pattern:

1. **Parse the GNN file** to extract variables, connections, and parameters
2. **Initialize variables** with specified dimensions and types
3. **Set up matrices** (A, B, D, C) according to parameterization
4. **Implement inference equations** using the specified mathematical relationships
5. **Run simulation** over the specified time horizon

### Python Implementation Template

```python
import numpy as np
from scipy.special import softmax

class GNNModel:
    def __init__(self, gnn_file_path):
        self.variables = {}
        self.matrices = {}
        self.load_gnn_model(gnn_file_path)

    def load_gnn_model(self, file_path):
        # Parse GNN file and initialize variables
        # Implementation depends on specific GNN parser
        pass

    def infer_states(self, observation):
        # Implement the state inference equation
        # s = softmax(ln(D) + ln(A^T * o)) for static
        # s = softmax(ln(B^T * s_prev) + ln(A^T * o)) for dynamic
        pass

    def select_action(self):
        # For active inference models:
        # Compute expected free energy G
        # Select policy π = softmax(-G)
        # Sample action from π
        pass

    def simulate(self, time_steps):
        # Run simulation loop
        results = {'states': [], 'actions': [], 'observations': []}

        for t in range(time_steps):
            if t == 0:
                # Initial inference
                state = self.infer_states(self.observations[t])
            else:
                # Temporal inference
                state = self.infer_states(self.observations[t])

            results['states'].append(state)

            if hasattr(self, 'policies'):
                action = self.select_action()
                results['actions'].append(action)

        return results
```

## Learning Progression

The models demonstrate a natural learning progression:

1. **Model 1** teaches basic GNN syntax and static inference
2. **Model 2** introduces temporal dynamics and recursive inference
3. **Model 3** adds active inference and policy selection
4. **Model 4** introduces meta-control and adaptive behavior

Each step builds on the previous one, adding complexity while maintaining conceptual coherence.

## References

1. Smith, R., Friston, K.J., & Whyte, C.J. (2022). A step-by-step tutorial on active inference and its application to empirical data. Journal of Mathematical Psychology, 107, 102632.
2. Friston, K. J. (2010). The free-energy principle: a unified brain theory? Nature Reviews Neuroscience, 11(2), 127-138.

