# Basic GNN Model Template

Use this template as a starting point for creating your own GNN models. Replace the placeholder content with your specific model details.

## GNNVersionAndFlags
GNN v1

## ModelName
[Your Model Name] v[Version]

## ModelAnnotation
[Describe your model's purpose, key features, and application domain. 
 Explain what problem it solves and what makes it unique.
 Keep this section concise but informative.]

## StateSpaceBlock
# Hidden State Factors (s_fX[num_states, 1, type=dataType])
s_f0[X,1,type=int]   # Hidden State Factor 0: [Description] (0:[State0], 1:[State1], ...)

# Observation Modalities (o_mX[num_outcomes, 1, type=dataType])
o_m0[Y,1,type=int]   # Observation Modality 0: [Description] (0:[Outcome0], 1:[Outcome1], ...)

# Control Factors (if applicable)
# pi_cX[num_actions, type=float]  # Policy for Control Factor X
# u_cX[1,type=int]               # Chosen action for Control Factor X

# Model Matrices
A_m0[Y,X,type=float] # Likelihood matrix: P(o_m0 | s_f0)
B_f0[X,X,type=float] # Transition matrix: P(s_f0' | s_f0) [for dynamic models]
C_m0[Y,type=float]   # Preferences over o_m0 outcomes
D_f0[X,type=float]   # Prior beliefs over s_f0 states

# Time (for dynamic models)
t[1,type=int]        # Time step

## Connections
# Basic connections for perception model
(D_f0) -> (s_f0)
(s_f0) -> (A_m0)
(A_m0) -> (o_m0)

# For dynamic models, add:
# (s_f0, u_c0) -> (B_f0)  # If actions influence transitions
# (B_f0) -> s_f0_next

# For preference-based models, add:
# (C_m0, [other relevant variables]) > G  # Expected Free Energy
# G > pi_c0                               # Policy selection

## InitialParameterization
# A_m0: Likelihood matrix [describe the mapping logic]
A_m0={
  # Example for 2x2 case:
  # ((P(o=0|s=0), P(o=0|s=1)), (P(o=1|s=0), P(o=1|s=1)))
  "description": "Replace with actual probability values"
}

# B_f0: Transition matrix [for dynamic models]
B_f0={
  # Example: 
  # ((P(s'=0|s=0), P(s'=0|s=1)), (P(s'=1|s=0), P(s'=1|s=1)))
  "description": "Replace with actual transition probabilities"
}

# C_m0: Preferences [log preferences, higher values = more preferred]
C_m0={
  # Example: (log_pref_outcome_0, log_pref_outcome_1, ...)
  "description": "Replace with actual preference values"
}

# D_f0: Prior beliefs [probability distribution over initial states]
D_f0={
  # Example: (P(s=0), P(s=1), ...)
  "description": "Replace with actual prior probabilities"
}

## Equations
# Standard Active Inference equations:
# For static models:
# q(s) = σ(ln(D) + ln(A^T o))

# For dynamic models:
# q(s_t) = σ(ln(D) + ln(B^dagger s_{t+1}) + ln(A^T o_t))

# For models with preferences:
# G(π) = E_q[ln q(o,s|π) - ln P(o,s|π) - ln C(o)]
# P(π) = σ(-G(π))

## Time
# Choose one:
Static                    # No time dynamics
# OR
Dynamic                   # With time dynamics
DiscreteTime=t           # Time variable
ModelTimeHorizon=10      # Planning/simulation horizon

## ActInfOntologyAnnotation
# Map your variables to Active Inference Ontology terms
s_f0=[OntologyTerm]      # e.g., HiddenState, LocationState, etc.
o_m0=[OntologyTerm]      # e.g., Observation, VisualObservation, etc.
A_m0=[OntologyTerm]      # e.g., LikelihoodMatrix, ObservationModel, etc.
B_f0=[OntologyTerm]      # e.g., TransitionMatrix, DynamicsModel, etc.
C_m0=[OntologyTerm]      # e.g., PreferenceVector, UtilityFunction, etc.
D_f0=[OntologyTerm]      # e.g., PriorDistribution, InitialBelief, etc.
t=[OntologyTerm]         # e.g., TimeStep, TemporalIndex, etc.

## ModelParameters
num_hidden_state_factors: [X]  # Number of state factors
num_observation_modalities: [Y] # Number of observation modalities
num_control_factors: [Z]        # Number of control factors (if applicable)

## Footer
[Your Model Name] v[Version] - End of Specification

## Signature
Creator: [Your Name]
Date: [Date]
Status: [Development stage, e.g., Draft, Tested, Production]
Contact: [Email or other contact information]

---

## Usage Instructions

1. **Replace all bracketed placeholders** `[...]` with your specific content
2. **Define your state spaces** - What states and observations does your model track?
3. **Specify connections** - How do variables influence each other?
4. **Set parameters** - Provide concrete probability values and matrices
5. **Add equations** - Include any custom mathematical relationships
6. **Map to ontology** - Connect your variables to standard Active Inference terms
7. **Validate** - Use GNN type checker to ensure correctness
8. **Test** - Run through the GNN pipeline to verify functionality

## Tips for Success

- **Start simple** - Begin with basic perception, add complexity incrementally
- **Check dimensions** - Ensure matrix dimensions are compatible
- **Normalize probabilities** - All probability distributions must sum to 1
- **Document thoroughly** - Clear annotations help others understand your model
- **Use meaningful names** - Choose variable names that reflect their purpose
- **Reference examples** - Look at similar models in the `doc/archive/` directory 