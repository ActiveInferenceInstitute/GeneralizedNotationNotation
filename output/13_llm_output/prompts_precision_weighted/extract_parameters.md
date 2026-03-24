# EXTRACT_PARAMETERS

Based on the specifications, here's a systematic parameter breakdown of the GNN model:

1. **Model Matrices**:
   - A matrices represent the state space and action spaces for each modality. The matrix representing sensory precision is represented as A in the model specification. The matrix representing policy precision is represented as B in the model specification. The matrix representing epistemic confidence is represented as C in the model specification, with a specific structure (A) that corresponds to the parameter set of GNN v1.

2. **Precision Parameters**:
   - α represents learning rate and adaptation parameters for each modality. The parameter space is represented by A in the model specification. The parameter space representing epistemic confidence is represented as B in the model specification, with a specific structure (A) that corresponds to the parameter set of GNN v1.

3. **Dimensional Parameters**:
   - State space dimensions: A matrices represent the state space and action spaces for each modality. The matrix representing sensory precision is represented as A in the model specification. The matrix representing policy precision is represented as B in the model specification, with a specific structure (A) that corresponds to the parameter set of GNN v1.

4. **Temporal Parameters**:
   - Time horizons: A matrices represent the time horizon for each modality and its corresponding temporal dependencies. The parameter space represents epistemic confidence over initial states. The parameter space representing sensory precision is represented as B in the model specification, with a specific structure (A) that corresponds to the parameter set of GNN v1.

5. **Initial Conditions**:
   - Initial conditions: A matrices represent the initial state and action spaces for each modality. The parameter space represents epistemic confidence over initial states. The parameter space representing sensory precision is represented as B in the model specification, with a specific structure (A) that corresponds to the parameter set of GNN v1.

6. **Configuration Summary**:
   - Parameter file format recommendations: A structured representation of each parameter and its role within the model. This provides insight into how different parameters interact with one another and can be used for analysis purposes.

This systematic approach allows you to understand the relationships between the parameters, their roles in the model, and their interactions with each other.