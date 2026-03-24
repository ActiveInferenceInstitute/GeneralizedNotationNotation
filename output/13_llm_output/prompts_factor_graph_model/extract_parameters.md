# EXTRACT_PARAMETERS

Based on the document, here are the key components of the model:

1. **Model Matrices**:
   - A matrices representing the matrix representation of the model (A) and its structure (B).
   - B represents the vectorized information about the variables in the model.
   - C represents the vectorized information about the variable nodes, which are used to compute the probabilities for each action/control factor pair.

2. **Precision Parameters**:
   - γ: precision parameters and their roles.
   - α: learning rates and adaptation parameters.
   - Other precision/confidence parameters (e.g., δ)

3. **Dimensional Parameters**:
   - State space dimensions for each modality
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor

4. **Temporal Parameters**:
   - Time horizons (t): time horizon and its interpretation priorities.
   - Temporal dependencies and windows: temporal dependencies, window sizes, and their interpretations.
   - Update frequencies and timescales: update frequency and timescale are used to adjust the parameters based on the current state space dimensions of each modality.

5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies (e.g., using a specific number of timesteps)

6. **Configuration Summary**:
   - Parameter file format recommendations: describe how to generate and interpret the model parameters from the input data files.