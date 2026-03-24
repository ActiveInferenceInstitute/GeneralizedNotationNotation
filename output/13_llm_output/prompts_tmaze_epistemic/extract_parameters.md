# EXTRACT_PARAMETERS

Based on the information provided, here are the key parameters that make up the GNN model:

1. **Model Matrices**:
   - A matrices representing the location likelihood and reward likelihoods for each action/observation pair.
   - B matrices representing the location transition probabilities between actions/observations.
   - C matrices representing the location probability matrix of each action/observation pair, with a bias to visit the cue location first (bias=1).
   - D matrices representing the location transition probabilities and reward likelihoods for each action/observation pair.

2. **Precision Parameters**:
   - γ: precision parameters and their roles in GNN models.
   - α: learning rates and adaptation parameters, with a bias to increase accuracy over time (bias=1).
   - Other precision/confidence parameters are not specified but can be inferred from the structure of the model matrices.

3. **Dimensional Parameters**:
   - State space dimensions for each action parameter
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor

4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows
   - Update frequencies and timescales

5. **Initial Conditions**:
   - Initial parameters: prior beliefs over initial states, observation spaces, action sets, etc.