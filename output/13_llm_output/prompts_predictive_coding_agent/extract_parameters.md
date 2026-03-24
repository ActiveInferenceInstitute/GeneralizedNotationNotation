# EXTRACT_PARAMETERS

Based on the information provided, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the model structure and interpretation of predictions (beliefs) and actions.
   - B matrices representing the dimensionality of each factor in the model matrix representation.
   - C matrices representing the dimensions of each modality, such as sensory perception or action prediction.
   - D matrices representing the dimensionality of each control variable.

2. **Precision Parameters**:
   - γ (gamma): precision parameters and their roles.
   - α (alpha): learning rates and adaptation parameters for improving predictions accuracy.
   - Other precision/confidence parameters, such as bias-variance tradeoff or sensitivity analysis priorities.
3. **Dimensional Parameters**:
   - State space dimensions: 3x4 matrices representing the model structure and interpretation of predictions.
   - Observation space dimensions: 2x1 matrix representing each modality (sensory perception vs. action prediction).
   - Action space dimensions: 2x1 matrix representing each control variable, with dimensionality determined by the learning rate parameter α.

4. **Temporal Parameters**:
   - Time horizons for initial states and actions
   
   - Temporal dependencies and windows for updating predictions over time
5. **Initial Conditions**:
   - Prior beliefs over initial states (initial state)
   
   - Initial parameters values for each modality (sensory perception vs. action prediction).

6. **Configuration Summary**:
   - Parameter file format recommendations:
   - Tunable vs. fixed parameters, sensitivity analysis priorities and sensitivity analysis prioritization strategies
7. **Sensitivity Analysis Prior Beliefs Over Initial States**:
   - Sensitivity analysis priorities:
   - Sensitivity analysis priorities for initial state biases (bias-variance tradeoff) and sensitivity analysis priorities for initial parameter values.