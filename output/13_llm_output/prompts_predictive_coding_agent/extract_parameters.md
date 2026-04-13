# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the model structure and its relationships with other models (e.g., BeliefNet)
   - B matrices representing the model's predictions and biases
   - C matrices representing the model's actions and expectations
   - D matrices representing the model's predictions and uncertainties
2. **Precision Parameters**:
   - γ = minimizing sensory prediction errors, learning rate estimation, and adaptation parameters
   - α = minimizing expected sensory prediction errors (e.g., sensitivity analysis)

3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor
4. **Temporal Parameters**:
   - Time horizons (t)
   - Temporal dependencies and windows
   - Update frequencies and timescales
5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies

6. **Configuration Summary**:
   - Parameter file format recommendations
   - Tunable vs. fixed parameters
   - Sensitivity analysis priorities