# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN model:

1. **Model Matrices**:
   - A matrices representing the state space and action spaces
   - B matrices representing the belief matrix and prediction matrix
   - C matrices representing the probability matrices (beliefs) and predictions
   - D matrices representing the decision variables and actions
2. **Precision Parameters**:
   - γ: precision parameters for each modality
   - α: learning rate for each modality

3. **Dimensional Parameters**:
   - State space dimensions for each modality
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor

4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows
   - Update frequencies and timescales
5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies

6. **Configuration Summary**:
   - Parameter file format recommendations for each modality:
    - Decision variables and actions (decision probabilities)
    - Action parameters (action weights, action biases)
    - Prediction parameters (prediction probability matrices)
7. **Tunable vs. Fixed Parameters**:
   - Sensitivity analysis priorities based on parameter values

8. **Parameter File Format Recommendations**:
   - Use tabular format for parameter file formats:
      - Tabulate the parameter values and their corresponding decision probabilities
      - Use tabular formatting to specify parameters in a tabular format
9. **Tunable vs. Fixed Parameters**:
   - Use tabular format for parameter evaluation metrics (e.g., accuracy, precision)
    - Use tabular format for parameter interpretation (e.g., sensitivity analysis priorities)