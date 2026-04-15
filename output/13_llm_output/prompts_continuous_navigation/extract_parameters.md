# EXTRACT_PARAMETERS

Based on the provided specifications, here are the key parameters for Active Inference (AI) and GNN:

**Parameter Breakdown:**

1. **Model Matrices**:
   - A matrices representing the model's state space structure, including the number of states, actions, and observations.
   - B matrices representing the belief matrix representation of each action-action pair.
   - C matrices representing the conditional probability distributions over the available actions (actions).
   - D matrices representing the conditional probabilities over the available actions.

2. **Precision Parameters**:
   - γ: precision parameters for each modality, including the number of predictions and their corresponding confidence levels.
   - α: learning rate parameter for each action-action pair.
   - Other precision/confidence parameters (optional)

**Parameter Breakdown:**

1. **Model Matrices**:
   - A matrices representing the model's state space structure, including the number of states, actions, and observations.
   - B matrices representing the belief matrix representation of each action-action pair.
   - C matrices representing the conditional probability distributions over the available actions (actions).
   - D matrices representing the conditional probabilities over the available actions.

2. **Precision Parameters**:
   - γ: precision parameters for each modality, including the number of predictions and their corresponding confidence levels.
   - α: learning rate parameter for each action-action pair.
   - Other precision/confidence parameters (optional)

**Configuration Summary:**

   - **Initial Conditions**:
    - Initial state space dimensions for each modality
    - Initial observation space dimensions for each modality

3. **Tunable Parameters**:
    - **Parameter File Format Recommendations**:
      - **Transformation of parameter file format recommendations**:
        - **"transformation_file_format_recommendations":
            - "model_matrices/transformation_file_format_recommendation".json`: Transformed model matrices and parameters.
          - **"transformation_file_format_recommendation":
                - "model_matrices/transformation_file_format_recommendation".json**: Transformed parameter file format recommendations for each modality.
    - **Tunable Parameters**:
      - **Parameter File Format Recommendations**:
        - **Transformation of parameter file format recommendations**:
            - **"transformation_file_format_recommendations":
                - "model_matrices/transformation_file_format_recommendation".json`: Transformed model matrices and parameters.
         