# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN model:

- **Model Matrices**:
  - A matrices representing the state space and observation dimensions of the agent (represented as `A`)
  - B matrices representing the action spaces and hidden states dimensions (represented as `B`)
  - C matrices representing the policy, goal, and habit distributions over actions (`C`)
  - D matrices representing the decision-making parameters (represented as `D`)

**Model Parameters**:
  - `γ`: precision parameter for each factor
  - `α`: learning rate for each modality
  - `Other Precision/Confidence Parameters`
    - `other_precision`, other_confidence, and `sensitivity_to_bias` are used to evaluate the agent's performance on different types of data (e.g., action-based vs. state-based)

**Initial Conditions**:
  - Initial parameters for each factor:
    - `initial_state`: initial state space dimensions (`A`)
    - `initial_observation`: initial observation dimension (`B`), and
    - `action_space`: initial action space dimension (`C`)
    - `actions`: initial actions (represented as `D`, `F`, etc.)
  - Initialization strategies:
    - **Random initialization**: random parameters for each factor
    - **Fixed initialization**: fixed parameter values for each factor

**Configuration Summary**:
  - **Initial Parameters**:
    - `initial_state` and `initial_observation`: initial state space dimensions (`A`) and
      initial observation dimension (`B`, etc.)
    - **Initialization strategy**: random parameters for each factor (e.g., random choice of action)
    - **Random initialization**: random parameter values for each factor

**Tunable Parameters**:
  - `gamma` is a hyperparameter that controls the precision and confidence of each
      parameter in the model, with higher values indicating more precise
      predictions.
  - `α`: learning rate for each modality (e.g., action-based vs. state-based)
    - `other_precision`, other_confidence**: additional precision/confidence parameters
    - `sensitivity_to_bias` is a hyperparameter that controls the sensitivity of
      the agent to bias in its decision, with higher values indicating more
      sensitive predictions
  - **Other Precision/Confidence Parameters** are used to evaluate the agent's performance on different types of data (e.g., action-based vs. state-based