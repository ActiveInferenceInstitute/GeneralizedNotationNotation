# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
    - A matrices representing the model structure and its components (agents, environment, etc.)
    - B matrices representing the agent-agent interactions and their components
   - C matrices representing the agent-action relationships and their components
2. **Precision Parameters**:
    - γ (gamma): precision parameters for each agent
    - α (alpha): learning rates and adaptation parameters
    - Other precision/confidence parameters

3. **Dimensional Parameters**:
    - State space dimensions for each factor
    - Observation space dimensions for each modality
    - Action space dimensions for each control factor
4. **Temporal Parameters**:
    - Time horizons (T)
   - Temporal dependencies and windows
    - Update frequencies and timescales