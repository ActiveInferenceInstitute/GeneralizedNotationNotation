# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices: dimensions, structure, interpretation
   - B matrices: dimensions, structure, interpretation
   - C matrices: dimensions, structure, interpretation

   These matrices represent the model representations of the agents and their actions. The initial state is represented as a matrix with 3x2 cells (agent positions) and 4x1 cells (action probabilities). The initial parameter values are represented in the form of matrices representing the initial states and action parameters for each agent.

2. **Precision Parameters**:
   - γ: precision parameters, role interpretation
   - α: learning rates and adaptation parameters
   - Other precision/confidence parameters

   These parameters represent the predictions made by the agents based on their actions over time. The parameter values are represented as matrices representing the initial state and action parameters for each agent.

3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor

   These dimensionality represent the predictions made by the agents based on their actions over time. The parameter values are represented as matrices representing the initial state and action parameters for each agent.

4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows
   - Update frequencies and timescales

   These parameters represent the update frequency of the agents based on their actions over time. The parameter values are represented as matrices representing the initial state and action parameters for each agent.

5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies

6. **Configuration Summary**:
   - Parameter file format recommendations

   These provide a systematic description of the model representations, predictions made by agents based on their actions, and configuration summaries of the parameters for each agent.