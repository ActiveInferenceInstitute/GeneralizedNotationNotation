# EXTRACT_PARAMETERS

Based on the document, here are the key parameters for the GNN model:

1. **Model Matrices**:
   - A matrices representing the input data (e.g., `A`) and output data (e.g., `B`, `C`).
   - B matrices represent the input data of each level, while C matrices represent the outputs of each level.
   - C matrices represent the predictions made by each level based on its own hidden state and actions.

2. **Precision Parameters**:
   - γ (gamma): precision parameters for each level
   - α (alpha): learning rate parameters for each level
   - Other precision/confidence parameters, such as sensitivity analysis priorities or other parameter values

3. **Dimensional Parameters**:
   - State space dimensions for each factor
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
   - Parameter file format recommendations for each level of the model

So in summary, the GNN model consists of:
- A set of matrices representing input data and output data (e.g., `A`, `B`)
- A matrix representing prediction accuracy over initial states (`C`)
- A matrix representing predictions made by each level based on its own hidden state and actions (`D`)
- A matrix representing predictions made by each level based on their own hidden state and actions (`E`)
- A set of parameters for initialization strategies, sensitivity analysis, and other parameter values.