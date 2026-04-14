# EXTRACT_PARAMETERS

Based on the document, here are the key parameters:

1. **Model Matrices**:
   - A matrices representing the model structure and interpretation of each level (A0-B0)
   - B matrices representing the model structures and interpretation of each level (C0-D0)
   - C matrices representing the model relationships between levels, transitions, and actions (C1-D2)

2. **Precision Parameters**:
   - γ: precision parameters for each level
   - α: learning rates and adaptation parameters
   - Other precision/confidence parameters

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
   - Parameter file format recommendations for each level

So in summary, the parameters are:
   - A matrices representing model structure and interpretation of each level (A0-B0)
   - B matrices representing model structures and interpretation of each level (C0-D0)
   - C matrices representing model relationships between levels, transitions, and actions (C1-D2)
   - D matrices representing model relationships between control factors and actions (D1-D2)

These parameters are organized into a structured format to facilitate analysis.