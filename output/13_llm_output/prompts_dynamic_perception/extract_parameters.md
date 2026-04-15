# EXTRACT_PARAMETERS

Based on the provided information, here are the key parameters for the Active Inference Model (AIM):

1. **Model Matrices**:
   - A matrices: dimensions, structure, interpretation
   - B matrices: dimensions, structure, interpretation
   - C matrices: dimensions, structure, interpretation
   - D matrices: dimensions, structure, interpretation
2. **Precision Parameters**:
   - γ (gamma) = 0.95
   - α = 1

3. **Dimensional Parameters**:
   - State space dimensions for each factor
   - Observation space dimensions for each modality
   - Action space dimensions for each control factor

4. **Temporal Parameters**:
   - Time horizons (t): 2

Here are the parameter breakdowns:

1. **Model Matrices**:
   - A matrices: dimensions, structure, interpretation
   - B matrices: dimensions, structure, interpretation
   - C matrices: dimensions, structure, interpretation
   - D matrices: dimensions, structure, interpretation

**Parameter Breakdown:**

   - **A Matrix**:
      - Initialization parameters (initializations):
         - γ = 0.95
       - α = 1

2. **B Matrix**:
    - Initial parameter values:
        - α = 1
   - **C Matrix**:
    - Initial parameter value(s) for A matrix:
        - γ = 0.95
      - α = 1
3. **D Matrix**:
    - Initial parameter values:
        - γ = 0.95
       - α = 1

4. **State Space Dimensions**:
   - Initial parameters (initialization):
     - A=RecognitionMatrix
     - B=TransitionMatrix
     - D=Prior
      - C=InitialParameterizations

**Temporal Parameters:**

   - Time horizons:
    - T = 2
 
**Prediction Parameters:**

   - **Initial Parameter Values**:
        - γ = 0.95
       - α = 1

3. **Configuration Summary**:
   - Initialization strategies (tunable vs fixed parameters):
      - Sensitivity analysis priorities