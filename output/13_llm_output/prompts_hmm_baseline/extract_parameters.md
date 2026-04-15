# EXTRACT_PARAMETERS

Based on the provided documentation, here are the key parameters for the GNN implementation:

1. **Model Matrices**:
   - A matrices representing the model matrix and its structure (e.g., 4x4)
   - B matrices representing the transition matrix and its structure (e.g., 6x6)
   - C matrices representing the action matrix and its structure (e.g., 10x10)

2. **Precision Parameters**:
   - γ: precision parameters for each factor
   - α: learning rates and adaptation parameters
   - Other precision/confidence parameters

3. **Dimensional Parameters**:
   - State space dimensions for each modality
   - Observation space dimensions for each control variable
   - Action space dimensions for each control variable

4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows
   - Update frequencies and timescales

5. **Initial Conditions**:
   - Prior beliefs over initial states
   - Initial parameter values
   - Initialization strategies

6. **Configuration Summary**:
   - Parameter file format recommendations for each parameter type:
     - Initial conditions
     - Initial parameters (initializations)
     - Configuration summary

Here are the key parameters and their corresponding descriptions in JSON format:
```json
{
  "model_matrices": {
    "A": [
      {
        "name": "statespace",
        "dimensions": 4,
        "structure": 6,
        "interpretation": "states"
      },
      {
        "name": "observation",
        "dimensions": 10,
        "structure": 32,
        "interpretation": "observations"
      }
    ],
    "gamma": [
      {
        "name": "precision_parameters",
        "dimension": 4,
        "shape": (6,)
      },
      {
        "name": "bias_parameters",
        "dimension": 10,
        "shape": (32,)
      }
    ]
  },
  "precision_params": [
    {
      "name": "precision_parameters",
      "dimensions": 4,
      "structure": 6,
      "interpretation": "prior"
    },
    {
      "name": "bias_parameters",
      "dimension": 10,
      "shape": (32,)
    }
  ],
  "confidence_params": [
    {
      "name": "confidence_parameters",
      "dimensions": 4,
     