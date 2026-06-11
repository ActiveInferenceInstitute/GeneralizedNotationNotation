# EXTRACT_PARAMETERS

Based on the provided specifications, here are the key parameters for Active Inference (AI) on GNNs:

1. **Model Matrices**:
   - A matrices representing the model structure and interpretation of the input data.
   - B matrices representing the model parameters and their roles.
   - C matrices representing the model parameters and their roles.
   - D matrices representing the model parameters and their roles.

2. **Precision Parameters**:
   - γ (gamma): precision parameters and their roles.
   - α (alpha): learning rates and adaptation parameters.
   - Other precision/confidence parameters, such as γ or α, are not specified in this specification but can be inferred from the context of the model description. However, they should also be considered when analyzing GNNs with noisy data.

3. **Dimensional Parameters**:
   - State space dimensions for each factor:
    - A matrices representing the state space dimensionality and interpretation.
    - B matrices representing the state space dimensionality and interpretation.
    - C matrices representing the state space dimensionality and interpretation.
    - D matrices representing the state space dimensionality and interpretation

4. **Temporal Parameters**:
   - Time horizons (T)
   - Temporal dependencies and windows:
    - Time horizons can be inferred from the context of the model description, but they should also be considered when analyzing GNNs with noisy data.

5. **Initial Conditions**:
   - Prior beliefs over initial states:
    - Initial parameter values:
      - γ (gamma): precision parameters and their roles.
      - α (alpha): learning rates and adaptation parameters.
      - Other precision/confidence parameters are not specified in this specification but can be inferred from the context of the model description. However, they should also be considered when analyzing GNNs with noisy data.

6. **Configuration Summary**:
   - Parameter file format recommendations:
    - The configuration summary is a concise and informative representation of the input data that summarizes the parameters for each factor in the model structure. It provides an overview of the model's behavior, including its predictions, uncertainties, and sensitivity to changes in parameter values.