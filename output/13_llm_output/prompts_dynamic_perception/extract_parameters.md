# EXTRACT_PARAMETERS

Based on the provided documentation, here are the key parameters for the GNN model:

1. **Model Matrices**:
   - A matrices representing the input and output data sets (e.g., `A`)
   - B matrices representing the learned beliefs (`B`), which represent the predictions made by the agent based on its actions (`D`, `F`, etc.)
2. **Initial Parameters**:
   - γ = 0.1
   - α = 0.5
   - Other parameters:
      - γ (gamma) is a hyperparameter that controls the rate at which the model learns from data points with different probabilities of occurrence, and it has various roles in enabling parameter exploration and tuning.

3. **Dimensional Parameters**:
   - State space dimensions for each factor (`A`)
   
   - Observation space dimensions for each modality (e.g., `s_t`), where `s_prime`, `o_t`, etc. are the observed states of the agent, which can be thought as a set of input data points with different probabilities of occurrence.

4. **Temporal Parameters**:
   - Time horizons (`T`)
   
   - Temporal dependencies and windows (`W`)
   
   - Update frequencies (e.g., `F`)
   
   - Time scales for each control factor (`C`)
5. **Initial Conditions**:
   - Initial parameters:
      - γ = 0.1

6. **Configuration Summary**:
   - Parameter file format recommendations
 
Here are the key performance metrics and their corresponding values in the GNN model:
- **Accuracy**: `accuracy` (number of correct predictions / total number of predictions)
- **Precision**: `precision`, which is a hyperparameter that controls the rate at which the model learns from data points with different probabilities of occurrence. It has various roles and priorities based on the type of parameter being explored.
- **F1 score**: `f1_score` (average precision), which is a metric for evaluating performance across multiple categories, including accuracy. It also has several values that can be tuned to focus on specific types of parameters or their role in achieving optimal performance.