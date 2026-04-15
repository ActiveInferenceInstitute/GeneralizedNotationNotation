# EXTRACT_PARAMETERS

Based on the provided documentation, here are the key parameters for the GNN model:

1. **Model Matrices**:
   - A matrices representing the state space and observation spaces (represented as a list of lists)
   - B matrices representing the transition matrix and action distributions (represented as a list of lists)
   - D matrices representing the hidden states, actions, and observations (represented as a list of lists)

2. **Precision Parameters**:
   - γ: precision parameters for each factor
   - α: learning rates and adaptation parameters
   - Other precision/confidence parameters

Here are some key parameter breakdowns:

  1. **Model Matrices**:
    - A matrices representing the state space and observation spaces (represented as a list of lists)

    Parameters:
      - A[observations, states] = [
        [[0.7, 0.3, 0.2],
          [[0.5, 0.1, 0.6]]])
    - B[states_next, states_previous] = [[(0.4, 0.8), (0.9)]
      for each state and observation

      Parameters:
        - A[observations, states] = [
          [[0.7, 0.3],
          [[0.5, 0.1]]])
    - B[states_next, states_previous] = [[(0.4, 0.8), (0.9)]
      for each state and observation

      Parameters:
        - A[observations, states] = [
            [[0.7],
            [[0.3]],
            [[0.5]]])
    - B[states_next, states_previous] = [[(0.4, 0.8), (0.9)]
      for each state and observation

      Parameters:
        - A[observations, states] = [
          [[0.7],
          [[0.3]],
          [[0.5]]])
    - B[states_next, states_previous] = [[(0.4, 0.8), (0.9)]
      for each state and observation

      Parameters:
        - A[observations, states] = [
            [[0.7],
            [[0.3]],
            [[0.5]]])
    - B[states_next, states_previous] = [[(0.4, 0.8), (0.9)]
      for each state and observation