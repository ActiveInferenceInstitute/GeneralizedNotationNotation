# EXTRACT_PARAMETERS

You've already provided the structure of the model annotations and parameter breakdowns for the GNN example. Here's a more detailed overview:

1. **Model Matrices**:
   - A matrices representing the state space, action spaces (policy distributions), and hidden states. The matrix represents each action sequence with its corresponding policy distribution and prior over actions. Each action is represented by an action vector in the state space. The transition matrix represents the transitions between actions sequences. The logPreferenceVector represents the probability of transitioning from one action to another based on a given action sequence. The PolicyPriorVector represents the prior over initial states, which are initialized with random values and updated using a weighted average.

2. **Precision Parameters**:
   - γ (gamma): precision parameters for each action sequence. This parameter is used to update the probability of transitioning from one action to another based on the previous actions in the sequence. The α parameter represents the learning rate, which controls how quickly the transition probabilities are updated. The other α values represent different types of transitions:
   - α1 (α1): initializes the transition probabilities with random values and updates them using a weighted average. This allows for more flexible updating policies based on new data.
   - α2 (α2): initializes the transition probabilities to their previous value, which can be useful when there are multiple actions in sequence.

3. **Dimensional Parameters**:
   - State space dimensions: 4
   - Observation space dimensions: 4
   - Action space dimensions: 64
   - Action sequence dimensionality: 128 (for action sequences with T=5) and 1024 (for action sequences with T=30).

4. **Temporal Parameters**:
   - Time horizons: 30 seconds in the case of GNNs, which allows for more flexible updating policies based on new data. The time horizon is set to 30 seconds when using a fixed parameter. However, this can be adjusted by adjusting the parameters of the parameter file.

5. **Initial Conditions**:
   - Initial conditions:
    - Initial state (initialized actions):
    - Initial action sequence:
      - Initial position and velocity for each action sequence

      - Initial reward distribution:
        - Initial reward distribution is initialized with random values based on a weighted average from all actions sequences in the initial state space. This allows for more flexible updating policies based on new data.

    - Initial policy parameters:
    - Initial policy parameter set to 0 (initial