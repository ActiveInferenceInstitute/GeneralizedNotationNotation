# Runtime Behavior

Prompt:

> Describe what happens when this GNN model runs and how it would behave in different settings or domains.

Response:

You can use the following code to generate a GNN representation of the Hierarchical Active Inference POMDP:

1. Create an instance of the `GNNRepresentation` class with the following attributes:
   - `model`: A two-level hierarchical POMDP represented by the `HierarchicalActiveInferencePomdp`.
   - `state_space`: A list of state spaces representing the POMDP states and actions. Each state space is a dictionary containing key-value pairs, where each value represents an observation in the state space.
   - `action_states` (optional): A list of action states for each level 1 transition. Each action state is a dictionary containing key-value pairs, where each value represents an action in the action state.

2. Create a copy of the `model`, which will be used to represent the POMDP:
   - `state_space`: A list of state spaces representing the POMDP states and actions. Each state space is a dictionary containing key-value pairs, where each value represents an observation in the state space.

3. Create a copy of the `action_states`, which will be used to represent the POMDP action states:
   - `state_space`: A list of state spaces representing the POMDP actions and their corresponding observables (actions). Each state space is a dictionary containing key-value pairs, where each value represents an observable in the state space.

4. Create a copy of the `model`, which will be used to represent the POMDP:
   - `state_space`: A list of state spaces representing the POMDP states and actions. Each state space is a dictionary containing key-value pairs, where each value represents an observation in the state space.

5. Create a copy of the `action_states`, which will be used to represent the POMDP action states:
   - `state_space`: A list of state spaces representing the POMDP actions and their corresponding observables (actions). Each state space is a dictionary containing key-value pairs, where each value represents an observable in the state space.

6. Create a copy of the `model`, which will be used to represent the POMDP:
   - `state_space`: A list of state spaces representing the POMDP states and actions. Each state space is a dictionary containing key-value pairs, where each value represents an observation in the state space.

7.