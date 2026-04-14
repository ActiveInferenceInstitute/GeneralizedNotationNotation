# SUMMARIZE_CONTENT

Here's a concise summary of the GNN specification:

**Model Overview**
This is a hierarchical active inference POMDP (POMDP) that models a sequence of actions (`A`) and their associated beliefs (`B`). The input data consists of observations, hidden states, and actions. The model takes in two types of data: `actions`, which are the inputs to the model; and `beliefs`, which represent the predictions made by the model based on the input data.

**Key Variables**
- **hidden_states**: A list containing information about the state space where the model models actions (`A`) as a sequence of observations (inputs) and beliefs (`B`). The hidden states are used to encode the prediction for each observation, which is then propagated through the network.
- **observations**: A list containing data from the input data that will be fed into the model. It can include any type of data: actions, predictions, etc.
- **actions**: A list containing information about the actions being modeled (`A`) and their corresponding beliefs (`B`). The actions are used to encode the prediction for each observation.
- **beliefs** (optional): A list containing information about the beliefs represented by `action`, `observation`, or `control`. These can be used to represent predictions made by the model based on the input data.

**Critical Parameters**
- **num_hidden_states**: The number of hidden states in the POMDP, which are used for encoding actions and their corresponding beliefs (`B`).
- **num_obs_l1**: The number of observations that are labeled as `low level`, meaning they have no action or belief associated with them. These can be considered to represent a sequence of actions (e.g., "action A").
- **num_actions_l1**: The number of actions in the POMDP, which are used for encoding predictions (`B`).
- **num_context_states_l2**: The number of actions and their corresponding beliefs that have no action or belief associated with them. These can be considered to represent a sequence of actions (e.g., "action A").
- **num_timesteps**: The number of timesteps in the POMDP, which are used for updating the model parameters (`timescale_ratio=5`).

**Notable Features**
This is a hierarchical active inference POMDP with two types of