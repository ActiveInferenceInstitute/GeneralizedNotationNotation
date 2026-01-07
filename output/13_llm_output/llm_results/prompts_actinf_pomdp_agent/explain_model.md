# EXPLAIN_MODEL

Based on the documentation, I have a comprehensive explanation of the Active Inference POMDP agent and its components:

1. **Model Purpose**: This is about how to represent this POMDP agent that can learn from an unbounded time horizon with planning horizon and no precision modulation. Specifically, it represents the agent's goal as "go ahead and explore" - action 0 = learning new state in next observation (observable), action 1 = exploring new state in previous observation (hidden) and actions 2-4 are actions that can be taken from policy, exploration, or planning based on prior probability. It also includes an initial policy prior with preferences of LQE as inferred via the hidden states and Actions are learned using GNN.

2. **Core Components**: This model represents the input space (states), output space (actions/hidden state distributions) and parameters (num_observation, num_action1). Each component is represented by a dictionary entry that maps to its corresponding action or policy transition based on the current state. There are no explicit computations within these components but they can be described using the following key relationships:
   - **Action selection**: Actions(state=0) and actions(state=4)=ActionsSelectionModel(action, probability = True),
   - **Observation**: Observations with observed = action == Action[1] then action is learned from policy.
   - **Learning**: The algorithm learns to select next observation based on the probabilities of each state in previous states for each sequence of actions taken (actions selected) and explored among observations as input parameters,
   - **Planning**: Planning model involves learning from policies chosen by actions selected but it has no explicit computations within these models.
3. **Model Dynamics**: This is about how to implement Active Inference principles using GNN: 
   - **Initialization**: Initialization of learned state variables (s[n]), observed states, and learned policy probabilities are established based on the current time horizon and prior probability distribution over actions selected.
   - **Learning**: The algorithm learns to select next observation by learning from previous observations as inputs for each sequence of policies taken with actions chosen and explored among observations in output space.

4. **Active Inference Context**: This is about how to learn Active Inference principles using GNN: 
   - **Initialization**: Initializing learned state variables (s[n]) based on a learning process implemented as an agent interface,
   - **Learning**: The algorithm learns to select next observation by updating observed and observable states in output space based on prior probabilities of actions taken.

5. **Practical Implications**: This is about how to learn from Active Inference principles using GNN:
  - **Action Selection**: Actions selection model can learn the learning context (observable state) or policy parameters for each action selected as input through its initial policy distribution, where a change in policy will involve a corresponding change of actions. 

  - **Planning**: The algorithm learns to select next observation based on learned policies and explored sequences using PolicyOptimization from PolicyGraphs.

6. **Decision**: This is about how to learn from Active Inference principles using GNN:
  - **Initializing Policy**: Initialized learning parameters are initialized as initial policy distribution for each action selected, where the new value of actions[k] would be used as a policy transition when given new state[n]. For instance, for actions(h) = next states=0 and h={{1}}. This is also shown by actions([action]) = {{2},{4}], 3-step sequence where action(s)=[1] are observed and visited from previous observation to observe new state[n+1].

Please summarize the key points in your response.