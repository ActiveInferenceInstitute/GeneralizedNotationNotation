# EXPLAIN_MODEL

This is a comprehensive overview of the Generative Model (GNN) presented as part of the Machine Learning Foundation's GCN framework for Active Inference and GNN-based models for generative neural networks like DeepMind's AI agent model Ai2019POMDPs.

**Signature** 

This model represents a classic Active Inference agent for discrete POMDP with two hidden states (states_next, states_previous). The agents are initialized to the same observation and action distributions, but their behavior is influenced by actions taken during each iteration of the simulation. A sequence of policy transitions will steer them toward different actions, while a sequence of state choices in each path corresponds to a specific action selection.

**Core Components** 
   - **Likelihood Matrix**: A dictionary representing the likelihood maps over available actions for each observation and hidden states. This matrix encodes probabilities across multiple actions and their corresponding states.
   - **Transition Matrix**: A list of tuples containing one observation, one or more possible observables (outputs) from a given action selection, and the previous state/observation associated with that action.
   
   - **Probabilities**: The probability distribution over all observed observations over different actions across multiple actions based on these transition matrices. This matrix provides input for inference.
**Model Dynamics** 
   - **Actions**: Policy transitions (one policy each) that steer the agents toward specific actions and their associated observables, while also allowing for depth exploration of the network.
   - **State Transition**: Transition from one observable to a new observable based on previous states/observations or other transitions over actions leading back to previous action(s).
**Active Inference Context** 
   - **Initialization**: Initializing the network by assigning initial policy parameters (habit) and then starting inference. The inference is done iteratively until convergence in accuracy, learning rate, and evaluation metrics are reached.

Key Concepts:
   - **GNN**: Generative Neural Network models for neural networks like DeepMind's Ai2019POMDPs.

 **Active Inference** 
   - **Permute** **Initialization** *FIRST* *DIFFERENTLY* (every action selection is independent).
   - **Decay**: Decrease the cost of updating policy parameters based on past observations and actions, which helps to stabilize the network with an increase in training accuracy.

**Practical Implications** 
   - **Learning Curves**: A continuous learning curve where the algorithm's performance improves or stabilizes over time as more information is available/learned from it (e.g., through a generative neural network like DeepMind).

 **Decision Curves**: The rate at which the agent learns based on its actions and their corresponding observables changes depending on how well they learn. This can help predict when to update or stop learning.

**Example** 
   - **Initialization**: Initializing the network with all observed observations (observations).
   - **Decay**: Decreasing training accuracy by learning rate proportional to an observation value.
 
   
   - **Learning Curves**: Plotting the learning curve over time and exploring how well the algorithm improves or stabilizes as more information is learned from it, which helps predict when to update/stop learning.