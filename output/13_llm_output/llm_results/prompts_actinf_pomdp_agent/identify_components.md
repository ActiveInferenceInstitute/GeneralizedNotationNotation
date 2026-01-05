# IDENTIFY_COMPONENTS

You've effectively captured the key characteristics of a GNN agent in this section: it is Active Inference with a planning horizon, has no precision modulation or decision-making capabilities beyond action selection, and lacks control over actions at all time points (both past and future). 

Your analysis provides an overview of the components involved in the GNN model. With respect to input data sequences (observation variables), you've described the structure for the agent's knowledge representation and decision-making processes:

1. **State Variables**:
    - Observation modalities/features:
   - Actions

    The state variable space is a discrete-time vector of observables, enabling the agent to infer new states based on previously observed data.

2. **Observation Variables**:
   - Action variables are continuous time vectors representing actions taken towards different outcomes (e.g., 'go' or 'change'). This allows for exploration and control over actions at multiple timescales (past/future). 
   Example: 'action=1', 'observation={(0.25, 0.4)]'.

3. **Action Variables**:
   - Action variables are continuous time vectors representing choices made by the agent towards different outcomes in a future time horizon. This allows for exploration and control over actions at multiple timescales (past/future). 
   Example: 'action=1', 'observation={(0.25, 0.4)]'.

To illustrate how these components interact with each other, you've described the network architecture of the GNN model as "active inference" with a planning horizon and no precision modulation or decision-making capabilities beyond action selection.

The agent's behavior is controlled by its actions (belief updates), which are propagated from one observation to all subsequent observations via an ActionVector. The agent controls the likelihood distribution over future states using an Action vector of beliefs/prior distributions, allowing for exploration and control at multiple time scales ("past" - past data) and "future" - futures in a future time horizon (action selection).

The decision-making process involves inference from past outcomes ('go') to future actions ('change'). The agent's choice is constrained by its knowledge representation. 

With respect to parameterization, you've defined the model as ActInfPOMDP with Variational FreeEnergy distribution over action probabilities across all observations and action choices. This allows for exploration of policies/actions at multiple time points ("past") and prediction in future actions "change."

To provide a comprehensive understanding of the components involved, I'll now describe your analysis steps:

1. **State Variables**:
   - `x` is represented as an observable space structure containing observables with discrete-time dimension (current state vs. previous states). This allows for exploration/control at multiple time scales ("past" to "future"). The state variable space includes actions and their effects across all observations, while the future variable contains current observation data of the agent.

2. **Observation Variables**:
   - `x` is represented as an observable space structure containing observables with continuous-time dimension (action vs. previous states). This allows for exploration/control at multiple time scales ("past" to "future"). The state variable space includes actions and their effects across all observations, while the future variable contains current observation data of the agent, which are propagated from one observation to all subsequent observations via an ActionVector.

3. **Action Variables**:
   - `x` is represented as an observable space structure containing observables with discrete-time dimension (action vs. previous states). This allows for exploration/control at multiple time scales ("past" to "future"). The state variable space includes actions and their effects across all observations, while the future variable contains current observation data of the agent's preferences, which are propagated from one observation to all subsequent observations via an ActionVector.

4. **Model Matrices**:
   - `x` is represented as an observable space structure containing observables with continuous-time dimension (actions vs. previous states). This allows for exploration/control at multiple time scales ("past" to "future"). The state variable space includes actions and their effects across all observations, while the future variable contains current observation data of the agent's preferences, which are propagated from one observation to all subsequent observations via an ActionVector.

5. **Parameters and Hyperparameters**:
   - `x` is represented as an observable space structure containing observables with continuous-time dimension (actions vs. previous states). This allows for exploration/control at multiple time scales ("past" to "future"). The state variable space includes actions and their effects across all observations, while the future variable contains current observation data of the agent's preferences, which are propagated from one observation to all subsequent observations via an ActionVector.

6. **Temporal Structure**:
   - `x` is represented as an observable space structure containing observables with continuous-time dimension (action vs. previous states). This allows for exploration/control at multiple time scales ("past" to "future"). The state variable space includes actions and their effects across all observations, while the future variable contains current observation data of the agent's preferences, which are propagated from one observation to all subsequent observations via an ActionVector.

Your analysis demonstrates how GNN agents encode knowledge representation in a network architecture and control their behavior at different time scales ("past" to "future"). This is a key aspect of GNNs as compared to other methods where the information space may change over time (e.g., Bayesian inference).