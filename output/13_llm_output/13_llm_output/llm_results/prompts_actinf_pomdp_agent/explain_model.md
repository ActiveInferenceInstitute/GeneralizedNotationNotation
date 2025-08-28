# EXPLAIN_MODEL

Your answer: 

GNN Example: ActInf POMDP Agent v1 - GNN Representation and Active Inference Context
Here is a comprehensive explanation of the model's purpose, components, core parts, and practical implications for understanding how this agent performs data-driven inference in a GNN context.
**Purpose:**
This model is designed to represent a classic Active Inference agent that has been used in reinforcement learning research for a discrete Markov decision process with one observable modality (observation) and multiple hidden states (hidden state). The agent's role is to make decisions based on the observed outcomes, which are encoded as log-probabilities over observations.
**Core Components:**

1. **StateSpaceBlock**: This block represents each observation/decision pair in a GNN graph represented by `g`. It stores the action probabilities and actions for that particular state.

2. **TransitionMatrix**: It maps observed transitions to states, indicating whether an action is taken towards the current state or another target state. It allows for actions selection from previous states to choose actions (e.g., policy assignment).
**ProbabilityVectorBlock**(s): This block represents a subset of states and actions that are available in a GNN graph represented by `v`. Each state has an associated probability vector that is used as the transition matrix between states, which can be computed using inference networks to update belief probabilities.

3. **ActionVectorBlock**: It maps observed actions (inferred from the previous step) back into observed actions for subsequent steps in the GNN flow graph. Actions are represented by a sequence of probability vectors corresponding to each possible action. For instance, an action "t" is mapped to a sequence of probabilities that describe it:
   - Actions with lower values represent higher-priority actions ("action") and are sampled from initial policy prior distribution.
   - Actions with higher values represent lower-priority actions (e.g., "stay", "move").

4. **TransitionVector**: It maps observed transitions between states to action probabilities, which are used as the transition matrix for inference in this context.
**ActionPermutation:** This block represents each of those actions from policy prior distribution and its corresponding action vector.
**Observation:** Each observable is a sequence of two numbers:
   - The first number denotes an observation (represented by state-wise log-probability)
     - The second number indicates the next observed observation, which can be one step forward or backward along path to current state/observation.
   
In this context, each observation corresponds to one action on the chain leading to the next observable. This allows for making policy decisions based on actions taken towards a target (target states) and actions chosen from previous states ("actions").
**InitialStateProbabilities:** These are computed using inference networks that represent the POMDP graph represented by `v`. It is computationally expensive because they require sequential computation of action probabilities across paths. However, it allows for prediction based on beliefs in each observation (obeyables), which can be further used to make decisions in future steps.
**PolicyVector:** This block represents policies that are computed using the inference networks representing GNN graphs represented by `v`. It maps observed actions into predicted actions and their corresponding probabilities over states, which allows for policy selection from previous states ("actions") to choose actions (action probabilities) based on beliefs about next state/observation.
**Observation:** These are obtained through applying action to observation in order of probability towards the current state/obeyable, representing a sequence of observed outcomes along path back to current state or observing across history for instance that is used in reinforcement learning as well as decision-making based on future actions. This enables predictions about expected actions (facts) and their associated probabilities during each step using inference networks represented by `v`.
**InitialStateProbabilities:** These are computed using inference networks, which determine the probability of observation after moving forward through a chain of observations to one state or observing across history for an action from policy prior distribution. This enables predictions about actions towards that specific next observable ("action"). In addition, this provides information on past behaviors based upon observed behavior in previous steps (beliefs) and actions used as decisions.
**ActInfPOMDPAgent:** This is the agent model implemented within GNN graphs represented by `v`. It represents an inference-driven agent with active inference capability applied to predictions of next observable paths along history for each observation, which allow for data-driven inference in a GNN context.
Here is what this agent does:

1. **Initialization**:
   - The initial state and actions are initialized from the observed actions (observes) across different histories, given their corresponding probabilities over states. This provides accurate predictions of expected future outcomes based on beliefs about next observable path forward.
   
   - It also performs inference into past observations to generate predictions of actions towards its goal-directed action (policy).

2. **State Transition**:
   - Actions are initialized from the observed actions across all previous histories, resulting in a sequence of states and actions that compute predicted probabilities of next observation's outcomes based on beliefs about current observation/action. It performs inference into past observations to generate predictions of expected future outcome for each observable path forward (actions) towards its goal-directed action.

3. **Action Permutation**:
   - Actions are mapped from observed transitions across history to their predicted actions, which can be computed using the inferences derived within GNN graphs represented by `v`. It also performs inference into past observations to generate predictions of future observables for each observable path forward.
**Observation:**
   - The observation data is collected as a sequence of two number from previous histories and observed through step-wise sequence across history, given their corresponding probabilities over the current state/obeyable.

4. **InitialStateProbabilities**:
   - It computes initial policies by mapping actions towards next observable path forward in history and uses them to predict predicted outcomes based on its beliefs about future observations for each observable observation (observation).
**InitialActionProbabilities:**
   
   - The first number represents the current observable path forward, which can be one step back or backward along a previous path. It computes predictions of action probabilities across path forward from observed actions across all histories in order to generate predicted outcome based on its past beliefs towards an initial state/observation and actions (next-state).

5. **Constraints**:
   - The agent takes into account its current observable path forward using the observation data, predicting future observables as a sequence of probabilities corresponding to observations and action predictions across paths forward along history in order to generate predicted outcomes based on information about observed actions that computed from inference networks represented by `v`.

6. **Constraints**:
   - The agent takes into account its current observable path forward using the observation data, predicting future observables as a sequence of probabilities corresponding to observations and action predictions across paths forwards along history in order to generate predicted outcome based on information about observed actions that computed from inference networks represented by `v`.
**ModelHistory:**
   
   - The model histories contain the updated state transition probabilities towards each observable/action, with time-dependent probability gradients. It also performs prediction into future observation path forward (state) and predictions of actions towards it's goal direction based on belief update using action selection from history represented by `v`.
**InitialPolicy:**
   
   - The policy is initialized as a sequence of probabilities, which are computed using inference networks representing GNN graphs represented by `v`, given their corresponding probability across the current path forward. It also performs prediction into future observation path forward and predictions based on its belief updating from observed actions towards goal-directed action (policy).