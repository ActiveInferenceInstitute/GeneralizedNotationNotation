# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

GNN model provides an interactive interface for visualizing and exploring the behavior of your agent in action. The following sections provide information on the components of the GNN model:

1) **Model Context**: It gives a basic overview on what the system is doing, its parameters (model annotations), and how to understand them by reading through this section. It also provides an understanding of where each component fits into the overall context.

2) **Action Selection**: It describes actions taken by the agent in action selection. The Actions are represented as a dictionary of objects with 3 elements: 

- `state`: the current observation being analyzed (represented as string values). These can be either "observation" or "actions").
  - `observe_next` and `observate_next` describe where each observation is located in the next level.
  - `action`, which can have an action name, a sequence of actions, etc.
- `actions`: list of actions that are currently being performed by the agent for its current state (represented as dictionary). This also includes actions associated with previous states and actions used to generate new observations.

3) **Learning**: It describes how to learn from observed data or prior knowledge to infer future state information based on past learned information and interactions between observation types. The Learning can be done using Action Selection, Hidden State, Policy, Habit, or other methods that provide inference capabilities (such as Hidden States).
4) **Action Selection**: This is a list of actions implemented in the GNN agent. Actions are defined by representing states where they have probability associated with each action being taken (represented as strings). It also includes actions used to generate new observations when combined with other actions/actions.
5) **Learning Curve**: It provides an overview on how learning happens for the current observation at different levels of information, which enables exploration and prediction capabilities in action selection. This can be represented by a graph structure in the GNN model (like a directed acyclic graph). The learning curve is defined based on these steps:
  - When observing next state and actions used to generate new observations from it?
    - If there are states, observe_next() returns the observed observation at current time.
  - when observing next state and Actions applied as prior?
    - if a single state has probability associated with action that will be taken in future next state?
      If not yet have this behavior (so we can add to learning array), apply actions from previous states.
    - if another state or two has probability associated with an action already, but its probabilities are different (means the next state is initialized and after applying actions)
  - when observing next state and Actions applied as sequence of actions?
    - If no specific path exists for observing next state to be taken based on previous states/actions. Then we apply new observations from other parts of observation at current time (follow chain of action, observe_next() returns observed observation).
6) **Learning Iterations**: This is a list that lists the actions performed by the GNN agent at each level of its learning history. These correspond to where they will be applied in future actions based on previous learned information and interactions with other actions/actions (including Actions, Hidden States, etc.) 
The Learning Iteration is represented as a dictionary named `LearningIter`. It defines how learning progresses from current observation to the next level of learning until we reach the goal of prediction for observing state at the top-left. The history can be represented by a graph structure in GNN model (like an Undirected Graph). A transition table represents the action and their probabilities over previous states that lead to future actions.

7) **Learning Curve**: This shows how the learning curve progresses through the exploration of the goal space based on information learned from observed observation data or prior knowledge. It can be represented by a graph structure in GNN model (like an Undirected Graph). A sequence of Actions and Actions are defined as steps that correspond to actions performed at different levels of learning history. For example, if we learn "actions 1-2" when learning the first step, then "actions 3-4" is learned from this action and so on (remember that a single state already has probabilities associated with previously observed actions).
Learning Iterations are represented as a dictionary named `learningIter`. It defines how learning progresses through the exploration of the goal space based on information learned from observed observation data or prior knowledge. A transition table represents the actions performed by the GNN agent at each level of its learning history (representing steps that correspond to previous actions) and corresponding probabilities for the given path forward in time during learning process is stored in a graph structure in GNN model. A sequence of Actions and Actions are defined as steps that correspond to actions per step and corresponding probabilities from prior knowledge, which can be represented by a chain of Actions (representing an inputted state). 

## LearningIter 2: An Iteration
LearningIter= {
  'action': "Action",
  'steps_forward': [],
  'actions', # current action is applied as previous one and next state was initialized.
  'states',  
    - [state = (observation, action),
                  action = actions[next(previous step)]]  # Apply the given action into the following states.
  # Each time we apply a particular action in each subsequent steps:
  # StepForward() returns a dictionary of Actions and their probabilities.
  StepBackward():
    # Steps forward involve applying next actions from previous states.
    Actions = [next(previous step) for i, state in enumerate(observation)], 
    # Next is applied as the current one has probability associated with it now (as an action), which means the observed observation was initialized and after its subsequent actions have their probabilities associated with it).
    Action=action_probability()

  # Steps backward involve applying next actions from previously stored states.
  StepBackward():
    # Steps back are done as a step forward, but there is still probability association between the last state-given action and the next one that has given this probability for the previous steps.
    Actions = [next(previous step) for i, (state, action)=transition_dict from observation to state.]
  # Next in a backward iteration involves applying all actions on the right path forward into the following states; no prior knowledge is involved here because we know that there are probabilities associated with each of these actions already.
    Actions = [next(previous step) for i, (state, action)=transition_dict from previous observation to next state.]
  # Steps backward in a forward iteration involves applying all actions on the left path forward into the following states; prior knowledge is involved here because we know that there are probabilities associated with each of these actions already.
    Actions = [previous step for i, (state, action)=transition_dict from previous observation to state]
}