# Technical Description

Prompt:

> Describe this GNN model comprehensively, in technical detail.

Response:

In this document:

1. The main content and code for the GNN agent implementation.
   - ModelName: Active Inference POMDP Agent
   - ModelAnnotation: Likelihood mapping hidden states to actions, no planning or decision-making based on policy and prior knowledge of action choices, no planning history in order to initialize initial hypothesis probability distribution (H)

2. ModelInfo for the GNN agent implementation with 1 observation modality ("state_observation") and one observable state factor ("location").
   - Information: Likelihood map, transition matrix, preference vector, policy parameters.
 3. ModelAnnotation: Initialization of Actionspace.
   - Instruction: A set of actions is initialized based on the given action selection (action=), policy prior probabilities are computed using Variational Inference and learned from observed actions/states.

4. ModelInfo for the GNN agent implementation with 1 observation modality ("state_observation") and one observable state factor ("location").
   - Information: Likelihood mapping hidden states to observations, policy map information is learned based on observed actions/states using Variational Inference.

5. ModelAnnotation: Initialization of Hidden State distribution and Prior Distribution for Policy Estimation.
   - Instruction: A set of Actionspace's history is initialized based on the action selection given an observation (observations).

6. ModelInfo for the GNN agent implementation with 1 observation modality ("state_observation") with identity mapping, prior over actions and habit usage as initial policy prior.
   - Information: Actions are initialized by applying a single action to each observed observation/policy space, and subsequent actions selection is based on history of past actions and policies.

7. ModelAnnotation: Initialization of Belief distribution for Agent Policy estimation.
   - Instruction: A set of actions is initialized using the history established in modelInfo#InitializingHabitDistributionSet() which has been learned from observed actions/states and can be used to initialize a belief based on policy prior probabilities and action selection histories.

8. ModelAnnotation: Initialization of History of Actionspace's history of all available actions (state_observation).
   - Instruction: A set of Actions are initialized by applying an action with respect to each observed observation/policy space, and subsequent actions selections have been learned from observing observations based on the policy prior probabilities that were trained using Variational Inference.

9. ModelInfo for the GNN agent implementation with 1 observation modality ("state_observation") with identity mapping, prior over all observable states and habit usage as initial policies prior.
   - Information: History of Actionspace's history is initialized by applying a single action to each observed observation/policy space, then subsequent actions selection are based on state of the art and policy previews.

10. ModelAnnotation: Initialization of History of Actionspace's history of all observable states (i.e. all actions).
   - Instruction: A set of Actions is initialized by applying an action with respect to each observed observation/policy space, then subsequent actions selection are based on state and policy histories that have been learned using Variational Inference and habit usage as initial policies prior.