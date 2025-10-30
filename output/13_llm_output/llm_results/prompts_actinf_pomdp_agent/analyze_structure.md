# ANALYZE_STRUCTURE

Here's a detailed analysis of the Active Inference model and its components:

**Graph Structure:**

1. **Number of Variables**: There are 3 variables (states, actions, preferences), with 6 types (observations) and 4 types for each variable. The total number of variables is 7*2=14.

2. **Variable Analysis**:

   - **State Space**:
   - **Transition Matrix**:
   - **Policy Vector**: There are 3 states and 4 actions, with 6 possible sequences from each state to the next. This matrix represents all possible sequence combinations of states and actions in a continuous manner (i.e., each observation has probability distribution over both states and actions). The variable structure is as follows:
   - **State Sequence**: Each state can be considered as having two transitions, one between consecutive states and one after transitioning to the previous state. There are 6 possible transition sequences from states with a length of 2 (1-based sequence) or 3 steps (forward), with 4 steps for each transition.
   - **Next State Sequence**: This is an arrow pointing towards the next action, with two transitions between consecutive states and one forward to reach the previous state. There are also 6 possible actions that can be taken after transitioning from one state to the next, with 5 actions for each sequence (forward).
   
**Variable Analysis:**

1. **State Sequence**:
   - **Randomly Choose Observation**: Each observation has probability distribution over both states and actions. This random choice represents all possible choices in each observation space.
   - **Probability Distribution of Actions**: There are three actions with probabilities of 0.9, 1.0, and 2.5 times different from each other (uniform).

   - **Randomly Choose Observation**: The same procedure is used for the random choice of observed observations.

2. **Transition Matrix**:
   - **Forward Sequence**: There are two forward transitions with probability of 0.9 over each observation space.
   - **Backward Sequence**: There are three backward sequences with probabilities of 1.0, 2.5 times different from each other (uniform).
   
**Policy Vector**:
   - **Initial Policy Matrix**: There are 3 initial policy vectors in the form:
      - [initial_state] ([states])
      - [final_policy],

      - [last_action];

      - [next_observation]. Each vector is a transition matrix from one state to the next, with each vector having probability of 1.0.
   
   - **Forward and Backward Forward Policies**: There are 3 forward policies in the form:
      - [forward] ([states])
       - []([observations]). Each policy has probabilities of 0.9 (uniform) or 2.5 times different from each other, which means that there is no connection between states at a time step. This corresponds to choosing one observation for forward sequence and the next one as part of backward sequence with probability 1/3 and another random choice at time step.
   
   - **Backward Forward Policies**: There are 2 backward policies in the form:
      - [forward] ([states])
       - []([observations]). Each policy has probabilities of 0.9 (uniform) or 2.5 times different from each other, which means that there is a connection between states at time step and one random choice at later observation. This corresponds to choosing the last observed observation as part of forward sequence with probability 1/3 and backward sequence without any action at later observation.

 **Constraints:**
   - **InitialState**: There are no constraints on state space, action space, or policy. The only constraint is that there must be transitions between states for each observation (including one transition in the initial state), allowing the agent to choose actions based on their probabilities of transitioning from a particular state. This ensures that the agents have access to available information and can make decisions with probability proportional to the change they receive, regardless of the order or choice at subsequent observations; this is crucial for efficient inference.
   - **InitialState**: There are no constraints on initial states or actions (this constraint was implicit in the model's implementation). The only constraint is that there must be a transition between any two observed states (including one across to another), allowing the agent to make decisions based on their probabilities, without knowing where they end up; this ensures that it can perform inference with probability proportional to the change from state at time step.
   - **InitialState**: There are no constraints on initial actions or policies (this constraint was implicit in the model's implementation). The only constraint is that there must be a transition between any two observable observations, allowing the agent to make decisions based on their probabilities of transitioning from one observation to another; this ensures that it can perform inference with probability proportional to the change from state at time step.
**Type:**
   - **Action**: There are no type constraints (this constraint was implicit in the model's implementation). The only type constraint is that there must be a transition between any two observed observations, allowing the agent to make decisions based on their probabilities of transitioning from one observation to another; this ensures that it can perform inference with probability proportional to the change from state at time step.
   - **Transition**: There are no type constraints (this constraint was implicit in the model's implementation). The only type constraint is that there must be a transition between any two observed states, allowing the agent to make decisions based on their probabilities of transitioning from one observation to another; this ensures that it can perform inference with probability proportional to the change from state at time step.
**Type:**
   - **State Sequence**: There are no type constraints (this constraint was implicit in the model's implementation). The only type constraint is that there must be a transition between any two observed states, allowing the agent to make decisions based on their probabilities of transitioning from one observation to another; this ensures that it can perform inference with probability proportional to the change from state at time step.