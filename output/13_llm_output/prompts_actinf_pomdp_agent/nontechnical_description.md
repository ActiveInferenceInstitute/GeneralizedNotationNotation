# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

You've already done that! Your breakdowns nicely summarize the different components of the GNN representation:

1. **Action Sequence**: This defines the sequence of actions taken by the agent (represented as a set of objects). It includes actions, states and probabilities for each action. In this case, you have 3 "actions" with 2 "states", one hidden state and one observable (each action is initialized to "hidden"), along with two initial policies: Policy "stay=1" and "leave=0".

2. **State Sequence**: This defines the sequence of states taken by the agent during its time running in this POMDP. In this case, we have 4 states and a hidden state that is fully controllable via action probabilities (including some actions). There are also three actions and two policies that allow for planning to take action at arbitrary times, as well as "stay=1" and "leave=0".

3. **Probabilities**: The belief associated with each observation, where each observable corresponds to a particular policy or action sequence in the POMDP. 

4. **State-action Transition Matrix**: This is where you specify the rules for your decision space (policy/actions). There are 5 transitions between states and actions:
   - 0 → -1
   - 1 → -1
   - 2 → +1
The probability of transitioning from one state to another depends on a certain policy or action sequence, based on their probabilities. This means that the transition matrix will depend on how you define your decision space (policy/actions). You would have some actions like "stay=0" and others like "leave=1".

5. **Previous Observations**: This stores information about previous observations for each state-action sequence in history. It is also used as a policy prior, allowing the agent to use its biases during future decisions based on past predictions (see the following sections).
You've already done that! Your breakdowns summarize all the components of the GNN representation nicely:

1. **Action Sequence**: This defines the sequence of actions taken by the agent in this POMDP. It includes actions, states and probabilities for each action. There are also "actions" with 2 "states", one hidden state and one observable (each action is initialized to "hidden"), along with two initial policies: Policy "stay=1".

2. **State Sequence**: This defines the sequence of states taken by the agent during its time running in this POMDP, representing a decision space for their actions at different times. There are also 3 transitions between states and actions that define where these decisions happen to be made (policy/actions). 

**Action Sequences:**
Actions come with action probabilities, so we know how to make "stay=1" or "leave=0". We can infer other actions from the probability of transitioning from one state to another based on their histories. For example, if a policy sequence has a "stay=1", then we expect an "action(s)" that is only taken during time t = 2.
Actions are represented as a set of objects and their initial probabilities (with probability) for each action. We can't infer from actions whether they will be visited by the agent at some future time, so this represents an action sequence in history. The probability for "stay=1" is then inferred to happen if we were able to avoid visiting it during the last 2 timesteps and return after a second.
**State Sequence:** This defines the sequence of states taken by the agent during its time running in this POMDP, representing a decision space where each state corresponds to an observation within history (policy/actions). 

**Probabilities:** This represents probability for actions in history to occur. It is also inferred from "stay=1" or "leave=0".

 **State-action Transition Matrix:** This is then used as a policy prior, allowing the agent to use their biases during future decisions based on past predictions (see the following sections).
**Previous Observations:** 
  These are stored in history for each observation and can be inferred from actions that have been visited by the agent. They represent the knowledge of where we will take our next action at a particular time step, given our previous observations.
They correspond to the fact that when an observation is visited during its timestep (policy/actions), it has "gone" to a state which is controlled by this policy and then went back to a state from which another observation was taken for subsequent time steps ("stay=1") later on (policies 2,3). This means that we can infer where actions will be visited in the next timestep based on where past observations are going.