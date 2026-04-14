# SUMMARIZE_CONTENT

Here's a concise overview of the GNN specification:

**Overview:**
This is a generic active inference model that can be used as an example or in conjunction with other models like GMMs or Bayesian networks. It provides a framework for analyzing and modeling epistemic foraging behavior, where agents navigate through a maze to explore different locations while exploiting rewards from nearby arms.

Key Variables:

1. **Hidden States**: A list of 4 variables representing the location state (center, left arm, right arm, cue) that encode information about the agent's current position and exploration direction. These states are used for inference purposes but not explicitly defined in this specification.

2. **Observations**: A list of 3 variables representing the location observation (left or right arm). These observations represent the agent's current state during exploration, while also providing context to guide future actions.

3. **Actions/Controls**: A list of 4 variables representing the agent's actions and control strategies (choices) in relation to each other. These actions are used for inference purposes but not explicitly defined in this specification.

4. **Action**: A list of 2 variables representing the action taken by the agent during exploration, which can be either "exploit" or "visit". This is a crucial aspect of the model design and should not change based on the specific actions being explored.

**Key Parameters:**

1. **Hidden States**: A list of 4 variables describing the location state (center, left arm, right arm). These states encode information about the agent's current position and exploration direction.

2. **Observations**: A list of 3 variables representing the location observation (left or right arm) that represent the agent's current state during exploration. This is a crucial aspect of the model design but should not change based on the specific actions being explored.

**Critical Parameters:**

1. **Random Direction**: A parameter describing how to explore different directions in space, which can be either "exploit" or "visit". The choice between these two strategies depends on the specific actions taken by the agent during exploration.

2. **Initialization**: A parameter representing the initial state of the maze (center, left arm, right arm). This is a crucial aspect of the model design but should not change based on the specific actions being explored.

**Notable Features:**

1. **Random Direction**: A parameter describing how to explore different directions in space, which can