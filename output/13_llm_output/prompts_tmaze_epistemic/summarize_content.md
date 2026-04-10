# SUMMARIZE_CONTENT

Here is a concise summary of the GNN specification:

**Overview**
This GNN represents an active inference agent that navigates a T-shaped maze with 4 locations (center, left arm, right arm, cue) and two observation modalities (location and reward). The model takes into account the location likelihood, reward likelihood, and context dependence to make decisions based on available information. It is capable of exploring different actions in each direction, exploiting or avoiding certain locations.

**Key Variables**
- **Location**: A list containing coordinates for each location within the maze (e.g., center, left arm).
- **Reward**: A list containing coordinates for each reward observed by the agent (e.g., right arm) and its corresponding reward value (e.g., cue).
- **Context**: A list containing coordinates for each context observed by the agent (e.g., cue location, left arm), along with their respective rewards (e.g., right arm).
- **LocationObservation**: A list containing coordinates of all locations within the maze and corresponding observations (e.g., center, left arm) from the agent's perspective.
- **Action**: A list containing coordinates for each action observed by the agent (e.g., direction), along with their respective rewards (e.g., cue).
- **ContextPreferenceVector**: A list containing coordinates of all actions and corresponding reward values across different contexts, along with their respective probabilities.
- **LocationProbabilities**: A list containing coordinates for each location within the maze and corresponding probability distributions over the available locations.
- **RewardProbabilities**: A list containing coordinates for each reward observed by the agent (e.g., cue) and its corresponding probability distribution across different rewards.
- **ContextProbabilities**: A list containing coordinates of all actions and their corresponding probabilities across different contexts, along with their respective probabilities.
- **LocationPreferenceVector**: A list containing coordinates of all locations within the maze and corresponding preferences over different actions (e.g., direction).
- **RewardPreferenceVector**: A list containing coordinates of all rewards observed by the agent (e.g., cue) and its corresponding preferences across different rewards.
- **ContextPreferenceVector**: A list containing coordinates of all actions and their corresponding preferences over different rewards, along with their respective probabilities.