# Nontechnical Description

Prompt:

> Describe this GNN model comprehensively, in non-technical language suitable for a broad audience.

Response:

This GNN model is a classic example of an epistemic foraging agent, where the agent navigates a maze with 4 locations (center, left arm, right arm, and cue location) to explore different regions of the map. The agent's behavior is driven by two types of actions: exploration-exploitation tradeoff between directionality and information gain.

The model consists of three main components:

1. **Location**: A 4x3 grid with 20 locations, each represented as a 2D array (represented in the code) with 8 values (representing the location's x-coordinate and y-coordinate). Each location is initialized to be randomly distributed across the grid.
2. **Location Observation**: A 16x4 matrix representing the location observation data. The position of each location is represented as a vector, which can be used for exploration or exploitation purposes. The coordinates are normalized by dividing them by their sum (represented in the code).
3. **Reward**: A 2D array representing reward values from one arm to another. Each element represents an action taken by the agent towards the corresponding arm. The reward is represented as a vector, which can be used for exploration or exploitation purposes. The coordinates are normalized by dividing them by their sum (represented in the code).
4. **Location Transition**: A 16x3 matrix representing the location transition data. Each element represents an action taken towards one arm and its corresponding reward value. The coordinates are normalized by dividing it by its sum (represented in the code) to ensure that all actions have equal probabilities of occurrence.
5. **Context**: A 2D array representing the context information, which can be used for exploration or exploitation purposes. Each element represents a location's x-coordinate and y-coordinate within the context matrix. The coordinates are normalized by dividing them by their sum (represented in the code).
6. **Action**: A 1x3 vector representing an action taken towards one arm. The coordinates are normalized by dividing it by its sum (represented in the code) to ensure that all actions have equal probabilities of occurrence.
7. **Location**: A 2D array representing the location observation data, which can be used for exploration or exploitation purposes. Each element represents a location's x-coordinate and y-coordinate within the context matrix. The coordinates are normalized by dividing it by its sum (represented in the code) to ensure that all actions have equal probabilities of occurrence.