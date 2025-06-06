# Multi-agent Trajectory Planning Results

Generated at: 2025-06-06 13:52:09

## Contents

### Animations
- `door_42.gif` - Door environment (seed 42)
- `door_123.gif` - Door environment (seed 123)
- `wall_42.gif` - Wall environment (seed 42)
- `wall_123.gif` - Wall environment (seed 123)
- `combined_42.gif` - Combined environment (seed 42)
- `combined_123.gif` - Combined environment (seed 123)

### Visualizations
- `control_signals.gif` - Control signals for each agent
- `control_magnitudes.png` - Plot of control signal magnitudes over time
- `obstacle_distance.png` - Heatmap of distances to obstacles
- `path_uncertainty.png` - Visualization of path uncertainties
- `path_visualization.png` - Static visualization of agent paths
- `convergence.png` - Convergence plot of the inference (may show placeholder if ELBO tracking unavailable)

### Environment Heatmaps
- `door_environment_heatmap.png` - Distance field visualization for door environment
- `wall_environment_heatmap.png` - Distance field visualization for wall environment
- `combined_environment_heatmap.png` - Distance field visualization for combined environment

### Data Files
- `paths.csv` - Raw path data (agent positions over time)
- `controls.csv` - Raw control signals (agent control inputs)
- `uncertainties.csv` - Path uncertainties (variance in agent positions)
- `experiment.log` - Detailed log of experiment execution

## Experiment Setup

The experiments demonstrate multi-agent trajectory planning in three environments:
1. **Door environment**: Two parallel walls with a gap between them
2. **Wall environment**: A single wall obstacle in the center
3. **Combined environment**: A combination of walls and obstacles

Each experiment is run with 4 agents that need to navigate from their starting positions
to their target positions while avoiding obstacles and other agents.

## Implementation Details

The trajectory planning is implemented using probabilistic inference with the RxInfer.jl framework.
The agents follow a linear dynamical model with control inputs, and constraints are enforced
through observations in the probabilistic model.

For more details, see the `DOCUMENTATION.md` file in the project root.
