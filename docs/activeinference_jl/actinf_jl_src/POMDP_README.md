# ActiveInference.jl POMDP Gridworld Simulation

A comprehensive POMDP (Partially Observable Markov Decision Process) gridworld simulation using ActiveInference.jl with all configuration variables at the top for easy modification and experimentation.

## Overview

This simulation implements a gridworld environment where an Active Inference agent navigates through a partially observable environment to reach a goal while avoiding obstacles. The agent uses belief updating and policy inference to make decisions under uncertainty.

## Files

- `ActiveInference_jl_POMDP.jl` - **Main simulation script** with all variables configured at top
- `test_pomdp_gridworld.jl` - **Test script** for verification with smaller configuration
- `POMDP_README.md` - This documentation file

## Key Features

### 🔧 **Fully Configurable**
All simulation parameters are defined as constants at the top of the script:
- Grid size and layout
- Start/goal positions and obstacles
- POMDP state space dimensions
- Active Inference agent parameters
- Reward structure and costs
- Observation model configuration

### 🧠 **Active Inference Implementation**
- Proper POMDP model construction with A, B, C, D, E matrices
- Belief state updating using `infer_states!()`
- Policy inference using `infer_policies!()`
- Action selection using `sample_action!()`
- Multi-step planning with configurable horizon

### 🌍 **Gridworld Environment**
- Configurable grid size (default: 5x5)
- Obstacle placement and collision detection
- Partial observability with observation noise
- Four observation types: empty, wall, goal, obstacle
- Four actions: up, down, left, right

### 📊 **Comprehensive Data Collection**
- Position traces over time
- Belief state evolution
- Action and observation sequences
- Reward accumulation
- Performance metrics and analysis

## Configuration Variables

### Gridworld Configuration
```julia
const GRID_SIZE = 5                    # 5x5 gridworld
const START_POSITION = [1, 1]          # Starting position [row, col]
const GOAL_POSITION = [5, 5]           # Goal position [row, col]
const OBSTACLE_POSITIONS = [           # Obstacle positions [row, col]
    [2, 2], [2, 3], [3, 2],           # Small obstacle cluster
    [4, 1], [4, 2]                    # Additional obstacles
]
```

### POMDP State Space
```julia
const N_STATES = [GRID_SIZE * GRID_SIZE]    # Total number of grid positions
const N_OBSERVATIONS = [4]                  # 4 observation types
const N_CONTROLS = [4]                      # 4 actions
const POLICY_LENGTH = 3                     # Planning horizon
```

### Active Inference Parameters
```julia
const ALPHA = 8.0                          # Precision parameter
const BETA = 1.0                           # Inverse temperature
const GAMMA = 1.0                          # Policy precision
const LAMBDA = 1.0                         # Learning rate
const OMEGA = 1.0                          # Evidence accumulation rate
```

### Simulation Settings
```julia
const N_SIMULATION_STEPS = 50              # Number of simulation steps
const OBSERVATION_NOISE = 0.1              # Probability of incorrect observations
const GOAL_REWARD = 10.0                   # Reward for reaching goal
const STEP_COST = -0.1                     # Cost per step
```

## How to Run

### Prerequisites

- Julia 1.6+ installed
- ActiveInference.jl package installed
- Internet connection (for package installation)

### Quick Start

1. Navigate to the directory:
   ```bash
   cd doc/activeinference_jl/actinf_jl_src
   ```

2. Run the main simulation:
   ```bash
   julia ActiveInference_jl_POMDP.jl
   ```

3. Or run the test version (smaller, faster):
   ```bash
   julia test_pomdp_gridworld.jl
   ```

### What Happens

1. **Environment Setup**: Creates gridworld with obstacles and goal
2. **POMDP Model Construction**: Builds A, B, C, D, E matrices
3. **Agent Initialization**: Creates Active Inference agent with specified parameters
4. **Simulation Execution**: Runs agent through environment with belief updating
5. **Data Collection**: Saves comprehensive traces and analysis
6. **Results Generation**: Creates organized output directory with all results

## Output Structure

```
pomdp_gridworld_outputs/gridworld_outputs_[timestamp]/
├── logs/
│   └── pomdp_gridworld.log              # Detailed execution log
├── models/
│   └── gridworld_model.json             # Model configuration
├── simulation_results/
│   └── gridworld_simulation.csv         # Main simulation results
├── data_traces/
│   ├── positions_trace.csv              # Position over time
│   ├── beliefs_trace.csv                # Belief state evolution
│   ├── actions_trace.csv                # Action sequence
│   ├── observations_trace.csv           # Observation sequence
│   └── rewards_trace.csv                # Reward accumulation
├── analysis/
│   └── simulation_analysis.txt          # Performance analysis
└── visualizations/                      # Generated plots (if enabled)
```

## Example Output

### Simulation Analysis
```
ActiveInference.jl POMDP Gridworld Simulation Analysis
============================================================
Generated: 2025-01-07T12:30:45.123

SIMULATION OVERVIEW:
  Total steps: 25
  Total reward: 7.5
  Mean reward per step: 0.3
  Goal reached: true
  Unique positions visited: 12
  Final position: [5, 5]

ACTION DISTRIBUTION:
  Up (Action 1): 8 times (32.0%)
  Down (Action 2): 6 times (24.0%)
  Left (Action 3): 5 times (20.0%)
  Right (Action 4): 6 times (24.0%)

OBSERVATION DISTRIBUTION:
  Empty (Obs 1): 15 times (60.0%)
  Wall (Obs 2): 5 times (20.0%)
  Goal (Obs 3): 1 times (4.0%)
  Obstacle (Obs 4): 4 times (16.0%)

PERFORMANCE METRICS:
  Efficiency: 0.48
  Exploration ratio: 0.48
  Steps to goal: 25
```

## Customization Examples

### Change Grid Size
```julia
const GRID_SIZE = 10                   # 10x10 gridworld
const N_STATES = [GRID_SIZE * GRID_SIZE]
```

### Modify Obstacles
```julia
const OBSTACLE_POSITIONS = [
    [3, 3], [3, 4], [4, 3],           # Different obstacle pattern
    [7, 7], [8, 8]                    # Additional obstacles
]
```

### Adjust Agent Parameters
```julia
const ALPHA = 16.0                     # Higher precision
const POLICY_LENGTH = 5                # Longer planning horizon
const OBSERVATION_NOISE = 0.05         # Lower observation noise
```

### Change Reward Structure
```julia
const GOAL_REWARD = 20.0               # Higher goal reward
const STEP_COST = -0.05                # Lower step cost
const OBSTACLE_COST = -2.0             # Higher obstacle penalty
```

## Scientific Validation

✅ **POMDP Correctness**: Proper state space, observation model, and transition dynamics  
✅ **Active Inference Implementation**: Correct belief updating and policy inference  
✅ **Partial Observability**: Realistic observation noise and uncertainty  
✅ **Reproducibility**: Deterministic behavior with fixed random seeds  
✅ **Data Integrity**: Comprehensive metadata and provenance tracking  
✅ **Performance Monitoring**: Resource usage and efficiency metrics  

## Integration with GNN Pipeline

This POMDP simulation is designed to integrate seamlessly with the broader GeneralizedNotationNotation (GNN) pipeline:

- **Consistent Output Structure**: Follows GNN standards for organized data directories
- **Comprehensive Logging**: Uses structured logging compatible with GNN pipeline tools
- **Modular Design**: Can be called as component within larger GNN workflows
- **Data Compatibility**: Generates outputs suitable for further GNN processing steps

## Troubleshooting

### Package Installation Issues
```julia
using Pkg
Pkg.add("ActiveInference")
Pkg.add("JSON")
```

### Julia Version Issues
Ensure Julia 1.6+:
```bash
julia --version
```

### Configuration Issues
- Check that `GRID_SIZE` is consistent with `N_STATES`
- Ensure `START_POSITION` and `GOAL_POSITION` are within grid bounds
- Verify `OBSTACLE_POSITIONS` don't block start or goal

### Performance Issues
- Reduce `N_SIMULATION_STEPS` for faster execution
- Set `VERBOSE_LOGGING = false` to reduce output
- Use smaller `GRID_SIZE` for testing

## License

This code follows the same license as the parent GNN project and ActiveInference.jl (MIT License). 