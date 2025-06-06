# GNN Example: RxInfer Multi-agent Trajectory Planning
# Format: Markdown representation of a Multi-agent Trajectory Planning model for RxInfer.jl
# Version: 1.0
# This file is machine-readable and represents the configuration for the RxInfer.jl multi-agent trajectory planning example.

## GNNSection
RxInferMultiAgentTrajectoryPlanning

## GNNVersionAndFlags
GNN v1

## ModelName
Multi-agent Trajectory Planning

## ModelAnnotation
This model represents a multi-agent trajectory planning scenario in RxInfer.jl.
It includes:
- State space model for agents moving in a 2D environment
- Obstacle avoidance constraints
- Goal-directed behavior
- Inter-agent collision avoidance
The model can be used to simulate trajectory planning in various environments with obstacles.

## StateSpaceBlock
# Model parameters
dt[1,type=float]               # Time step for the state space model
gamma[1,type=float]            # Constraint parameter for the Halfspace node
nr_steps[1,type=int]           # Number of time steps in the trajectory
nr_iterations[1,type=int]      # Number of inference iterations
nr_agents[1,type=int]          # Number of agents in the simulation
softmin_temperature[1,type=float] # Temperature parameter for the softmin function
intermediate_steps[1,type=int] # Intermediate results saving interval
save_intermediates[1,type=bool] # Whether to save intermediate results

# State space matrices
A[4,4,type=float]              # State transition matrix
B[4,2,type=float]              # Control input matrix
C[2,4,type=float]              # Observation matrix

# Prior distributions
initial_state_variance[1,type=float]    # Prior on initial state
control_variance[1,type=float]          # Prior on control inputs
goal_constraint_variance[1,type=float]  # Goal constraints variance
gamma_shape[1,type=float]               # Parameters for GammaShapeRate prior
gamma_scale_factor[1,type=float]        # Parameters for GammaShapeRate prior

# Visualization parameters
x_limits[2,type=float]            # Plot boundaries (x-axis)
y_limits[2,type=float]            # Plot boundaries (y-axis)
fps[1,type=int]                   # Animation frames per second
heatmap_resolution[1,type=int]    # Heatmap resolution
plot_width[1,type=int]            # Plot width
plot_height[1,type=int]           # Plot height
agent_alpha[1,type=float]         # Visualization alpha for agents
target_alpha[1,type=float]        # Visualization alpha for targets
color_palette[1,type=string]      # Color palette for visualization

# Environment definitions
door_obstacle_center_1[2,type=float]    # Door environment, obstacle 1 center
door_obstacle_size_1[2,type=float]      # Door environment, obstacle 1 size
door_obstacle_center_2[2,type=float]    # Door environment, obstacle 2 center
door_obstacle_size_2[2,type=float]      # Door environment, obstacle 2 size

wall_obstacle_center[2,type=float]      # Wall environment, obstacle center
wall_obstacle_size[2,type=float]        # Wall environment, obstacle size

combined_obstacle_center_1[2,type=float] # Combined environment, obstacle 1 center
combined_obstacle_size_1[2,type=float]   # Combined environment, obstacle 1 size
combined_obstacle_center_2[2,type=float] # Combined environment, obstacle 2 center
combined_obstacle_size_2[2,type=float]   # Combined environment, obstacle 2 size
combined_obstacle_center_3[2,type=float] # Combined environment, obstacle 3 center
combined_obstacle_size_3[2,type=float]   # Combined environment, obstacle 3 size

# Agent configurations
agent1_id[1,type=int]                   # Agent 1 ID
agent1_radius[1,type=float]             # Agent 1 radius
agent1_initial_position[2,type=float]   # Agent 1 initial position
agent1_target_position[2,type=float]    # Agent 1 target position

agent2_id[1,type=int]                   # Agent 2 ID
agent2_radius[1,type=float]             # Agent 2 radius
agent2_initial_position[2,type=float]   # Agent 2 initial position
agent2_target_position[2,type=float]    # Agent 2 target position

agent3_id[1,type=int]                   # Agent 3 ID
agent3_radius[1,type=float]             # Agent 3 radius
agent3_initial_position[2,type=float]   # Agent 3 initial position
agent3_target_position[2,type=float]    # Agent 3 target position

agent4_id[1,type=int]                   # Agent 4 ID
agent4_radius[1,type=float]             # Agent 4 radius
agent4_initial_position[2,type=float]   # Agent 4 initial position
agent4_target_position[2,type=float]    # Agent 4 target position

# Experiment configurations
experiment_seeds[2,type=int]            # Random seeds for reproducibility
results_dir[1,type=string]              # Base directory for results
animation_template[1,type=string]       # Filename template for animations
control_vis_filename[1,type=string]     # Filename for control visualization
obstacle_distance_filename[1,type=string] # Filename for obstacle distance plot
path_uncertainty_filename[1,type=string]  # Filename for path uncertainty plot
convergence_filename[1,type=string]       # Filename for convergence plot

## Connections
# Model parameters
dt > A
(A, B, C) > state_space_model

# Agent trajectories
(state_space_model, initial_state_variance, control_variance) > agent_trajectories

# Goal constraints
(agent_trajectories, goal_constraint_variance) > goal_directed_behavior

# Obstacle avoidance
(agent_trajectories, gamma, gamma_shape, gamma_scale_factor) > obstacle_avoidance

# Collision avoidance
(agent_trajectories, nr_agents) > collision_avoidance

# Complete planning system
(goal_directed_behavior, obstacle_avoidance, collision_avoidance) > planning_system

## InitialParameterization
# Model parameters
dt=1.0
gamma=1.0
nr_steps=40
nr_iterations=350
nr_agents=4
softmin_temperature=10.0
intermediate_steps=10
save_intermediates=false

# State space matrices
# A = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
A={(1.0, 1.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0), (0.0, 0.0, 0.0, 1.0)}

# B = [0 0; dt 0; 0 0; 0 dt]
B={(0.0, 0.0), (1.0, 0.0), (0.0, 0.0), (0.0, 1.0)}

# C = [1 0 0 0; 0 0 1 0]
C={(1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0)}

# Prior distributions
initial_state_variance=100.0
control_variance=0.1
goal_constraint_variance=0.00001
gamma_shape=1.5
gamma_scale_factor=0.5

# Visualization parameters
x_limits={(-20, 20)}
y_limits={(-20, 20)}
fps=15
heatmap_resolution=100
plot_width=800
plot_height=400
agent_alpha=1.0
target_alpha=0.2
color_palette="tab10"

# Environment definitions
door_obstacle_center_1={(-40.0, 0.0)}
door_obstacle_size_1={(70.0, 5.0)}
door_obstacle_center_2={(40.0, 0.0)}
door_obstacle_size_2={(70.0, 5.0)}

wall_obstacle_center={(0.0, 0.0)}
wall_obstacle_size={(10.0, 5.0)}

combined_obstacle_center_1={(-50.0, 0.0)}
combined_obstacle_size_1={(70.0, 2.0)}
combined_obstacle_center_2={(50.0, 0.0)}
combined_obstacle_size_2={(70.0, 2.0)}
combined_obstacle_center_3={(5.0, -1.0)}
combined_obstacle_size_3={(3.0, 10.0)}

# Agent configurations
agent1_id=1
agent1_radius=2.5
agent1_initial_position={(-4.0, 10.0)}
agent1_target_position={(-10.0, -10.0)}

agent2_id=2
agent2_radius=1.5
agent2_initial_position={(-10.0, 5.0)}
agent2_target_position={(10.0, -15.0)}

agent3_id=3
agent3_radius=1.0
agent3_initial_position={(-15.0, -10.0)}
agent3_target_position={(10.0, 10.0)}

agent4_id=4
agent4_radius=2.5
agent4_initial_position={(0.0, -10.0)}
agent4_target_position={(-10.0, 15.0)}

# Experiment configurations
experiment_seeds={(42, 123)}
results_dir="results"
animation_template="{environment}_{seed}.gif"
control_vis_filename="control_signals.gif"
obstacle_distance_filename="obstacle_distance.png"
path_uncertainty_filename="path_uncertainty.png"
convergence_filename="convergence.png"

## Equations
# State space model:
# x_{t+1} = A * x_t + B * u_t + w_t,  w_t ~ N(0, control_variance)
# y_t = C * x_t + v_t,                v_t ~ N(0, observation_variance)
#
# Obstacle avoidance constraint:
# p(x_t | obstacle) ~ N(d(x_t, obstacle), gamma)
# where d(x_t, obstacle) is the distance from position x_t to the nearest obstacle
#
# Goal constraint:
# p(x_T | goal) ~ N(goal, goal_constraint_variance)
# where x_T is the final position
#
# Collision avoidance constraint:
# p(x_i, x_j) ~ N(||x_i - x_j|| - (r_i + r_j), gamma)
# where x_i, x_j are positions of agents i and j, r_i, r_j are their radii

## Time
Dynamic
DiscreteTime
ModelTimeHorizon=nr_steps

## ActInfOntologyAnnotation
dt=TimeStep
gamma=ConstraintParameter
nr_steps=TrajectoryLength
nr_iterations=InferenceIterations
nr_agents=NumberOfAgents
softmin_temperature=SoftminTemperature
A=StateTransitionMatrix
B=ControlInputMatrix
C=ObservationMatrix
initial_state_variance=InitialStateVariance
control_variance=ControlVariance
goal_constraint_variance=GoalConstraintVariance

## ModelParameters
nr_agents=4
nr_steps=40
nr_iterations=350

## Footer
Multi-agent Trajectory Planning - GNN Representation for RxInfer.jl

## Signature
Creator: AI Assistant for GNN
Date: 2024-07-27
Status: Example for RxInfer.jl multi-agent trajectory planning 