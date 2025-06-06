# Multi-agent Trajectory Planning Visualizations

This document provides a comprehensive guide to the visualization functions available in the Multi-agent Trajectory Planning project.

## Table of Contents

- [Overview](#overview)
- [Animation Functions](#animation-functions)
- [Static Visualization Functions](#static-visualization-functions)
- [Heatmap Functions](#heatmap-functions)
- [Data Export Functions](#data-export-functions)
- [Visualization Interpretation](#visualization-interpretation)
- [Customization Options](#customization-options)
- [Using the Visualization Tools](#using-the-visualization-tools)
- [Data Format Requirements](#data-format-requirements)

## Overview

The visualization system is built around the Plots.jl package and provides a variety of functions for visualizing:
- Agent trajectories and dynamics
- Environmental constraints
- Inference convergence
- Uncertainty in planned paths
- Control signals

All visualization functions can be customized through their parameters, and most accept a `filename` parameter to save the output to disk.

## Animation Functions

### animate_paths

```julia
animate_paths(environment, agents, paths; 
              filename = "result.gif", 
              fps = 15, 
              x_limits = (-20, 20), 
              y_limits = (-20, 20), 
              plot_size = (800, 400), 
              show_targets = true,
              path_alpha = 0.8)
```

Creates an animation showing the movement of agents through the environment over time.

**Parameters:**
- `environment`: The environment containing obstacles
- `agents`: List of agent objects
- `paths`: Matrix of agent positions over time, dimensions: (nr_agents, nr_steps)
- `filename`: Output filename for the GIF (default: "result.gif")
- `fps`: Frames per second for the animation (default: 15)
- `x_limits`: Tuple with x-axis limits (default: (-20, 20))
- `y_limits`: Tuple with y-axis limits (default: (-20, 20))
- `plot_size`: Tuple with plot width and height (default: (800, 400))
- `show_targets`: Whether to show target positions (default: true)
- `path_alpha`: Alpha value for path lines (default: 0.8)

**Output:**
- GIF animation showing agent movement through the environment
- Example: `door_42.gif`, `wall_123.gif`, `combined_42.gif`

**Interpretation:**
- Each colored circle represents an agent (size proportional to radius)
- Dashed lines show the path taken so far
- Transparent circles at the end show target positions
- Gray rectangles represent obstacles

### animate_control_signals

```julia
animate_control_signals(environment, paths, controls;
                      filename = "control_signals.gif",
                      fps = 10,
                      control_scale = 5.0)
```

Creates an animation showing the control signals (acceleration vectors) for each agent over time.

**Parameters:**
- `environment`: The environment containing obstacles
- `paths`: Matrix of agent positions over time
- `controls`: Matrix of control inputs over time, dimensions: (nr_agents, nr_steps)
- `filename`: Output filename for the GIF (default: "control_signals.gif")
- `fps`: Frames per second for the animation (default: 10)
- `control_scale`: Scale factor for control vectors (default: 5.0)

**Output:**
- GIF animation showing agent positions with control vectors
- Example: `control_signals.gif`

**Interpretation:**
- Each colored circle represents an agent
- Arrows indicate the direction and magnitude of control inputs (acceleration)
- Longer arrows indicate stronger control inputs

### animate_path_uncertainties

```julia
animate_path_uncertainties(environment, agents, paths, uncertainties;
                         filename = "path_uncertainty.gif",
                         fps = 10,
                         uncertainty_scale = 3.0,
                         show_full_path = false)
```

Creates an animation showing the uncertainty in agent paths over time.

**Parameters:**
- `environment`: The environment containing obstacles
- `agents`: List of agent objects
- `paths`: Matrix of agent positions over time
- `uncertainties`: Matrix of position variances over time
- `filename`: Output filename (default: "path_uncertainties.gif")
- `fps`: Frames per second for the animation (default: 10)
- `uncertainty_scale`: Scaling factor for uncertainty visualization (default: 3.0)
- `show_full_path`: Whether to show the complete path at each frame (default: false)

**Output:**
- GIF animation showing agent paths with uncertainty visualized
- Example: `path_uncertainties.gif`

**Interpretation:**
- Each colored circle represents an agent
- Transparent circles around agents show position uncertainty
- Larger transparent circles indicate higher uncertainty

### animate_combined_visualization

```julia
animate_combined_visualization(environment, agents, paths, uncertainties, controls;
                             filename = "combined_visualization.gif",
                             fps = 10,
                             uncertainty_scale = 2.0,
                             control_scale = 5.0,
                             show_trail = true,
                             trail_length = 10)
```

Creates a combined animation showing paths, controls, and uncertainties.

**Parameters:**
- `environment`: The environment containing obstacles
- `agents`: List of agent objects
- `paths`: Matrix of agent positions over time
- `uncertainties`: Matrix of position variances
- `controls`: Matrix of control inputs
- `filename`: Output filename (default: "combined_visualization.gif")
- `fps`: Frames per second for the animation (default: 10)
- `uncertainty_scale`: Scale factor for uncertainty ellipses (default: 2.0)
- `control_scale`: Scale factor for control vectors (default: 5.0)
- `show_trail`: Whether to show a trail of recent positions (default: true)
- `trail_length`: Number of recent positions to show (default: 10)

**Output:**
- GIF animation with multiple panels showing different aspects of the solution
- Example: `combined_visualization.gif`

**Interpretation:**
- Top panel: Agent paths through environment
- Middle panel: Control signals
- Bottom panel: Path uncertainties

## Static Visualization Functions

### plot_environment

```julia
plot_environment(environment; 
                x_limits = (-20, 20), 
                y_limits = (-20, 20), 
                plot_size = (800, 400))
```

Creates a static plot of the environment showing obstacles.

**Parameters:**
- `environment`: The environment containing obstacles
- `x_limits`, `y_limits`, `plot_size`: Similar to animate_paths

**Output:**
- Plot object showing the environment with obstacles

**Interpretation:**
- Gray rectangles represent obstacles
- White space is navigable area

### plot_elbo_convergence

```julia
plot_elbo_convergence(elbo_values; filename = "convergence.png")
```

Creates a plot showing the ELBO convergence during inference.

**Parameters:**
- `elbo_values`: Vector of ELBO values from inference iterations
- `filename`: Output filename (default: "convergence.png")

**Output:**
- PNG image showing convergence plot
- Example: `convergence.png`

**Interpretation:**
- X-axis: Inference iteration
- Y-axis: ELBO value
- Blue line: Raw ELBO values
- Red line: Smoothed trend (if enough iterations)
- Increasing ELBO indicates improving inference quality

### plot_path_uncertainties

```julia
plot_path_uncertainties(environment, agents, paths, path_vars; 
                       filename = "path_uncertainty.png",
                       x_limits = (-20, 20), 
                       y_limits = (-20, 20), 
                       plot_size = (800, 400),
                       uncertainty_scale = 3.0,
                       show_step = 5)
```

Creates a static visualization of path uncertainties.

**Parameters:**
- `environment`: The environment containing obstacles
- `agents`: List of agent objects
- `paths`: Matrix of agent positions over time
- `path_vars`: Matrix of variances in agent positions
- `filename`: Output filename (default: "path_uncertainty.png")
- `uncertainty_scale`: Scale factor for the uncertainty ellipses (default: 3.0)
- `show_step`: Which steps to show uncertainties for (default: 5):
  - `:all`: Show all steps
  - `:endpoints`: Show only start and end
  - Integer: Show every n-th step
  - Vector: Show specific steps

**Output:**
- PNG image showing path uncertainties
- Example: `path_uncertainty.png`

**Interpretation:**
- Similar to animate_path_uncertainties, but shows all time steps at once
- Overlapping uncertainty circles indicate regions with higher uncertainty

### plot_control_signals

```julia
plot_control_signals(agents, controls; 
                    filename = "control_signals.png",
                    plot_size = (800, 400),
                    component = :both)
```

Creates a static visualization of control signals over time.

**Parameters:**
- `agents`: List of agent objects
- `controls`: Matrix of control inputs
- `filename`: Output filename (default: "control_signals.png")
- `plot_size`: Plot dimensions
- `component`: Which control component(s) to show:
  - `:both`: Show both X and Y components in a 2x1 layout
  - `:x`: Show only X component
  - `:y`: Show only Y component
  - `:magnitude`: Show the magnitude of the control signal (√(x² + y²))

**Output:**
- PNG image showing control signals
- Example: `control_signals.png`

**Interpretation:**
- X-axis: Time step
- Y-axis: Control signal value
- Each colored line represents a different agent's control signal component

### plot_control_magnitudes

```julia
plot_control_magnitudes(agents, controls; 
                       filename = "control_magnitudes.png",
                       plot_size = (800, 400))
```

Creates a plot showing the magnitude of control signals over time.

**Parameters:**
- `agents`: List of agent objects
- `controls`: Matrix of control inputs
- `filename`: Output filename (default: "control_magnitudes.png")
- `plot_size`: Plot dimensions

**Output:**
- PNG image showing control magnitudes
- Example: `control_magnitudes.png`

**Interpretation:**
- X-axis: Time step
- Y-axis: Control magnitude (norm of control vector)
- Each colored line represents a different agent
- Higher values indicate stronger control inputs (more aggressive acceleration)

### plot_path_visualization

```julia
plot_path_visualization(environment, agents, paths; 
                       filename = "path_visualization.png",
                       x_limits = (-20, 20), 
                       y_limits = (-20, 20), 
                       plot_size = (800, 400))
```

Creates a static visualization of agent paths.

**Parameters:**
- `environment`: The environment containing obstacles
- `agents`: List of agent objects
- `paths`: Matrix of agent positions over time
- `filename`: Output filename (default: "path_visualization.png")
- Other parameters similar to previous functions

**Output:**
- PNG image showing agent paths
- Example: `path_visualization.png`

**Interpretation:**
- Gray rectangles: Obstacles
- Colored lines: Agent paths
- Squares: Starting positions
- Stars: Target positions
- Each color represents a different agent

### plot_agent_path_3d

```julia
plot_agent_path_3d(environment, agents, paths; 
                  filename = "path_3d.png",
                  plot_size = (800, 600),
                  time_height = 10.0,
                  agent_idx = nil,
                  with_time = true)
```

Creates a 3D visualization of agent paths with time as the third dimension.

**Parameters:**
- `environment`: The environment containing obstacles
- `agents`: List of agent objects
- `paths`: Matrix of agent positions over time
- `filename`: Output filename (default: "path_3d.png")
- `plot_size`: Plot dimensions
- `time_height`: Height of the time dimension (default: 10.0)
- `agent_idx`: Index of the agent to visualize (if nil, show all agents)
- `with_time`: If true, use time as z-axis; if false, use uncertainty magnitude

**Output:**
- PNG image showing 3D visualization of paths
- Example: `path_3d.png`

**Interpretation:**
- X and Y axes: Spatial coordinates
- Z-axis: Time dimension or uncertainty magnitude
- Colored curves: Agent trajectories through space-time
- Flat gray shapes at bottom: Obstacles

## Heatmap Functions

### visualize_obstacle_distance

```julia
visualize_obstacle_distance(environment; 
                           filename = "obstacle_distance.png",
                           resolution = 100,
                           x_limits = (-20, 20), 
                           y_limits = (-20, 20), 
                           plot_size = (800, 400))
```

Creates a heatmap showing the distance to obstacles throughout the environment.

**Parameters:**
- `environment`: The environment containing obstacles
- `filename`: Output filename (default: "obstacle_distance.png")
- `resolution`: Resolution of the heatmap (default: 100)
- Other parameters similar to previous functions

**Output:**
- PNG image showing distance heatmap
- Example: `obstacle_distance.png`, `door_environment_heatmap.png`

**Interpretation:**
- Color scale: Distance to nearest obstacle
- Dark areas: Close to or inside obstacles
- Bright areas: Far from obstacles
- Gray rectangles: Obstacle outlines

### create_path_heatmap

```julia
create_path_heatmap(environment, agents, paths; 
                   filename = "path_heatmap.png",
                   resolution = 100,
                   x_limits = (-20, 20), 
                   y_limits = (-20, 20), 
                   plot_size = (800, 400),
                   colormap = :thermal)
```

Creates a heatmap showing the density of agent paths.

**Parameters:**
- `environment`: The environment containing obstacles
- `agents`: List of agent objects
- `paths`: Matrix of agent positions over time
- `filename`: Output filename (default: "path_heatmap.png")
- `resolution`: Resolution of the heatmap (default: 100)
- `colormap`: Colormap for the heatmap (default: :thermal)
- Other parameters similar to previous functions

**Output:**
- PNG image showing path density heatmap
- Example: `path_heatmap.png`

**Interpretation:**
- Color scale: Density of agent paths
- Dark areas: Rarely traversed
- Bright areas: Frequently traversed
- Gray rectangles: Obstacle outlines

## Data Export Functions

### save_path_data

```julia
save_path_data(paths, output_file)
```

Saves agent path data to a CSV file.

**Parameters:**
- `paths`: Matrix of agent positions over time
- `output_file`: Output CSV filename

**Output:**
- CSV file with columns: agent_id, time_step, x_position, y_position
- Example: `paths.csv`

**Format:**
```csv
agent,step,x,y
1,1,-4.00000,10.00000
1,2,-4.02986,10.02189
...
```

### save_control_data

```julia
save_control_data(controls, output_file)
```

Saves agent control data to a CSV file.

**Parameters:**
- `controls`: Matrix of control inputs over time
- `output_file`: Output CSV filename

**Output:**
- CSV file with columns: agent_id, time_step, x_control, y_control
- Example: `controls.csv`

**Format:**
```csv
agent,step,x_control,y_control
1,1,-0.02985,0.02188
1,2,-0.03390,0.00257
...
```

### save_uncertainty_data

```julia
save_uncertainty_data(uncertainties, output_file)
```

Saves path uncertainty data to a CSV file.

**Parameters:**
- `uncertainties`: Matrix of variances in agent positions
- `output_file`: Output CSV filename

**Output:**
- CSV file with columns: agent_id, time_step, x_variance, y_variance
- Example: `uncertainties.csv`

**Format:**
```csv
agent,step,x_variance,y_variance
1,1,0.00002,0.00002
1,2,0.05892,0.06050
...
```

## Visualization Interpretation

### Path Animations

Path animations show the motion of agents through the environment over time. Key aspects to observe:

1. **Obstacle Avoidance**: Agents should maintain a distance from obstacles based on their radius
2. **Collision Avoidance**: Agents should maintain separation from each other
3. **Path Smoothness**: Paths should be smooth and efficient, not jerky or unnecessarily complex
4. **Goal Reaching**: Agents should successfully reach their target positions

The color coding consistently identifies agents across different visualizations.

### Uncertainty Visualization

Uncertainty visualizations show the variance in agent positions. Key aspects to observe:

1. **Initial Uncertainty**: Usually high at the beginning of inference
2. **Obstacle Proximity**: Uncertainty typically increases near obstacles
3. **Goal Uncertainty**: Usually low at the start and end positions (constrained by goals)
4. **Crossing Regions**: Areas where paths cross often show higher uncertainty

Larger transparent circles indicate higher uncertainty in the agent's position.

### Control Signal Visualization

Control signal visualizations show the acceleration inputs required to follow the planned paths. Key aspects to observe:

1. **Control Magnitude**: The size of control inputs indicates how aggressive the motion is
2. **Direction Changes**: Sudden changes in control direction indicate reactive avoidance
3. **Smooth Controls**: Well-planned paths typically have smooth, low-magnitude controls
4. **Obstacle Effects**: Control signals often spike near obstacles for avoidance

Control signals are represented either as arrows (in animations) or as line plots showing magnitude over time.

### Distance Heatmaps

Distance heatmaps show the distance field around obstacles. Key aspects to observe:

1. **Gradient**: The gradient of the distance field guides agent motion
2. **Bottlenecks**: Narrow passages between obstacles appear as channels in the heatmap
3. **Optimal Paths**: Agents typically follow paths along higher-valued regions
4. **Constraint Effects**: The Halfspace node pushes agents toward higher-valued regions

The color scale in heatmaps represents distance: darker areas are closer to obstacles, brighter areas are farther away.

### Convergence Plots

Convergence plots show the progression of the ELBO (Evidence Lower Bound) during inference. Key aspects to observe:

1. **Monotonic Increase**: ELBO should generally increase during inference
2. **Plateaus**: Flat regions indicate convergence or local optima
3. **Jumps**: Sudden increases may indicate escaping local optima
4. **Final Value**: Higher final ELBO values generally indicate better solutions

The smoothed trend line (red) helps visualize the overall convergence pattern by filtering out noise.

## Customization Options

### Color Schemes

Most visualization functions use the Plots.jl `:tab10` color palette by default, which provides distinct colors for different agents. Alternative palettes include:

- `:viridis`: For heatmaps (perceptually uniform)
- `:inferno`: Alternative for heatmaps (higher contrast)
- `:plasma`: Another alternative for heatmaps
- `:rainbow`: Traditional rainbow palette (not recommended for scientific visualization)

### Plot Sizes and Resolutions

Plot size and resolution can be customized through parameters:

- `plot_size`: Tuple with width and height in pixels
- `resolution`: For heatmaps, controls the granularity of the grid

Recommended settings:
- Standard plots: (800, 400) for 2:1 aspect ratio
- Square plots: (600, 600) for 1:1 aspect ratio
- High-resolution: (1200, 600) for presentations or publications
- Heatmaps: resolution=200 for detailed visualization

### Animation Parameters

Animation parameters control the smoothness and quality of animated visualizations:

- `fps`: Frames per second (higher values create smoother but faster animations)
- `path_alpha`: Transparency of path lines (lower values make overlapping paths more visible)
- `show_targets`: Whether to show target positions in animations

Recommended settings:
- `fps=15`: Good balance between smoothness and animation speed
- `fps=30`: Smoother animation but faster playback
- `path_alpha=0.8`: Good visibility of paths
- `path_alpha=0.5`: Better for visualizing overlapping paths

## Using the Visualization Tools

### For Existing Results

The `visualize_results.jl` script can be used to generate all visualizations for an existing result directory:

```bash
julia visualize_results.jl results/2025-06-04_13-47-58
```

This will create a `visualizations` subdirectory containing all static and animated visualizations, along with a README.md file explaining each visualization.

### In Custom Scripts

You can also use the visualization functions directly in your own scripts:

```julia
using Plots
include("Environment.jl")
include("Visualizations.jl")

# Load your data
# ...

# Create visualizations
plot_path_uncertainties(agents, paths, uncertainties, filename = "path_uncertainty.png")
plot_control_signals(controls, filename = "control_signals.png")
# ...
```

## Data Format Requirements

The visualization functions expect data in the following formats:

- `paths`: Matrix of agent positions over time, dimensions: (nr_agents, nr_steps), each element is a tuple (x, y)
- `controls`: Matrix of control signals, dimensions: (nr_agents, nr_steps), each element is a tuple (x, y)
- `uncertainties`: Matrix of uncertainty values, dimensions: (nr_agents, nr_steps), each element is a tuple (unc_x, unc_y)
- `agents`: Vector of Agent objects with properties: initial_position, target_position, radius, id
- `environment`: Environment object containing obstacles

### Example of Loading CSV Data

```julia
# Load paths data
paths_data = CSV.read("paths.csv", header=false)
nr_agents = length(unique(paths_data[!, 1]))
nr_steps = div(size(paths_data, 1), nr_agents)

# Create paths matrix
paths = Matrix{Tuple{Float64, Float64}}(undef, nr_agents, nr_steps)
for i in 1:size(paths_data, 1)
    agent_idx = paths_data[i, 1]
    step_idx = paths_data[i, 2]
    x = paths_data[i, 3]
    y = paths_data[i, 4]
    paths[agent_idx, step_idx] = (x, y)
end

# Similarly for controls and uncertainties
# ...
``` 