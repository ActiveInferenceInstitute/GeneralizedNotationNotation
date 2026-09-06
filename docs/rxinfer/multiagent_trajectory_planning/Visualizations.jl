module Visualizations

using Plots
using ..Environment

export plot_marker_at_position!, plot_agent_naive_plan!, animate_paths
# Add new exports for functions moved from Environment.jl
export plot_rectangle!, plot_environment!, plot_environment
# Add export for the new ELBO convergence function
export plot_elbo_convergence
# Add exports for new visualization functions
export plot_path_uncertainties, animate_path_uncertainties
export plot_control_signals, animate_control_signals
export plot_combined_visualization, animate_combined_visualization
export plot_agent_path_3d, create_path_heatmap

"""
    plot_marker_at_position!(p, radius, position; color="red", markersize=10.0, alpha=1.0, label="")

Plot a circular marker at the specified position with the given radius.

# Arguments
- `p`: The plot to add the marker to
- `radius`: Radius of the circular marker
- `position`: A tuple (x, y) representing the center position
- `color`: Color of the marker (default: "red")
- `markersize`: Size of the marker (default: 10.0)
- `alpha`: Transparency of the marker (default: 1.0)
- `label`: Label for the marker in the plot legend (default: "")

# Returns
- The updated plot object
"""
function plot_marker_at_position!(p, radius, position; color="red", markersize=10.0, alpha=1.0, label="")
    # Draw the agent as a circle with the given radius
    θ = range(0, 2π, 100)
    
    x_coords = position[1] .+ radius .* cos.(θ)
    y_coords = position[2] .+ radius .* sin.(θ)
    
    plot!(p, Shape(x_coords, y_coords); color=color, label=label, alpha=alpha)
    return p
end

"""
    plot_agent_naive_plan!(p, agent; color = "blue")

Plot an agent's naive plan (straight line from initial to target position).

# Arguments
- `p`: The plot to add the plan to
- `agent`: The agent object containing initial and target positions
- `color`: Color to use for the agent (default: "blue")

# Returns
- The updated plot object
"""
function plot_agent_naive_plan!(p, agent; color = "blue")
    plot_marker_at_position!(p, agent.radius, agent.initial_position, color = color)
    plot_marker_at_position!(p, agent.radius, agent.target_position, color = color, alpha = 0.1)
    quiver!(p, [ agent.initial_position[1] ], [ agent.initial_position[2] ], 
            quiver = ([ agent.target_position[1] - agent.initial_position[1] ], 
                     [ agent.target_position[2] -  agent.initial_position[2] ]))
end

"""
    plot_agent_plans!(p, agents)

Plot naive plans for multiple agents.

# Arguments
- `p`: The plot to add the plans to
- `agents`: List of agent objects

# Returns
- The updated plot object
"""
function plot_agent_plans!(p, agents)
    colors = Plots.palette(:tab10)
    
    for (k, agent) in enumerate(agents)
        plot_agent_naive_plan!(p, agent, color = colors[k])
    end
    
    return p
end

"""
    animate_paths(environment, agents, paths; filename = "result.gif", fps = 15, x_limits = (-20, 20), y_limits = (-20, 20), plot_size = (800, 400), show_targets = true, path_alpha = 0.8)

Create an animation of agent paths through the environment.

# Arguments
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

# Returns
- `nothing`
"""
function animate_paths(environment, agents, paths; 
                      filename = "result.gif", 
                      fps = 15, 
                      x_limits = (-20, 20), 
                      y_limits = (-20, 20), 
                      plot_size = (800, 400), 
                      show_targets = true,
                      path_alpha = 0.8)
    nr_agents, nr_steps = size(paths)
    colors = Plots.palette(:tab10)

    println("Generating animation frames...")
    animation = @animate for t in 1:nr_steps
        frame = plot_environment(environment, 
                                x_limits = x_limits, 
                                y_limits = y_limits, 
                                plot_size = plot_size)
    
        for k in 1:nr_agents
            position = paths[k, t]          
            path = paths[k, 1:t]
            
            plot_marker_at_position!(frame, agents[k].radius, position, color = colors[k])
            
            if show_targets
                plot_marker_at_position!(frame, agents[k].radius, agents[k].target_position, color = colors[k], alpha = 0.2)
            end
            
            plot!(frame, getindex.(path, 1), getindex.(path, 2); 
                 linestyle = :dash, 
                 label = "", 
                 color = colors[k], 
                 alpha = path_alpha)
        end

        frame
    end

    # Save the animation
    println("Saving animation to $filename...")
    gif(animation, filename, fps=fps, show_msg = false)
    println("Animation saved successfully.")
    
    return nothing
end

"""
    plot_elbo_convergence(elbo_values; filename = "convergence.png")

Create a plot showing the ELBO convergence during inference.

# Arguments
- `elbo_values`: Vector of ELBO values from inference iterations
- `filename`: Output filename for the plot (default: "convergence.png")

# Returns
- The plot object
"""
function plot_elbo_convergence(elbo_values; filename = "convergence.png")
    p = plot(elbo_values, 
             xlabel = "Iteration", 
             ylabel = "ELBO", 
             title = "Convergence of Inference",
             legend = false, 
             linewidth = 2,
             grid = true,
             size = (800, 400))
    
    # Add a smoothed trend line
    if length(elbo_values) > 10
        window_size = max(5, div(length(elbo_values), 20))
        smoothed = movmean(elbo_values, window_size)
        plot!(p, smoothed, linewidth = 3, color = :red, alpha = 0.7, label = "Moving Average")
    end
    
    # Save the plot
    println("Saving convergence plot to $filename...")
    savefig(p, filename)
    println("Convergence plot saved successfully.")
    
    return p
end

# Helper function for moving average
function movmean(arr, window_size)
    n = length(arr)
    result = similar(arr)
    
    for i in 1:n
        start_idx = max(1, i - div(window_size, 2))
        end_idx = min(n, i + div(window_size, 2))
        result[i] = mean(arr[start_idx:end_idx])
    end
    
    return result
end

# Functions moved from Environment.jl

"""
    plot_rectangle!(p, rect::Rectangle)

Plot a rectangle on an existing plot.

# Arguments
- `p`: The plot to add the rectangle to
- `rect`: Rectangle object with center and size properties

# Returns
- The updated plot object
"""
function plot_rectangle!(p, rect::Rectangle)
    # Calculate the x-coordinates of the four corners
    x_coords = rect.center[1] .+ rect.size[1]/2 * [-1, 1, 1, -1, -1]
    # Calculate the y-coordinates of the four corners
    y_coords = rect.center[2] .+ rect.size[2]/2 * [-1, -1, 1, 1, -1]
    
    # Plot the rectangle with a black fill
    plot!(p, Shape(x_coords, y_coords), 
          label = "", 
          color = :black, 
          alpha = 0.5,
          linewidth = 1.5,
          fillalpha = 0.3)
end

"""
    plot_environment!(p, env::Environment)

Add all obstacles from an environment to an existing plot.

# Arguments
- `p`: The plot to add the environment obstacles to
- `env`: Environment object containing obstacles

# Returns
- The updated plot object
"""
function plot_environment!(p, env::Environment)
    for obstacle in env.obstacles
        plot_rectangle!(p, obstacle)
    end
    return p
end

"""
    plot_environment(env::Environment; x_limits=(-20, 20), y_limits=(-20, 20), plot_size=(800, 400))

Create a new plot showing the environment.

# Arguments
- `env`: Environment object containing obstacles
- `x_limits`: Tuple with x-axis limits (default: (-20, 20))
- `y_limits`: Tuple with y-axis limits (default: (-20, 20))
- `plot_size`: Tuple with plot width and height (default: (800, 400))

# Returns
- A new plot showing the environment
"""
function plot_environment(env::Environment; x_limits=(-20, 20), y_limits=(-20, 20), plot_size=(800, 400))
    p = Plots.plot(size = plot_size, xlims = x_limits, ylims = y_limits, aspect_ratio = :equal)
    plot_environment!(p, env)
    return p
end

"""
    plot_path_uncertainties(agents, paths, uncertainties; 
                          filename = "path_uncertainty.png",
                          x_limits = (-20, 20), 
                          y_limits = (-20, 20), 
                          plot_size = (800, 600))

Create a visualization of agent paths with uncertainty ellipses.

# Arguments
- `agents`: List of agent objects
- `paths`: Matrix of agent positions over time, dimensions: (nr_agents, nr_steps)
- `uncertainties`: Matrix of uncertainty values (x, y) for each agent at each step
- `filename`: Output filename for the plot (default: "path_uncertainty.png")
- `x_limits`: Tuple with x-axis limits (default: (-20, 20))
- `y_limits`: Tuple with y-axis limits (default: (-20, 20))
- `plot_size`: Tuple with plot width and height (default: (800, 600))

# Returns
- The plot object
"""
function plot_path_uncertainties(agents, paths, uncertainties; 
                               filename = "path_uncertainty.png",
                               x_limits = (-20, 20), 
                               y_limits = (-20, 20), 
                               plot_size = (800, 600),
                               uncertainty_scale = 3.0,
                               show_step = :all)
    
    nr_agents, nr_steps = size(paths)
    colors = Plots.palette(:tab10)
    
    p = plot(size = plot_size, 
             xlims = x_limits, 
             ylims = y_limits, 
             aspect_ratio = :equal,
             title = "Agent Paths with Uncertainty",
             xlabel = "X", 
             ylabel = "Y",
             legend = true)
    
    # Determine which steps to show uncertainties for
    if show_step == :all
        steps_to_show = 1:nr_steps
    elseif show_step == :endpoints
        steps_to_show = [1, nr_steps]
    elseif show_step isa Integer
        steps_to_show = 1:show_step:nr_steps
    else
        steps_to_show = show_step
    end
    
    for k in 1:nr_agents
        agent_path = paths[k, :]
        
        # Plot the path
        plot!(p, getindex.(agent_path, 1), getindex.(agent_path, 2), 
              color = colors[k], 
              linewidth = 2, 
              alpha = 0.7,
              label = "Agent $k Path")
        
        # Plot uncertainty ellipses at selected steps
        for t in steps_to_show
            if t <= nr_steps
                position = agent_path[t]
                # Extract uncertainties for this agent at this step
                unc_x = uncertainties[k, t][1] * uncertainty_scale
                unc_y = uncertainties[k, t][2] * uncertainty_scale
                
                # Draw ellipse representing uncertainty
                θ = range(0, 2π, 100)
                x_coords = position[1] .+ unc_x .* cos.(θ)
                y_coords = position[2] .+ unc_y .* sin.(θ)
                plot!(p, Shape(x_coords, y_coords), 
                      color = colors[k], 
                      alpha = 0.2, 
                      linewidth = 0,
                      label = t == steps_to_show[1] ? "Agent $k Uncertainty" : "")
            end
        end
        
        # Plot start and end positions with markers
        scatter!(p, [agent_path[1][1]], [agent_path[1][2]], 
                markersize = 8, 
                color = colors[k], 
                markershape = :circle,
                label = "")
        scatter!(p, [agent_path[end][1]], [agent_path[end][2]], 
                markersize = 8, 
                color = colors[k], 
                markershape = :star5,
                label = "")
    end
    
    # Save the plot
    if !isempty(filename)
        println("Saving path uncertainty visualization to $filename...")
        savefig(p, filename)
        println("Path uncertainty visualization saved successfully.")
    end
    
    return p
end

"""
    animate_path_uncertainties(environment, agents, paths, uncertainties; 
                             filename = "path_uncertainty.gif", 
                             fps = 10,
                             x_limits = (-20, 20), 
                             y_limits = (-20, 20), 
                             plot_size = (800, 600),
                             uncertainty_scale = 3.0)

Create an animation showing agent paths with uncertainty ellipses over time.

# Arguments
- `environment`: The environment containing obstacles
- `agents`: List of agent objects
- `paths`: Matrix of agent positions over time, dimensions: (nr_agents, nr_steps)
- `uncertainties`: Matrix of uncertainty values (x, y) for each agent at each step
- `filename`: Output filename for the GIF (default: "path_uncertainty.gif")
- `fps`: Frames per second for the animation (default: 10)
- `x_limits`: Tuple with x-axis limits (default: (-20, 20))
- `y_limits`: Tuple with y-axis limits (default: (-20, 20))
- `plot_size`: Tuple with plot width and height (default: (800, 600))
- `uncertainty_scale`: Scale factor for uncertainty ellipses (default: 3.0)

# Returns
- Nothing
"""
function animate_path_uncertainties(environment, agents, paths, uncertainties; 
                                  filename = "path_uncertainty.gif", 
                                  fps = 10,
                                  x_limits = (-20, 20), 
                                  y_limits = (-20, 20), 
                                  plot_size = (800, 600),
                                  uncertainty_scale = 3.0,
                                  show_full_path = true)
    
    nr_agents, nr_steps = size(paths)
    colors = Plots.palette(:tab10)
    
    println("Generating uncertainty animation frames...")
    animation = @animate for t in 1:nr_steps
        p = plot_environment(environment, 
                           x_limits = x_limits, 
                           y_limits = y_limits, 
                           plot_size = plot_size)
        
        title!(p, "Agent Paths with Uncertainty (Step $t/$nr_steps)")
        
        for k in 1:nr_agents
            position = paths[k, t]
            
            # Plot partial or full path
            path_to_plot = show_full_path ? paths[k, 1:end] : paths[k, 1:t]
            plot!(p, getindex.(path_to_plot, 1), getindex.(path_to_plot, 2), 
                 color = colors[k], 
                 alpha = 0.7,
                 linewidth = 2,
                 label = "")
            
            # Draw uncertainty ellipse
            unc_x = uncertainties[k, t][1] * uncertainty_scale
            unc_y = uncertainties[k, t][2] * uncertainty_scale
            
            θ = range(0, 2π, 100)
            x_coords = position[1] .+ unc_x .* cos.(θ)
            y_coords = position[2] .+ unc_y .* sin.(θ)
            plot!(p, Shape(x_coords, y_coords), 
                  color = colors[k], 
                  alpha = 0.3, 
                  linewidth = 0,
                  label = "")
            
            # Draw agent
            plot_marker_at_position!(p, agents[k].radius, position, color = colors[k], label = k == 1 ? "Agent $k" : "")
            
            # Show target position
            plot_marker_at_position!(p, agents[k].radius, agents[k].target_position, color = colors[k], alpha = 0.2, label = "")
        end
    end
    
    # Save the animation
    println("Saving uncertainty animation to $filename...")
    gif(animation, filename, fps=fps, show_msg = false)
    println("Uncertainty animation saved successfully.")
    
    return nothing
end

"""
    plot_control_signals(controls; 
                       filename = "control_signals.png",
                       plot_size = (800, 600))

Create a visualization of control signals over time.

# Arguments
- `controls`: Matrix of control signals (x, y) for each agent at each step
- `filename`: Output filename for the plot (default: "control_signals.png")
- `plot_size`: Tuple with plot width and height (default: (800, 600))

# Returns
- The plot object
"""
function plot_control_signals(controls; 
                            filename = "control_signals.png",
                            plot_size = (1000, 800),
                            component = :both)
    
    nr_agents, nr_steps = size(controls)
    colors = Plots.palette(:tab10)
    
    if component == :both
        # Create a 2x1 layout with both x and y components
        p = plot(layout = (2, 1), size = plot_size, legend = :topright)
        
        # Plot x component
        for k in 1:nr_agents
            plot!(p[1], getindex.(controls[k, :], 1), 
                 label = "Agent $k", 
                 color = colors[k],
                 linewidth = 2)
        end
        title!(p[1], "Control Signals (X Component)")
        xlabel!(p[1], "Time Step")
        ylabel!(p[1], "Control X")
        
        # Plot y component
        for k in 1:nr_agents
            plot!(p[2], getindex.(controls[k, :], 2), 
                 label = "Agent $k", 
                 color = colors[k],
                 linewidth = 2)
        end
        title!(p[2], "Control Signals (Y Component)")
        xlabel!(p[2], "Time Step")
        ylabel!(p[2], "Control Y")
        
    elseif component == :magnitude
        # Create a single plot with control magnitude
        p = plot(size = plot_size, legend = :topright)
        
        for k in 1:nr_agents
            magnitudes = [sqrt(c[1]^2 + c[2]^2) for c in controls[k, :]]
            plot!(p, magnitudes, 
                 label = "Agent $k", 
                 color = colors[k],
                 linewidth = 2)
        end
        title!(p, "Control Signal Magnitudes")
        xlabel!(p, "Time Step")
        ylabel!(p, "Control Magnitude")
        
    else
        # Create a single plot with specified component
        comp_idx = component == :x ? 1 : 2
        comp_name = component == :x ? "X" : "Y"
        
        p = plot(size = plot_size, legend = :topright)
        
        for k in 1:nr_agents
            plot!(p, getindex.(controls[k, :], comp_idx), 
                 label = "Agent $k", 
                 color = colors[k],
                 linewidth = 2)
        end
        title!(p, "Control Signals ($comp_name Component)")
        xlabel!(p, "Time Step")
        ylabel!(p, "Control $comp_name")
    end
    
    # Save the plot
    if !isempty(filename)
        println("Saving control signals plot to $filename...")
        savefig(p, filename)
        println("Control signals plot saved successfully.")
    end
    
    return p
end

"""
    animate_control_signals(paths, controls; 
                          filename = "control_signals.gif",
                          fps = 10,
                          x_limits = (-20, 20),
                          y_limits = (-20, 20),
                          plot_size = (800, 600),
                          control_scale = 1.0)

Create an animation showing agent positions and control vectors.

# Arguments
- `paths`: Matrix of agent positions over time, dimensions: (nr_agents, nr_steps)
- `controls`: Matrix of control signals (x, y) for each agent at each step
- `filename`: Output filename for the GIF (default: "control_signals.gif")
- `fps`: Frames per second for the animation (default: 10)
- `x_limits`: Tuple with x-axis limits (default: (-20, 20))
- `y_limits`: Tuple with y-axis limits (default: (-20, 20))
- `plot_size`: Tuple with plot width and height (default: (800, 600))
- `control_scale`: Scale factor for control vectors (default: 1.0)

# Returns
- Nothing
"""
function animate_control_signals(environment, paths, controls; 
                               filename = "control_signals.gif",
                               fps = 10,
                               x_limits = (-20, 20),
                               y_limits = (-20, 20),
                               plot_size = (800, 600),
                               control_scale = 5.0)
    
    nr_agents, nr_steps = size(paths)
    colors = Plots.palette(:tab10)
    
    println("Generating control signals animation frames...")
    animation = @animate for t in 1:nr_steps
        p = plot_environment(environment, 
                           x_limits = x_limits, 
                           y_limits = y_limits, 
                           plot_size = plot_size)
        
        title!(p, "Control Signals at Step $t/$nr_steps")
        
        for k in 1:nr_agents
            position = paths[k, t]
            
            # Plot path up to current position
            plot!(p, getindex.(paths[k, 1:t], 1), getindex.(paths[k, 1:t], 2), 
                 color = colors[k], 
                 alpha = 0.7,
                 linewidth = 2,
                 label = "")
            
            # Draw agent
            scatter!(p, [position[1]], [position[2]], 
                    markersize = 6, 
                    color = colors[k], 
                    label = k == 1 ? "Agent $k" : "")
            
            # Draw control vector if not at the last step
            if t < nr_steps
                control = controls[k, t]
                quiver!(p, [position[1]], [position[2]], 
                       quiver = ([control[1] * control_scale], [control[2] * control_scale]),
                       color = colors[k],
                       linewidth = 2)
            end
        end
        
        # Add a legend entry for control vectors
        if t < nr_steps
            plot!(p, [], [], linewidth = 2, color = :black, label = "Control Vector (scaled)")
        end
    end
    
    # Save the animation
    println("Saving control signals animation to $filename...")
    gif(animation, filename, fps=fps, show_msg = false)
    println("Control signals animation saved successfully.")
    
    return nothing
end

"""
    plot_combined_visualization(environment, agents, paths, uncertainties, controls; 
                               filename = "combined_visualization.png",
                               step = :last,
                               x_limits = (-20, 20),
                               y_limits = (-20, 20),
                               plot_size = (1000, 800),
                               uncertainty_scale = 2.0,
                               control_scale = 5.0)

Create a comprehensive visualization combining paths, uncertainties, and control signals.

# Arguments
- `environment`: The environment containing obstacles
- `agents`: List of agent objects
- `paths`: Matrix of agent positions over time, dimensions: (nr_agents, nr_steps)
- `uncertainties`: Matrix of uncertainty values (x, y) for each agent at each step
- `controls`: Matrix of control signals (x, y) for each agent at each step
- `filename`: Output filename for the plot (default: "combined_visualization.png")
- `step`: Which time step to visualize, can be :last, :first, or an integer (default: :last)
- `x_limits`: Tuple with x-axis limits (default: (-20, 20))
- `y_limits`: Tuple with y-axis limits (default: (-20, 20))
- `plot_size`: Tuple with plot width and height (default: (1000, 800))
- `uncertainty_scale`: Scale factor for uncertainty ellipses (default: 2.0)
- `control_scale`: Scale factor for control vectors (default: 5.0)

# Returns
- The plot object
"""
function plot_combined_visualization(environment, agents, paths, uncertainties, controls; 
                                   filename = "combined_visualization.png",
                                   step = :last,
                                   x_limits = (-20, 20),
                                   y_limits = (-20, 20),
                                   plot_size = (1000, 800),
                                   uncertainty_scale = 2.0,
                                   control_scale = 5.0,
                                   show_uncertainty_steps = 5)
    
    nr_agents, nr_steps = size(paths)
    colors = Plots.palette(:tab10)
    
    # Determine which step to visualize
    if step == :last
        t = nr_steps
    elseif step == :first
        t = 1
    else
        t = min(step, nr_steps)
    end
    
    # Create plot with environment
    p = plot_environment(environment, 
                       x_limits = x_limits, 
                       y_limits = y_limits, 
                       plot_size = plot_size)
    
    title!(p, "Combined Visualization (Step $t/$nr_steps)")
    
    # Determine which steps to show uncertainties for
    step_interval = max(1, div(nr_steps, show_uncertainty_steps))
    uncertainty_steps = 1:step_interval:t
    
    for k in 1:nr_agents
        # Plot full path
        plot!(p, getindex.(paths[k, 1:t], 1), getindex.(paths[k, 1:t], 2), 
             color = colors[k], 
             alpha = 0.7,
             linewidth = 2,
             label = "Agent $k")
        
        position = paths[k, t]
        
        # Draw uncertainty ellipses at selected steps
        for step_idx in uncertainty_steps
            pos = paths[k, step_idx]
            unc_x = uncertainties[k, step_idx][1] * uncertainty_scale
            unc_y = uncertainties[k, step_idx][2] * uncertainty_scale
            
            θ = range(0, 2π, 100)
            x_coords = pos[1] .+ unc_x .* cos.(θ)
            y_coords = pos[2] .+ unc_y .* sin.(θ)
            plot!(p, Shape(x_coords, y_coords), 
                  color = colors[k], 
                  alpha = 0.2, 
                  linewidth = 0,
                  label = "")
        end
        
        # Draw agent
        plot_marker_at_position!(p, agents[k].radius, position, color = colors[k], label = "")
        
        # Draw control vector if not at the last step
        if t < nr_steps
            control = controls[k, t]
            quiver!(p, [position[1]], [position[2]], 
                   quiver = ([control[1] * control_scale], [control[2] * control_scale]),
                   color = colors[k],
                   linewidth = 2)
        end
        
        # Show start and target positions
        scatter!(p, [agents[k].initial_position[1]], [agents[k].initial_position[2]], 
                markersize = 8, 
                color = colors[k], 
                markershape = :circle,
                label = "")
        scatter!(p, [agents[k].target_position[1]], [agents[k].target_position[2]], 
                markersize = 8, 
                color = colors[k], 
                markershape = :star5,
                label = "")
    end
    
    # Save the plot
    if !isempty(filename)
        println("Saving combined visualization to $filename...")
        savefig(p, filename)
        println("Combined visualization saved successfully.")
    end
    
    return p
end

"""
    animate_combined_visualization(environment, agents, paths, uncertainties, controls; 
                                 filename = "combined_visualization.gif",
                                 fps = 10,
                                 x_limits = (-20, 20),
                                 y_limits = (-20, 20),
                                 plot_size = (1000, 800),
                                 uncertainty_scale = 2.0,
                                 control_scale = 5.0)

Create an animation showing the combined visualization of paths, uncertainties, and controls.

# Arguments
- Same as plot_combined_visualization, with the addition of fps for animation control

# Returns
- Nothing
"""
function animate_combined_visualization(environment, agents, paths, uncertainties, controls; 
                                      filename = "combined_visualization.gif",
                                      fps = 10,
                                      x_limits = (-20, 20),
                                      y_limits = (-20, 20),
                                      plot_size = (1000, 800),
                                      uncertainty_scale = 2.0,
                                      control_scale = 5.0,
                                      show_trail = true,
                                      trail_length = 10)
    
    nr_agents, nr_steps = size(paths)
    colors = Plots.palette(:tab10)
    
    println("Generating combined visualization animation frames...")
    animation = @animate for t in 1:nr_steps
        # Create a new plot for each frame with the environment
        p = plot_environment(environment, 
                           x_limits = x_limits, 
                           y_limits = y_limits, 
                           plot_size = plot_size)
        
        title!(p, "Combined Visualization (Step $t/$nr_steps)")
        
        for k in 1:nr_agents
            # Determine the trail (past positions to show)
            if show_trail
                trail_start = max(1, t - trail_length)
                trail = paths[k, trail_start:t]
            else
                trail = paths[k, 1:t]
            end
            
            # Plot the trail
            plot!(p, getindex.(trail, 1), getindex.(trail, 2), 
                 color = colors[k], 
                 alpha = 0.7,
                 linewidth = 2,
                 label = k == 1 ? "Agent $k" : "")
            
            position = paths[k, t]
            
            # Draw uncertainty ellipse at current position
            unc_x = uncertainties[k, t][1] * uncertainty_scale
            unc_y = uncertainties[k, t][2] * uncertainty_scale
            
            θ = range(0, 2π, 100)
            x_coords = position[1] .+ unc_x .* cos.(θ)
            y_coords = position[2] .+ unc_y .* sin.(θ)
            plot!(p, Shape(x_coords, y_coords), 
                  color = colors[k], 
                  alpha = 0.3, 
                  linewidth = 0,
                  label = "")
            
            # Draw agent
            plot_marker_at_position!(p, agents[k].radius, position, color = colors[k], label = "")
            
            # Draw control vector if not at the last step
            if t < nr_steps
                control = controls[k, t]
                quiver!(p, [position[1]], [position[2]], 
                       quiver = ([control[1] * control_scale], [control[2] * control_scale]),
                       color = colors[k],
                       linewidth = 2)
            end
            
            # Show target position
            plot_marker_at_position!(p, agents[k].radius, agents[k].target_position, 
                                   color = colors[k], 
                                   alpha = 0.2, 
                                   label = "")
        end
        
        # Add a legend entry for uncertainty and control
        if t == 1
            plot!(p, [], [], linewidth = 0, color = :black, alpha = 0.3, label = "Uncertainty")
            if t < nr_steps
                plot!(p, [], [], linewidth = 2, color = :black, label = "Control Vector")
            end
        end
    end
    
    # Save the animation
    println("Saving combined visualization animation to $filename...")
    gif(animation, filename, fps=fps, show_msg = false)
    println("Combined visualization animation saved successfully.")
    
    return nothing
end

"""
    plot_agent_path_3d(paths, agent_idx; 
                     filename = "agent_path_3d.png",
                     plot_size = (800, 600),
                     with_time = true)

Create a 3D visualization of an agent's path, with time as the third dimension.

# Arguments
- `paths`: Matrix of agent positions over time, dimensions: (nr_agents, nr_steps)
- `agent_idx`: Index of the agent to visualize
- `filename`: Output filename for the plot (default: "agent_path_3d.png")
- `plot_size`: Tuple with plot width and height (default: (800, 600))
- `with_time`: If true, use time as z-axis; if false, use uncertainty magnitude (default: true)

# Returns
- The plot object
"""
function plot_agent_path_3d(paths, agent_idx; 
                          uncertainties = nothing,
                          filename = "agent_path_3d.png",
                          plot_size = (800, 600),
                          with_time = true,
                          colormap = :viridis)
    
    agent_path = paths[agent_idx, :]
    nr_steps = length(agent_path)
    
    # Extract x and y coordinates
    x_coords = getindex.(agent_path, 1)
    y_coords = getindex.(agent_path, 2)
    
    # Determine z coordinates based on the mode
    if with_time
        z_coords = 1:nr_steps
        z_label = "Time Step"
    else
        # Use uncertainty magnitude if provided
        if isnothing(uncertainties)
            error("Uncertainties must be provided when with_time=false")
        end
        agent_uncertainties = uncertainties[agent_idx, :]
        z_coords = [sqrt(unc[1]^2 + unc[2]^2) for unc in agent_uncertainties]
        z_label = "Uncertainty Magnitude"
    end
    
    # Create 3D plot
    p = plot(x_coords, y_coords, z_coords, 
            seriestype = :scatter,
            markersize = 4,
            markerstrokewidth = 0,
            markercolor = z_coords,
            colormap = colormap,
            label = "",
            size = plot_size)
    
    # Add line connecting points
    plot!(p, x_coords, y_coords, z_coords, 
         seriestype = :path,
         linewidth = 2,
         linealpha = 0.7,
         label = "Agent $agent_idx Path")
    
    # Add start and end markers
    scatter!(p, [x_coords[1]], [y_coords[1]], [z_coords[1]], 
            markersize = 8, 
            color = :green, 
            markershape = :circle,
            label = "Start")
    scatter!(p, [x_coords[end]], [y_coords[end]], [z_coords[end]], 
            markersize = 8, 
            color = :red, 
            markershape = :star5,
            label = "End")
    
    # Add labels
    xlabel!(p, "X Position")
    ylabel!(p, "Y Position")
    zlabel!(p, z_label)
    title!(p, "3D Path Visualization for Agent $agent_idx")
    
    # Save the plot
    if !isempty(filename)
        println("Saving 3D path visualization to $filename...")
        savefig(p, filename)
        println("3D path visualization saved successfully.")
    end
    
    return p
end

"""
    create_path_heatmap(environment, paths, resolution = 100;
                      filename = "path_heatmap.png",
                      plot_size = (800, 600),
                      x_limits = (-20, 20),
                      y_limits = (-20, 20))

Create a heatmap showing the density of agent paths.

# Arguments
- `environment`: The environment containing obstacles
- `paths`: Matrix of agent positions over time, dimensions: (nr_agents, nr_steps)
- `resolution`: Resolution of the heatmap grid (default: 100)
- `filename`: Output filename for the plot (default: "path_heatmap.png")
- `plot_size`: Tuple with plot width and height (default: (800, 600))
- `x_limits`: Tuple with x-axis limits (default: (-20, 20))
- `y_limits`: Tuple with y-axis limits (default: (-20, 20))

# Returns
- The plot object
"""
function create_path_heatmap(environment, paths, resolution = 100;
                           filename = "path_heatmap.png",
                           plot_size = (800, 600),
                           x_limits = (-20, 20),
                           y_limits = (-20, 20),
                           colormap = :thermal)
    
    nr_agents, nr_steps = size(paths)
    
    # Create a grid for the heatmap
    x_range = range(x_limits[1], x_limits[2], length = resolution)
    y_range = range(y_limits[1], y_limits[2], length = resolution)
    grid = zeros(resolution, resolution)
    
    # Populate the grid with path density
    for k in 1:nr_agents
        for t in 1:nr_steps
            pos = paths[k, t]
            
            # Find the nearest grid cell
            x_idx = argmin(abs.(x_range .- pos[1]))
            y_idx = argmin(abs.(y_range .- pos[2]))
            
            # Increment the count for this cell
            grid[y_idx, x_idx] += 1
        end
    end
    
    # Create the heatmap
    p = heatmap(x_range, y_range, grid,
               color = colormap,
               size = plot_size,
               xlabel = "X",
               ylabel = "Y",
               title = "Path Density Heatmap")
    
    # Overlay the environment
    plot_environment!(p, environment)
    
    # Save the plot
    if !isempty(filename)
        println("Saving path heatmap to $filename...")
        savefig(p, filename)
        println("Path heatmap saved successfully.")
    end
    
    return p
end

end # module 