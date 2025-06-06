module TrajectoryPlanning

using LinearAlgebra
using RxInfer
using Plots
using LogExpFunctions
using StableRNGs
using Dates
using Statistics
using DelimitedFiles

export run_all_experiments

#
# Environment structures and functions
#

# A simple struct to represent a rectangle, which is defined by its center (x, y) and size (width, height)
Base.@kwdef struct Rectangle
    center::Tuple{Float64, Float64}
    size::Tuple{Float64, Float64}
end

# A simple struct to represent an environment, which is defined by a list of obstacles,
# and in this demo the obstacles are just rectangles
Base.@kwdef struct Environment
    obstacles::Vector{Rectangle}
end

# Agent plan, encodes start and goal states
Base.@kwdef struct Agent
    radius::Float64
    initial_position::Tuple{Float64, Float64}
    target_position::Tuple{Float64, Float64}
end

#
# Visualization functions
#

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

function plot_environment!(p, env::Environment)
    for obstacle in env.obstacles
        plot_rectangle!(p, obstacle)
    end
    return p
end

function plot_environment(env::Environment)
    p = plot(size = (800, 400), xlims = (-20, 20), ylims = (-20, 20), aspect_ratio = :equal)
    plot_environment!(p, env)
    return p
end

function plot_marker_at_position!(p, radius, position; color="red", markersize=10.0, alpha=1.0, label="")
    # Draw the agent as a circle with the given radius
    θ = range(0, 2π, 100)
    
    x_coords = position[1] .+ radius .* cos.(θ)
    y_coords = position[2] .+ radius .* sin.(θ)
    
    plot!(p, Shape(x_coords, y_coords); color=color, label=label, alpha=alpha)
    return p
end

function plot_agent_naive_plan!(p, agent; color = "blue")
    plot_marker_at_position!(p, agent.radius, agent.initial_position, color = color)
    plot_marker_at_position!(p, agent.radius, agent.target_position, color = color, alpha = 0.1)
    quiver!(p, [ agent.initial_position[1] ], [ agent.initial_position[2] ], 
            quiver = ([ agent.target_position[1] - agent.initial_position[1] ], 
                    [ agent.target_position[2] -  agent.initial_position[2] ]))
end

function plot_agent_plans!(p, agents)
    colors = Plots.palette(:tab10)
    
    for (k, agent) in enumerate(agents)
        plot_agent_naive_plan!(p, agent, color = colors[k])
    end
    
    return p
end

function animate_paths(environment, agents, paths; output_dir=".", filename = "result.gif", fps = 15, logger=nothing)
    nr_agents, nr_steps = size(paths)
    colors = Plots.palette(:tab10)

    log_message("Generating animation frames...", logger)
    animation = @animate for t in 1:nr_steps
        frame = plot_environment(environment)
    
        for k in 1:nr_agents
            position = paths[k, t]          
            path = paths[k, 1:t]
            
            plot_marker_at_position!(frame, agents[k].radius, position, color = colors[k])
            plot_marker_at_position!(frame, agents[k].radius, agents[k].target_position, color = colors[k], alpha = 0.2)
            plot!(frame, getindex.(path, 1), getindex.(path, 2); linestyle=:dash, label="", color=colors[k])
        end

        frame
    end

    # Save the animation
    output_path = joinpath(output_dir, filename)
    log_message("Saving animation to $output_path...", logger)
    gif(animation, output_path, fps=fps, show_msg = false)
    log_message("Animation saved successfully.", logger)
    
    return nothing
end

# NEW: Visualize control signals as quiver plot
function visualize_controls(agents, controls, paths; output_dir=".", filename="control_signals.gif", fps=10, logger=nothing)
    nr_agents, nr_steps = size(controls)
    colors = Plots.palette(:tab10)
    
    log_message("Generating control signal visualization...", logger)
    animation = @animate for t in 1:nr_steps
        p = plot(xlims=(-20, 20), ylims=(-20, 20), aspect_ratio=:equal, 
                 title="Control Signals at Step $t", size=(800, 400))
        
        for k in 1:nr_agents
            position = paths[k, t]
            control = controls[k, t]
            
            # Scale control for better visualization
            scale = 2.0
            
            # Plot agent position
            scatter!([position[1]], [position[2]], 
                    color=colors[k], markersize=6, label="Agent $k")
            
            # Plot control vector as quiver
            quiver!([position[1]], [position[2]], 
                   quiver=([control[1]*scale], [control[2]*scale]), 
                   color=colors[k], linewidth=2)
        end
        
        p
    end
    
    output_path = joinpath(output_dir, filename)
    log_message("Saving control visualization to $output_path...", logger)
    gif(animation, output_path, fps=fps, show_msg=false)
    log_message("Control visualization saved successfully.", logger)
    
    return nothing
end

# NEW: Visualize distance to obstacles as heatmap
function visualize_obstacle_distance(environment; output_dir=".", filename="obstacle_distance.png", resolution=100, logger=nothing)
    log_message("Generating obstacle distance heatmap...", logger)
    
    x_range = range(-20, 20, length=resolution)
    y_range = range(-20, 20, length=resolution)
    
    # Calculate distance values
    distance_values = zeros(resolution, resolution)
    for (i, x) in enumerate(x_range)
        for (j, y) in enumerate(y_range)
            state = [x, y]
            distance_values[i, j] = distance(environment, state)
        end
    end
    
    # Create plot
    p = heatmap(x_range, y_range, distance_values', 
                xlabel="X", ylabel="Y", 
                title="Distance to Obstacles", 
                color=:viridis, size=(800, 600))
    
    # Overlay environment
    plot_environment!(p, environment)
    
    output_path = joinpath(output_dir, filename)
    log_message("Saving obstacle distance heatmap to $output_path...", logger)
    savefig(p, output_path)
    log_message("Obstacle distance heatmap saved successfully.", logger)
    
    return nothing
end

# NEW: Visualize agent paths with uncertainty
function visualize_path_uncertainty(environment, agents, paths, path_vars; output_dir=".", filename="path_uncertainty.png", logger=nothing)
    nr_agents, nr_steps = size(paths)
    colors = Plots.palette(:tab10)
    
    log_message("Generating path uncertainty visualization...", logger)
    p = plot_environment(environment)
    
    for k in 1:nr_agents
        # Get agent path and uncertainty
        path_points = paths[k, :]
        uncertainties = [sqrt(var[1] + var[2]) for var in path_vars[k, :]]
        
        # Plot path
        x_coords = [point[1] for point in path_points]
        y_coords = [point[2] for point in path_points]
        
        # Plot path line
        plot!(p, x_coords, y_coords, linewidth=2, color=colors[k], label="Agent $k")
        
        # Plot uncertainty as varying point sizes
        for (i, (x, y, uncertainty)) in enumerate(zip(x_coords, y_coords, uncertainties))
            # Only plot some points to avoid clutter
            if i % 5 == 0
                scatter!([x], [y], markersize=3 + 20*uncertainty, 
                         color=colors[k], alpha=0.3, label="")
            end
        end
        
        # Plot start and end positions
        plot_marker_at_position!(p, agents[k].radius, agents[k].initial_position, color=colors[k])
        plot_marker_at_position!(p, agents[k].radius, agents[k].target_position, color=colors[k], alpha=0.3)
    end
    
    output_path = joinpath(output_dir, filename)
    log_message("Saving path uncertainty visualization to $output_path...", logger)
    savefig(p, output_path)
    log_message("Path uncertainty visualization saved successfully.", logger)
    
    return nothing
end

# NEW: Plot convergence metrics
function plot_convergence_metrics(metrics; output_dir=".", filename="convergence.png", logger=nothing)
    log_message("Generating convergence metrics plot...", logger)
    
    p = plot(metrics, linewidth=2, xlabel="Iteration", ylabel="ELBO",
             title="Convergence of Inference", legend=false, size=(800, 400))
    
    output_path = joinpath(output_dir, filename)
    log_message("Saving convergence plot to $output_path...", logger)
    savefig(p, output_path)
    log_message("Convergence plot saved successfully.", logger)
    
    return nothing
end

#
# Model functions
#

# Define the probabilistic model for obstacles using halfspace constraints
struct Halfspace end

@node Halfspace Stochastic [out, a, σ2, γ]

# rule specification
@rule Halfspace(:out, Marginalisation) (q_a::Any, q_σ2::Any, q_γ::Any) = begin
    return NormalMeanVariance(mean(q_a) + mean(q_γ) * mean(q_σ2), mean(q_σ2))
end

@rule Halfspace(:σ2, Marginalisation) (q_out::Any, q_a::Any, q_γ::Any, ) = begin
    # `BayesBase.TerminalProdArgument` is used to ensure that the result of the posterior computation is equal to this value
    return BayesBase.TerminalProdArgument(PointMass( 1 / mean(q_γ) * sqrt(abs2(mean(q_out) - mean(q_a)) + var(q_out))))
end

softmin(x; l=10) = -logsumexp(-l .* x) / l

# state here is a 4-dimensional vector [x, y, vx, vy]
function distance(r::Rectangle, state)
    if abs(state[1] - r.center[1]) > r.size[1] / 2 || abs(state[2] - r.center[2]) > r.size[2] / 2
        # outside of rectangle
        dx = max(abs(state[1] - r.center[1]) - r.size[1] / 2, 0)
        dy = max(abs(state[2] - r.center[2]) - r.size[2] / 2, 0)
        return sqrt(dx^2 + dy^2)
    else
        # inside rectangle
        return max(abs(state[1] - r.center[1]) - r.size[1] / 2, abs(state[2] - r.center[2]) - r.size[2] / 2)
    end
end

function distance(env::Environment, state)
    return softmin([distance(obstacle, state) for obstacle in env.obstacles])
end

# Helper function, distance with radius offset
function g(environment, radius, state)
    return distance(environment, state) - radius
end

# Helper function, finds minimum distances between agents pairwise
function h(environment, radiuses, states...)
    # Calculate pairwise distances between all agents
    distances = Real[]
    n = length(states)

    for i in 1:n
        for j in (i+1):n
            push!(distances, norm(states[i] - states[j]) - radiuses[i] - radiuses[j])
        end
    end

    return softmin(distances)
end

# For more details about the model, please refer to the original paper
@model function path_planning_model(environment, agents, goals, nr_steps)

    # Model's parameters are fixed, refer to the original 
    # paper's implementation for more details about these parameters
    local dt = 1
    local A  = [1 dt 0 0; 0 1 0 0; 0 0 1 dt; 0 0 0 1]
    local B  = [0 0; dt 0; 0 0; 0 dt]
    local C  = [1 0 0 0; 0 0 1 0]
    local γ  = 1

    local control
    local state
    local path   
    
    # Extract radiuses of each agent in a separate collection
    local rs = map((a) -> a.radius, agents)

    # Model is fixed for 4 agents
    for k in 1:4

        # Prior on state, the state structure is 4 dimensional, where
        # [ x_position, x_velocity, y_position, y_velocity ]
        state[k, 1] ~ MvNormal(mean = zeros(4), covariance = 1e2I)

        for t in 1:nr_steps

            # Prior on controls
            control[k, t] ~ MvNormal(mean = zeros(2), covariance = 1e-1I)

            # State transition
            state[k, t+1] ~ A * state[k, t] + B * control[k, t]

            # Path model, the path structure is 2 dimensional, where 
            # [ x_position, y_position ]
            path[k, t] ~ C * state[k, t+1]

            # Environmental distance
            zσ2[k, t] ~ GammaShapeRate(3 / 2, γ^2 / 2)
            z[k, t]   ~ g(environment, rs[k], path[k, t])
            
            # Halfspase priors were defined previousle in this experiment
            z[k, t] ~ Halfspace(0, zσ2[k, t], γ)

        end

        # goal priors (indexing reverse due to definition)
        goals[1, k] ~ MvNormal(mean = state[k, 1], covariance = 1e-5I)
        goals[2, k] ~ MvNormal(mean = state[k, nr_steps+1], covariance = 1e-5I)

    end

    for t = 1:nr_steps

        # observation constraint
        dσ2[t] ~ GammaShapeRate(3 / 2, γ^2 / 2)
        d[t] ~ h(environment, rs, path[1, t], path[2, t], path[3, t], path[4, t])
        d[t] ~ Halfspace(0, dσ2[t], γ)

    end

end

@constraints function path_planning_constraints()
    # Mean-field variational constraints on the parameters
    q(d, dσ2) = q(d)q(dσ2)
    q(z, zσ2) = q(z)q(zσ2)
end

function path_planning(; environment, agents, nr_iterations = 350, nr_steps = 40, seed = 42, 
                       save_intermediates = false, intermediate_steps = 10, 
                       output_dir = ".", logger=nothing)
    log_message("Starting path planning inference...", logger)
    # Fixed number of agents
    nr_agents = 4

    # Form goals compatible with the model
    goals = hcat(
        map(agents) do agent
            return [
                [ agent.initial_position[1], 0, agent.initial_position[2], 0 ],
                [ agent.target_position[1], 0, agent.target_position[2], 0 ]
            ]
        end...
    )
    
    rng = StableRNG(seed)
    
    # Initialize variables, more details about initialization 
    # can be found in the original paper
    init = @initialization begin

        q(dσ2) = repeat([PointMass(1)], nr_steps)
        q(zσ2) = repeat([PointMass(1)], nr_agents, nr_steps)
        q(control) = repeat([PointMass(0)], nr_steps)

        μ(state) = MvNormalMeanCovariance(randn(rng, 4), 100I)
        μ(path) = MvNormalMeanCovariance(randn(rng, 2), 100I)

    end

    # Define approximation methods for the non-linear functions used in the model
    # `Linearization` is a simple and fast approximation method, but it is not
    # the most accurate one. For more details about the approximation methods,
    # please refer to the RxInfer documentation
    door_meta = @meta begin 
        h() -> Linearization()
        g() -> Linearization()
    end

    # For tracking convergence and intermediate results
    convergence_metrics = Float64[]
    intermediate_results = []
    
    # Create a callback for tracking ELBO and intermediate results
    callback = (metadata) -> begin
        # Track ELBO 
        push!(convergence_metrics, metadata.free_energy)
        
        # Save intermediate results if requested
        if save_intermediates && (metadata.iteration % intermediate_steps == 0 || 
                              metadata.iteration == nr_iterations)
            log_message("Saving intermediate results at iteration $(metadata.iteration)", logger)
            push!(intermediate_results, (
                iteration = metadata.iteration,
                path = [mean(marginal) for marginal in metadata.marginals[:path]],
                control = [mean(marginal) for marginal in metadata.marginals[:control]],
                state = [mean(marginal) for marginal in metadata.marginals[:state]]
            ))
        end
        
        # Print progress every 50 iterations
        if metadata.iteration % 50 == 0
            log_message("Iteration $(metadata.iteration)/$(nr_iterations) - ELBO: $(metadata.free_energy)", logger)
        end
        
        return nothing
    end

    log_message("Running inference with $(nr_iterations) iterations...", logger)
    results = infer(
        model 			= path_planning_model(environment = environment, agents = agents, nr_steps = nr_steps),
        data  			= (goals = goals, ),
        initialization  = init,
        constraints 	= path_planning_constraints(),
        meta 			= door_meta,
        iterations 		= nr_iterations,
        callbacks       = (inference = callback, ),
        returnvars 		= KeepLast(), 
        options         = (limit_stack_depth = 300, warn = false)
    )
    log_message("Inference completed.", logger)
    
    # Save convergence metrics
    if !isempty(convergence_metrics)
        metrics_file = joinpath(output_dir, "convergence_metrics.csv")
        open(metrics_file, "w") do f
            for (i, metric) in enumerate(convergence_metrics)
                println(f, "$i,$metric")
            end
        end
        log_message("Saved convergence metrics to $metrics_file", logger)
        
        # Manually create and save the convergence plot
        log_message("Generating convergence metrics plot...", logger)
        p = plot(convergence_metrics, linewidth=2, xlabel="Iteration", ylabel="ELBO",
                title="Convergence of Inference", legend=false, size=(800, 400))
        output_path = joinpath(output_dir, "convergence.png")
        savefig(p, output_path)
        log_message("Convergence plot saved successfully to $output_path", logger)
    else
        # Generate a flat line with "Data Not Found" message instead of sample data
        log_message("No convergence metrics available. Generating placeholder convergence plot...", logger)
        flat_data = zeros(100)  # Flat line at zero
        p = plot(flat_data, linewidth=2, xlabel="Iteration", ylabel="ELBO",
                title="Convergence of Inference", legend=false, size=(800, 400),
                color=:gray, alpha=0.5)
        annotate!(p, 50, 0.5, text("Data Not Found", :red, :center, 12))
        output_path = joinpath(output_dir, "convergence.png")
        savefig(p, output_path)
        log_message("Placeholder convergence plot saved to $output_path", logger)
    end
    
    # Save intermediate results if available
    if save_intermediates && !isempty(intermediate_results)
        # Create animations for intermediate steps
        for (i, result) in enumerate(intermediate_results)
            paths_matrix = reshape(result.path, nr_agents, nr_steps)
            
            # Only create animations for some key steps
            if i == 1 || i == length(intermediate_results) || 
               (i % (length(intermediate_results) ÷ 5) == 0)
                iter_num = result.iteration
                animate_paths(environment, agents, paths_matrix, 
                              output_dir=output_dir, 
                              filename="path_iteration_$(iter_num).gif", 
                              logger=logger)
            end
        end
    end
    
    # Extract and return mean paths and controls
    paths = mean.(results.posteriors[:path])
    path_vars = var.(results.posteriors[:path])
    controls = mean.(results.posteriors[:control])
    
    # Save path data
    paths_matrix = reshape(paths, nr_agents, nr_steps)
    controls_matrix = reshape(controls, nr_agents, nr_steps)
    path_vars_matrix = reshape(path_vars, nr_agents, nr_steps)
    
    # Create control visualization
    visualize_controls(agents, controls_matrix, paths_matrix, 
                      output_dir=output_dir, logger=logger)
    
    # Create obstacle distance heatmap
    visualize_obstacle_distance(environment, output_dir=output_dir, logger=logger)
    
    # Create path uncertainty visualization
    visualize_path_uncertainty(environment, agents, paths_matrix, path_vars_matrix, 
                              output_dir=output_dir, logger=logger)
    
    # Save raw data for further analysis
    save_path_data(paths_matrix, controls_matrix, path_vars_matrix, output_dir, logger)
    
    return (
        paths = paths_matrix,
        controls = controls_matrix,
        path_vars = path_vars_matrix,
        convergence = convergence_metrics,
        result = results
    )
end

# NEW: Save path data to CSV files
function save_path_data(paths, controls, path_vars, output_dir=".", logger=nothing)
    log_message("Saving raw path and control data...", logger)
    
    # Save paths
    open(joinpath(output_dir, "paths.csv"), "w") do f
        nr_agents, nr_steps = size(paths)
        for k in 1:nr_agents
            for t in 1:nr_steps
                position = paths[k, t]
                println(f, "$k,$t,$(position[1]),$(position[2])")
            end
        end
    end
    
    # Save controls
    open(joinpath(output_dir, "controls.csv"), "w") do f
        nr_agents, nr_steps = size(controls)
        for k in 1:nr_agents
            for t in 1:nr_steps
                control = controls[k, t]
                println(f, "$k,$t,$(control[1]),$(control[2])")
            end
        end
    end
    
    # Save uncertainties
    open(joinpath(output_dir, "uncertainties.csv"), "w") do f
        nr_agents, nr_steps = size(path_vars)
        for k in 1:nr_agents
            for t in 1:nr_steps
                uncertainty = path_vars[k, t]
                println(f, "$k,$t,$(uncertainty[1]),$(uncertainty[2])")
            end
        end
    end
    
    log_message("Raw data saved successfully.", logger)
end

#
# Logging and output management
#

# Utility function to create timestamp-based directory
function create_timestamped_dir(base_dir="results")
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    output_dir = joinpath(base_dir, timestamp)
    mkpath(output_dir)
    return output_dir
end

# Simple logger to both console and file
mutable struct DualLogger
    log_file::Union{IOStream, Nothing}
    
    function DualLogger(log_path)
        mkpath(dirname(log_path))
        log_file = open(log_path, "w")
        new(log_file)
    end
end

function close_logger(logger::DualLogger)
    if logger.log_file !== nothing
        close(logger.log_file)
        logger.log_file = nothing
    end
end

function log_message(message, logger::Union{DualLogger, Nothing}=nothing)
    # Always print to console
    println(message)
    
    # If logger is provided, also write to log file
    if logger !== nothing && logger.log_file !== nothing
        println(logger.log_file, message)
        flush(logger.log_file)
    end
end

#
# Experiment functions
#

# Create standard environments
function create_door_environment()
    return Environment(obstacles = [
        Rectangle(center = (-40, 0), size = (70, 5)),
        Rectangle(center = (40, 0), size = (70, 5))
    ])
end

function create_wall_environment()
    return Environment(obstacles = [
        Rectangle(center = (0, 0), size = (10, 5))
    ])
end

function create_combined_environment()
    return Environment(obstacles = [
        Rectangle(center = (-50, 0), size = (70, 2)),
        Rectangle(center = (50, -0), size = (70, 2)),
        Rectangle(center = (5, -1), size = (3, 10))
    ])
end

# Create standard agents
function create_standard_agents()
    return [
        Agent(radius = 2.5, initial_position = (-4, 10), target_position = (-10, -10)),
        Agent(radius = 1.5, initial_position = (-10, 5), target_position = (10, -15)),
        Agent(radius = 1.0, initial_position = (-15, -10), target_position = (10, 10)),
        Agent(radius = 2.5, initial_position = (0, -10), target_position = (-10, 15))
    ]
end

function execute_and_save_animation(environment, agents; 
                                   output_dir = ".", 
                                   gifname = "result.gif", 
                                   seed = 42,
                                   save_intermediates = false,
                                   logger = nothing,
                                   kwargs...)
    log_message("Planning paths for environment with $(length(environment.obstacles)) obstacles...", logger)
    
    # Run path planning
    result = path_planning(
        environment = environment, 
        agents = agents, 
        seed = seed,
        save_intermediates = save_intermediates,
        output_dir = output_dir,
        logger = logger,
        kwargs...
    )
    
    # Create animation and save it
    animate_paths(
        environment, 
        agents, 
        result.paths, 
        output_dir = output_dir,
        filename = gifname,
        logger = logger
    )
    
    # Generate ELBO convergence plot if available
    if haskey(result, :convergence) && !isempty(result.convergence)
        log_message("Generating ELBO convergence plot...", logger)
        plot_convergence_metrics(
            result.convergence,
            output_dir = output_dir,
            filename = "convergence.png",
            logger = logger
        )
    end
    
    return result
end

# NEW: Generate a comprehensive summary of the experiment
function generate_experiment_summary(; environment, agents, result, output_dir=".", logger=nothing)
    log_message("Generating experiment summary...", logger)
    
    summary_file = joinpath(output_dir, "experiment_summary.txt")
    open(summary_file, "w") do f
        println(f, "# Experiment Summary")
        println(f, "Generated at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(f, "")
        
        println(f, "## Environment")
        println(f, "Number of obstacles: $(length(environment.obstacles))")
        for (i, obstacle) in enumerate(environment.obstacles)
            println(f, "  Obstacle $i: center=$(obstacle.center), size=$(obstacle.size)")
        end
        println(f, "")
        
        println(f, "## Agents")
        for (i, agent) in enumerate(agents)
            println(f, "  Agent $i: radius=$(agent.radius)")
            println(f, "    Initial position: $(agent.initial_position)")
            println(f, "    Target position: $(agent.target_position)")
            
            # Calculate path length
            path_length = 0.0
            for t in 1:(size(result.paths, 2)-1)
                path_length += norm(result.paths[i, t+1] - result.paths[i, t])
            end
            println(f, "    Path length: $(round(path_length, digits=2))")
            
            # Calculate average control magnitude
            avg_control = mean([norm(control) for control in result.controls[i, :]])
            println(f, "    Average control magnitude: $(round(avg_control, digits=4))")
            
            # Calculate average uncertainty
            avg_uncertainty = mean([sqrt(var[1] + var[2]) for var in result.path_vars[i, :]])
            println(f, "    Average position uncertainty: $(round(avg_uncertainty, digits=4))")
            println(f, "")
        end
        
        println(f, "## Inference Summary")
        if !isempty(result.convergence)
            initial_elbo = result.convergence[1]
            final_elbo = result.convergence[end]
            improvement = final_elbo - initial_elbo
            println(f, "  Initial ELBO: $initial_elbo")
            println(f, "  Final ELBO: $final_elbo")
            println(f, "  Improvement: $improvement")
        end
    end
    
    log_message("Experiment summary saved to $summary_file", logger)
end

function run_all_experiments()
    # Create a timestamped directory for outputs
    output_dir = create_timestamped_dir()
    log_message("Created output directory: $output_dir")
    
    # Create a logger that writes to both console and file
    log_file = joinpath(output_dir, "experiment.log")
    logger = DualLogger(log_file)
    
    log_message("Starting Multi-agent Trajectory Planning experiments...", logger)
    
    # Create environments
    door_environment = create_door_environment()
    wall_environment = create_wall_environment()
    combined_environment = create_combined_environment()
    
    # Create agents
    agents = create_standard_agents()
    
    try
        # Create environment heatmaps first
        log_message("Generating environment visualizations...", logger)
        visualize_obstacle_distance(door_environment, output_dir=output_dir, 
                                  filename="door_environment_heatmap.png", logger=logger)
        visualize_obstacle_distance(wall_environment, output_dir=output_dir, 
                                  filename="wall_environment_heatmap.png", logger=logger)
        visualize_obstacle_distance(combined_environment, output_dir=output_dir, 
                                  filename="combined_environment_heatmap.png", logger=logger)
        
        # Run the door environment experiments
        log_message("Running experiments for door environment...", logger)
        door_result_1 = execute_and_save_animation(
            door_environment, agents, 
            output_dir=output_dir, 
            gifname="door_42.gif", 
            seed=42, 
            save_intermediates=true,
            logger=logger
        )

        log_message("Running experiments with different seed...", logger)
        door_result_2 = execute_and_save_animation(
            door_environment, agents, 
            output_dir=output_dir, 
            gifname="door_123.gif", 
            seed=123, 
            logger=logger
        )

        # Run the wall environment experiments
        log_message("Running experiments for wall environment...", logger)
        wall_result_1 = execute_and_save_animation(
            wall_environment, agents, 
            output_dir=output_dir, 
            gifname="wall_42.gif", 
            seed=42, 
            logger=logger
        )

        log_message("Running experiments with different seed...", logger)
        wall_result_2 = execute_and_save_animation(
            wall_environment, agents, 
            output_dir=output_dir, 
            gifname="wall_123.gif", 
            seed=123, 
            logger=logger
        )

        # Run the combined environment experiments
        log_message("Running experiments for combined environment...", logger)
        combined_result_1 = execute_and_save_animation(
            combined_environment, agents, 
            output_dir=output_dir, 
            gifname="combined_42.gif", 
            seed=42, 
            logger=logger
        )

        log_message("Running final experiment...", logger)
        combined_result_2 = execute_and_save_animation(
            combined_environment, agents, 
            output_dir=output_dir, 
            gifname="combined_123.gif", 
            seed=123, 
            save_intermediates=true,
            logger=logger
        )

        # Create README file with experiment overview
        create_readme(output_dir, logger)

        log_message("All experiments completed successfully.", logger)
        log_message("Results saved to: $output_dir", logger)
    catch e
        log_message("ERROR: Experiment failed with error: $e", logger)
        for (exc, bt) in Base.catch_stack()
            showerror(logger.log_file, exc, bt)
            println(logger.log_file)
        end
        rethrow(e)
    finally
        # Close the logger to ensure the file is properly closed
        close_logger(logger)
    end
    
    return output_dir
end

# NEW: Create a README file for the experiment results
function create_readme(output_dir, logger)
    log_message("Creating README for results...", logger)
    
    readme_file = joinpath(output_dir, "README.md")
    open(readme_file, "w") do f
        println(f, "# Multi-agent Trajectory Planning Results")
        println(f, "")
        println(f, "Generated at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(f, "")
        println(f, "## Contents")
        println(f, "")
        println(f, "### Animations")
        println(f, "- `door_42.gif` - Door environment (seed 42)")
        println(f, "- `door_123.gif` - Door environment (seed 123)")
        println(f, "- `wall_42.gif` - Wall environment (seed 42)")
        println(f, "- `wall_123.gif` - Wall environment (seed 123)")
        println(f, "- `combined_42.gif` - Combined environment (seed 42)")
        println(f, "- `combined_123.gif` - Combined environment (seed 123)")
        println(f, "")
        println(f, "### Visualizations")
        println(f, "- `control_signals.gif` - Control signals for each agent")
        println(f, "- `control_magnitudes.png` - Plot of control signal magnitudes over time")
        println(f, "- `obstacle_distance.png` - Heatmap of distances to obstacles")
        println(f, "- `path_uncertainty.png` - Visualization of path uncertainties")
        println(f, "- `path_visualization.png` - Static visualization of agent paths")
        println(f, "- `convergence.png` - Convergence plot of the inference (may show placeholder if ELBO tracking unavailable)")
        println(f, "")
        println(f, "### Environment Heatmaps")
        println(f, "- `door_environment_heatmap.png` - Distance field visualization for door environment")
        println(f, "- `wall_environment_heatmap.png` - Distance field visualization for wall environment")
        println(f, "- `combined_environment_heatmap.png` - Distance field visualization for combined environment")
        println(f, "")
        println(f, "### Data Files")
        println(f, "- `paths.csv` - Raw path data (agent positions over time)")
        println(f, "- `controls.csv` - Raw control signals (agent control inputs)")
        println(f, "- `uncertainties.csv` - Path uncertainties (variance in agent positions)")
        println(f, "- `experiment.log` - Detailed log of experiment execution")
        println(f, "")
        println(f, "## Experiment Setup")
        println(f, "")
        println(f, "The experiments demonstrate multi-agent trajectory planning in three environments:")
        println(f, "1. **Door environment**: Two parallel walls with a gap between them")
        println(f, "2. **Wall environment**: A single wall obstacle in the center")
        println(f, "3. **Combined environment**: A combination of walls and obstacles")
        println(f, "")
        println(f, "Each experiment is run with 4 agents that need to navigate from their starting positions")
        println(f, "to their target positions while avoiding obstacles and other agents.")
        println(f, "")
        println(f, "## Implementation Details")
        println(f, "")
        println(f, "The trajectory planning is implemented using probabilistic inference with the RxInfer.jl framework.")
        println(f, "The agents follow a linear dynamical model with control inputs, and constraints are enforced")
        println(f, "through observations in the probabilistic model.")
        println(f, "")
        println(f, "For more details, see the `DOCUMENTATION.md` file in the project root.")
    end
    
    log_message("README created at $readme_file", logger)
end

end # module TrajectoryPlanning 