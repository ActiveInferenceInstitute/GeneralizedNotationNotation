#!/usr/bin/env julia

"""
Test script for ActiveInference.jl POMDP Gridworld Simulation

This script tests the main POMDP gridworld simulation with a smaller
configuration to verify functionality.
"""

using Pkg
using Dates
using Logging
using DelimitedFiles
using Random
using Statistics
using LinearAlgebra
using ActiveInference
using Distributions
using JSON

# =============================================================================
# TEST CONFIGURATION - SMALLER GRIDWORLD FOR TESTING
# =============================================================================

# Gridworld Configuration
const GRID_SIZE = 3                    # 3x3 gridworld for testing
const START_POSITION = [1, 1]          # Starting position [row, col]
const GOAL_POSITION = [3, 3]           # Goal position [row, col]
const OBSTACLE_POSITIONS = [           # Minimal obstacles
    [2, 2]                             # Single obstacle in center
]

# POMDP State Space Configuration
const N_STATES = [GRID_SIZE * GRID_SIZE]    # Total number of grid positions
const N_OBSERVATIONS = [4]                  # 4 observation types: empty, wall, goal, obstacle
const N_CONTROLS = [4]                      # 4 actions: up, down, left, right
const POLICY_LENGTH = 2                     # Shorter planning horizon for testing

# Active Inference Agent Configuration
const ALPHA = 8.0                          # Precision parameter
const BETA = 1.0                           # Inverse temperature
const GAMMA = 1.0                          # Policy precision
const LAMBDA = 1.0                         # Learning rate
const OMEGA = 1.0                          # Evidence accumulation rate

# Simulation Configuration
const N_SIMULATION_STEPS = 15              # Shorter simulation for testing
const RANDOM_SEED = 42                     # Random seed for reproducibility

# Observation Model Configuration
const OBSERVATION_NOISE = 0.1              # Probability of incorrect observations
const WALL_DETECTION_RANGE = 1             # Range for wall detection
const GOAL_DETECTION_RANGE = 2             # Range for goal detection

# Reward Configuration
const GOAL_REWARD = 10.0                   # Reward for reaching goal
const STEP_COST = -0.1                     # Cost per step
const OBSTACLE_COST = -1.0                 # Cost for hitting obstacle
const WALL_COST = -0.5                     # Cost for hitting wall

# Output Configuration
const OUTPUT_DIRECTORY = "test_pomdp_outputs"
const SAVE_TRACES = true                   # Save detailed traces
const VERBOSE_LOGGING = false              # Reduce logging for testing

# =============================================================================
# UTILITY FUNCTIONS (copied from main script)
# =============================================================================

function setup_output_directories()
    """Create organized output directory structure with timestamp."""
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    output_dir = joinpath(OUTPUT_DIRECTORY, "test_gridworld_outputs_$timestamp")
    
    dirs = [
        "logs",
        "data_traces", 
        "models",
        "simulation_results",
        "analysis"
    ]
    
    for dir in dirs
        mkpath(joinpath(output_dir, dir))
    end
    
    return output_dir
end

function setup_logging(output_dir::String)
    """Setup comprehensive logging to file and console."""
    log_file = joinpath(output_dir, "logs", "test_pomdp_gridworld.log")
    logger = SimpleLogger(open(log_file, "w"))
    global_logger(logger)
    
    @info "ActiveInference.jl POMDP Gridworld Test Started" 
    @info "Configuration" grid_size=GRID_SIZE n_states=N_STATES n_observations=N_OBSERVATIONS
    @info "Output directory: $output_dir"
    
    return log_file
end

function save_data_with_metadata(data, filename::String, metadata::Dict)
    """Save data to CSV with comprehensive metadata headers."""
    open(filename, "w") do f
        println(f, "# ActiveInference.jl POMDP Gridworld Test Data Export")
        println(f, "# Generated: $(now())")
        println(f, "# Script Version: 1.0.0")
        for (key, value) in metadata
            println(f, "# $key: $value")
        end
        println(f, "# Data begins below:")
        
        if isa(data, AbstractMatrix)
            writedlm(f, data, ',')
        elseif isa(data, AbstractVector)
            writedlm(f, reshape(data, length(data), 1), ',')
        else
            println(f, string(data))
        end
    end
    @info "Data saved: $filename" rows=size(data, 1) cols=size(data, 2)
end

function save_trace_csv(data, filename::String, headers::Vector{String})
    """Save trace data to CSV with headers."""
    open(filename, "w") do f
        println(f, join(headers, ","))
        writedlm(f, data, ',')
    end
    @info "Trace saved: $filename" rows=size(data, 1) cols=size(data, 2)
end

# =============================================================================
# GRIDWORLD ENVIRONMENT FUNCTIONS (copied from main script)
# =============================================================================

function position_to_state(row::Int, col::Int)
    """Convert grid position to state index."""
    return (row - 1) * GRID_SIZE + col
end

function state_to_position(state::Int)
    """Convert state index to grid position."""
    row = div(state - 1, GRID_SIZE) + 1
    col = mod(state - 1, GRID_SIZE) + 1
    return [row, col]
end

function is_valid_position(row::Int, col::Int)
    """Check if position is within grid bounds."""
    return 1 <= row <= GRID_SIZE && 1 <= col <= GRID_SIZE
end

function is_obstacle(row::Int, col::Int)
    """Check if position contains an obstacle."""
    return [row, col] in OBSTACLE_POSITIONS
end

function is_goal(row::Int, col::Int)
    """Check if position is the goal."""
    return [row, col] == GOAL_POSITION
end

function get_observation(row::Int, col::Int)
    """Generate observation based on current position and surroundings."""
    if is_goal(row, col)
        return 3  # Goal observation
    elseif is_obstacle(row, col)
        return 4  # Obstacle observation
    else
        # Check for walls in detection range
        has_wall = false
        for r in max(1, row-WALL_DETECTION_RANGE):min(GRID_SIZE, row+WALL_DETECTION_RANGE)
            for c in max(1, col-WALL_DETECTION_RANGE):min(GRID_SIZE, col+WALL_DETECTION_RANGE)
                if !is_valid_position(r, c) || is_obstacle(r, c)
                    has_wall = true
                    break
                end
            end
        end
        
        if has_wall
            return 2  # Wall observation
        else
            return 1  # Empty observation
        end
    end
end

# =============================================================================
# POMDP MODEL CONSTRUCTION (copied from main script)
# =============================================================================

function create_gridworld_pomdp()
    """Create POMDP model matrices for the gridworld."""
    @info "Creating gridworld POMDP model"
    
    # Create matrix templates using ActiveInference.jl API
    A, B, C, D, E = create_matrix_templates(N_STATES, N_OBSERVATIONS, N_CONTROLS, POLICY_LENGTH)
    
    # Initialize A matrix (observation model)
    A[1] = zeros(4, GRID_SIZE * GRID_SIZE)
    
    # Fill A matrix based on gridworld structure
    for state in 1:(GRID_SIZE * GRID_SIZE)
        row, col = state_to_position(state)
        obs = get_observation(row, col)
        
        # Set observation probabilities with noise
        for obs_type in 1:4
            if obs_type == obs
                A[1][obs_type, state] = 1.0 - OBSERVATION_NOISE
            else
                A[1][obs_type, state] = OBSERVATION_NOISE / 3.0
            end
        end
    end
    
    # Initialize B matrices (transition model) for each action
    for action in 1:4
        B[action] = zeros(GRID_SIZE * GRID_SIZE, GRID_SIZE * GRID_SIZE)
        
        # Fill B matrix based on action effects
        for state in 1:(GRID_SIZE * GRID_SIZE)
            row, col = state_to_position(state)
            
            # Define action effects: 1=up, 2=down, 3=left, 4=right
            if action == 1  # Up
                new_row, new_col = row - 1, col
            elseif action == 2  # Down
                new_row, new_col = row + 1, col
            elseif action == 3  # Left
                new_row, new_col = row, col - 1
            else  # Right
                new_row, new_col = row, col + 1
            end
            
            # Check if new position is valid
            if is_valid_position(new_row, new_col) && !is_obstacle(new_row, new_col)
                new_state = position_to_state(new_row, new_col)
                B[action][new_state, state] = 1.0
            else
                # Stay in same position if invalid move
                B[action][state, state] = 1.0
            end
        end
    end
    
    # Initialize C vector (preferences)
    C = zeros(4)
    C[3] = GOAL_REWARD  # Prefer goal observations
    
    # Initialize D vector (prior beliefs about initial state)
    D = zeros(GRID_SIZE * GRID_SIZE)
    start_state = position_to_state(START_POSITION[1], START_POSITION[2])
    D[start_state] = 1.0
    
    # Initialize E vector (action preferences)
    E = zeros(4)  # No strong action preferences
    
    @info "POMDP model created" n_states=GRID_SIZE*GRID_SIZE n_observations=4 n_actions=4
    
    return A, B, C, D, E
end

# =============================================================================
# ACTIVE INFERENCE AGENT FUNCTIONS (copied from main script)
# =============================================================================

function create_active_inference_agent(A, B, C, D, E)
    """Create and configure Active Inference agent."""
    @info "Creating Active Inference agent"
    
    # Agent settings
    settings = Dict(
        "use_param_info_gain" => true,
        "policy_len" => POLICY_LENGTH,
        "modalities_to_learn" => [1],
        "verbose" => VERBOSE_LOGGING
    )
    
    # Agent parameters
    parameters = Dict(
        "alpha" => ALPHA,
        "beta" => BETA,
        "gamma" => GAMMA,
        "lambda" => LAMBDA,
        "omega" => OMEGA
    )
    
    # Initialize agent
    aif_agent = init_aif(A, B; C=C, D=D, E=E, settings=settings, parameters=parameters, verbose=false)
    
    @info "Active Inference agent initialized successfully"
    return aif_agent
end

function run_gridworld_simulation(aif_agent, output_dir::String)
    """Run the main gridworld simulation."""
    @info "Starting gridworld simulation" n_steps=N_SIMULATION_STEPS
    
    # Initialize tracking variables
    positions = []
    observations = []
    actions = []
    beliefs = []
    rewards = []
    steps = []
    
    # Set random seed for reproducibility
    Random.seed!(RANDOM_SEED)
    
    # Initialize position
    current_pos = copy(START_POSITION)
    push!(positions, current_pos)
    
    for step in 1:N_SIMULATION_STEPS
        @info "Simulation step $step" position=current_pos
        
        # Generate observation
        obs = get_observation(current_pos[1], current_pos[2])
        push!(observations, obs)
        
        # Agent inference and action selection
        infer_states!(aif_agent, [obs])
        infer_policies!(aif_agent)
        sample_action!(aif_agent)
        
        action = aif_agent.action[1]
        push!(actions, action)
        
        # Update position based on action
        new_pos = copy(current_pos)
        if action == 1  # Up
            new_pos[1] = max(1, new_pos[1] - 1)
        elseif action == 2  # Down
            new_pos[1] = min(GRID_SIZE, new_pos[1] + 1)
        elseif action == 3  # Left
            new_pos[2] = max(1, new_pos[2] - 1)
        else  # Right
            new_pos[2] = min(GRID_SIZE, new_pos[2] + 1)
        end
        
        # Check for obstacles
        if is_obstacle(new_pos[1], new_pos[2])
            new_pos = current_pos  # Stay in place
        end
        
        current_pos = new_pos
        push!(positions, current_pos)
        
        # Calculate reward
        reward = STEP_COST
        if is_goal(current_pos[1], current_pos[2])
            reward += GOAL_REWARD
        elseif is_obstacle(current_pos[1], current_pos[2])
            reward += OBSTACLE_COST
        end
        
        push!(rewards, reward)
        
        # Store belief state
        belief_state = aif_agent.qs_current[1]
        push!(beliefs, belief_state)
        push!(steps, step)
        
        # Log progress
        if step % 5 == 0
            @info "Step $step completed" position=current_pos action=action reward=reward
        end
        
        # Check if goal reached
        if is_goal(current_pos[1], current_pos[2])
            @info "Goal reached at step $step!"
            break
        end
    end
    
    # Save simulation results
    save_simulation_results(positions, observations, actions, beliefs, rewards, steps, output_dir)
    
    @info "Simulation completed" final_position=current_pos total_steps=length(steps)
    return positions, observations, actions, beliefs, rewards, steps
end

function save_simulation_results(positions, observations, actions, beliefs, rewards, steps, output_dir::String)
    """Save comprehensive simulation results."""
    @info "Saving simulation results"
    
    # Convert positions to state indices
    state_indices = [position_to_state(pos[1], pos[2]) for pos in positions]
    
    # Create main results matrix
    results_data = hcat(steps, state_indices[1:end-1], observations, actions, rewards)
    save_data_with_metadata(
        results_data,
        joinpath(output_dir, "simulation_results", "test_gridworld_simulation.csv"),
        Dict(
            "description" => "Test gridworld simulation: step, state, observation, action, reward",
            "n_steps" => length(steps),
            "grid_size" => GRID_SIZE,
            "start_position" => START_POSITION,
            "goal_position" => GOAL_POSITION
        )
    )
    
    # Save detailed traces
    if SAVE_TRACES
        trace_dir = joinpath(output_dir, "data_traces")
        
        # Position traces
        pos_data = hcat(steps, [pos[1] for pos in positions[1:end-1]], [pos[2] for pos in positions[1:end-1]])
        save_trace_csv(pos_data, joinpath(trace_dir, "test_positions_trace.csv"), ["step", "row", "col"])
        
        # Belief traces (first few states for readability)
        n_belief_states = min(5, length(beliefs[1]))
        belief_data = hcat(steps, [belief[1:n_belief_states] for belief in beliefs]...)
        save_trace_csv(belief_data, joinpath(trace_dir, "test_beliefs_trace.csv"), 
                      vcat(["step"], ["belief_state_$i" for i in 1:n_belief_states]))
        
        # Action and observation traces
        save_trace_csv(hcat(steps, actions), joinpath(trace_dir, "test_actions_trace.csv"), ["step", "action"])
        save_trace_csv(hcat(steps, observations), joinpath(trace_dir, "test_observations_trace.csv"), ["step", "observation"])
        save_trace_csv(hcat(steps, rewards), joinpath(trace_dir, "test_rewards_trace.csv"), ["step", "reward"])
    end
    
    @info "Simulation results saved successfully"
end

# =============================================================================
# ANALYSIS FUNCTIONS (copied from main script)
# =============================================================================

function analyze_simulation_results(positions, observations, actions, beliefs, rewards, steps, output_dir::String)
    """Analyze simulation results and generate reports."""
    @info "Analyzing simulation results"
    
    # Basic statistics
    total_reward = sum(rewards)
    mean_reward = mean(rewards)
    goal_reached = is_goal(positions[end][1], positions[end][2])
    unique_positions = length(unique(positions))
    
    # Action distribution
    action_counts = Dict()
    for action in actions
        action_counts[action] = get(action_counts, action, 0) + 1
    end
    
    # Observation distribution
    obs_counts = Dict()
    for obs in observations
        obs_counts[obs] = get(obs_counts, obs, 0) + 1
    end
    
    # Create analysis report
    analysis_file = joinpath(output_dir, "analysis", "test_simulation_analysis.txt")
    open(analysis_file, "w") do f
        println(f, "ActiveInference.jl POMDP Gridworld Test Analysis")
        println(f, "=" * 60)
        println(f, "Generated: $(now())")
        println(f, "")
        println(f, "SIMULATION OVERVIEW:")
        println(f, "  Total steps: $(length(steps))")
        println(f, "  Total reward: $(round(total_reward, digits=2))")
        println(f, "  Mean reward per step: $(round(mean_reward, digits=3))")
        println(f, "  Goal reached: $goal_reached")
        println(f, "  Unique positions visited: $unique_positions")
        println(f, "  Final position: $(positions[end])")
        println(f, "")
        println(f, "ACTION DISTRIBUTION:")
        for (action, count) in sort(collect(action_counts))
            action_name = ["Up", "Down", "Left", "Right"][action]
            percentage = round(count / length(actions) * 100, digits=1)
            println(f, "  $action_name (Action $action): $count times ($percentage%)")
        end
        println(f, "")
        println(f, "OBSERVATION DISTRIBUTION:")
        for (obs, count) in sort(collect(obs_counts))
            obs_name = ["Empty", "Wall", "Goal", "Obstacle"][obs]
            percentage = round(count / length(observations) * 100, digits=1)
            println(f, "  $obs_name (Obs $obs): $count times ($percentage%)")
        end
        println(f, "")
        println(f, "PERFORMANCE METRICS:")
        println(f, "  Efficiency: $(round(unique_positions / length(steps), digits=3))")
        println(f, "  Exploration ratio: $(round(unique_positions / (GRID_SIZE * GRID_SIZE), digits=3))")
        if goal_reached
            steps_to_goal = findfirst(pos -> is_goal(pos[1], pos[2]), positions)
            println(f, "  Steps to goal: $steps_to_goal")
        end
    end
    
    @info "Analysis completed" analysis_file=analysis_file
end

# =============================================================================
# MAIN TEST FUNCTION
# =============================================================================

function test_pomdp_gridworld()
    """Test the POMDP gridworld simulation."""
    println("🧪 Testing ActiveInference.jl POMDP Gridworld Simulation")
    println("=" * 60)
    
    try
        # Setup output directories and logging
        output_dir = setup_output_directories()
        log_file = setup_logging(output_dir)
        
        println("✅ Output directories and logging setup complete")
        
        # Create POMDP model
        A, B, C, D, E = create_gridworld_pomdp()
        println("✅ POMDP model created successfully")
        println("   - Grid size: $(GRID_SIZE)x$(GRID_SIZE)")
        println("   - States: $(N_STATES[1])")
        println("   - Observations: $(N_OBSERVATIONS[1])")
        println("   - Actions: $(N_CONTROLS[1])")
        
        # Create Active Inference agent
        aif_agent = create_active_inference_agent(A, B, C, D, E)
        println("✅ Active Inference agent created successfully")
        
        # Run simulation
        println("🔄 Running gridworld simulation...")
        positions, observations, actions, beliefs, rewards, steps = run_gridworld_simulation(aif_agent, output_dir)
        println("✅ Simulation completed successfully")
        println("   - Total steps: $(length(steps))")
        println("   - Final position: $(positions[end])")
        println("   - Goal reached: $(is_goal(positions[end][1], positions[end][2]))")
        
        # Analyze results
        analyze_simulation_results(positions, observations, actions, beliefs, rewards, steps, output_dir)
        println("✅ Analysis completed")
        
        # Display summary
        total_reward = sum(rewards)
        mean_reward = mean(rewards)
        goal_reached = is_goal(positions[end][1], positions[end][2])
        unique_positions = length(unique(positions))
        
        println("\n📊 SIMULATION SUMMARY:")
        println("   Total reward: $(round(total_reward, digits=2))")
        println("   Mean reward per step: $(round(mean_reward, digits=3))")
        println("   Goal reached: $goal_reached")
        println("   Unique positions visited: $unique_positions")
        println("   Final position: $(positions[end])")
        
        if goal_reached
            steps_to_goal = findfirst(pos -> is_goal(pos[1], pos[2]), positions)
            println("   Steps to goal: $steps_to_goal")
        end
        
        println("\n📁 Results saved to: $output_dir")
        println("✅ All tests passed!")
        
        return true
        
    catch e
        println("❌ Test failed: $e")
        @error "Test failed" exception=e
        return false
    end
end

# Run the test if this script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    success = test_pomdp_gridworld()
    exit(success ? 0 : 1)
end 