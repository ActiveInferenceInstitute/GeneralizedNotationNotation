#!/usr/bin/env julia

"""
Enhanced Visualization Module for ActiveInference.jl

This module provides comprehensive visualization capabilities beyond basic plotting:
- Interactive time series plots with zoom and pan capabilities
- 3D belief space visualizations and trajectory plotting
- Network diagrams for model structure and state transitions
- Heatmaps and matrix visualizations for A and B matrices
- Statistical distribution plots and confidence intervals
- Comparative analysis plots for model comparison
- Animation capabilities for temporal dynamics
- Export to multiple formats (PNG, SVG, PDF, HTML)
"""

using Pkg
using DelimitedFiles
using Statistics
using LinearAlgebra
using Printf
using Dates
using Colors
using ColorSchemes
using Distributions

# Enhanced plotting packages
for pkg in ["Plots", "PlotlyJS", "StatsPlots", "GraphPlot", "NetworkLayout", "LightGraphs"]
    try
        Pkg.add(pkg)
    catch e
        @warn "Failed to install $pkg: $e"
    end
end

using Plots
plotlyjs()  # Use PlotlyJS backend for interactivity

# Initialize enhanced plotting availability
ENHANCED_PLOTTING_AVAILABLE = false

try
    using StatsPlots
    using GraphPlot
    using NetworkLayout
    using LightGraphs
    ENHANCED_PLOTTING_AVAILABLE = true
catch
    @warn "Enhanced plotting packages not available, using basic functionality"
    ENHANCED_PLOTTING_AVAILABLE = false
end

# ====================================
# UTILITY FUNCTIONS
# ====================================

"""Create a comprehensive color palette for visualizations."""
function create_color_palette(n_colors::Int)
    if n_colors <= 10
        return palette(:tab10)[1:n_colors]
    elseif n_colors <= 20
        return palette(:tab20)[1:n_colors]
    else
        return collect(palette(:viridis, n_colors))
    end
end

"""Load and validate data from CSV files."""
function load_and_validate_data(filepath::String, required_cols::Int)
    if !isfile(filepath)
        error("Required data file not found: $filepath")
    end
    
    fileinfo = stat(filepath)
    if fileinfo.size < 10
        error("Data file is empty or too small: $filepath")
    end
    
    data = readdlm(filepath, ',', skipstart=6)  # Skip metadata headers
    if size(data, 1) == 0 || size(data, 2) < required_cols
        error("Data file $filepath is empty or missing required columns")
    end
    
    # Convert to numeric
    numeric_data = zeros(Float64, size(data, 1), size(data, 2))
    for i in 1:size(data, 1), j in 1:size(data, 2)
        try
            numeric_data[i, j] = parse(Float64, string(data[i, j]))
        catch
            numeric_data[i, j] = 0.0
        end
    end
    
    println("✅ Loaded data: $filepath ($(size(numeric_data, 1)) rows, $(size(numeric_data, 2)) cols)")
    return numeric_data
end

"""Create output directory with timestamp."""
function setup_viz_output_dir(base_dir::String)
    viz_dir = joinpath(base_dir, "enhanced_visualizations")
    mkpath(viz_dir)
    
    # Create subdirectories
    for subdir in ["interactive", "3d", "networks", "matrices", "distributions", "comparisons", "animations"]
        mkpath(joinpath(viz_dir, subdir))
    end
    
    return viz_dir
end

# ====================================
# INTERACTIVE TIME SERIES PLOTS
# ====================================

"""Create interactive time series plot with multiple traces."""
function create_interactive_timeseries(data_dict::Dict{String, Matrix{Float64}}, 
                                     output_dir::String, 
                                     title::String = "Interactive Time Series")
    
    interactive_dir = joinpath(output_dir, "interactive")
    
    # Create subplots for different data types
    plots_list = []
    
    for (name, data) in data_dict
        if size(data, 2) >= 2  # Ensure we have at least time and one data column
            time_col = data[:, 1]
            
            if size(data, 2) == 2
                # Single trace
                p = plot(time_col, data[:, 2], 
                        title="$name over Time",
                        xlabel="Time Step",
                        ylabel=name,
                        linewidth=2,
                        legend=false)
            else
                # Multiple traces
                p = plot(title="$name over Time", xlabel="Time Step", ylabel=name)
                colors = create_color_palette(size(data, 2) - 1)
                
                for i in 2:size(data, 2)
                    plot!(p, time_col, data[:, i], 
                         label="$(name)_$(i-1)",
                         linewidth=2,
                         color=colors[i-1])
                end
            end
            
            push!(plots_list, p)
        end
    end
    
    # Combine into dashboard
    if length(plots_list) > 0
        combined_plot = plot(plots_list..., 
                           layout=(length(plots_list), 1),
                           size=(800, 200*length(plots_list)),
                           title=title)
        
        # Save as interactive HTML
        savefig(combined_plot, joinpath(interactive_dir, "interactive_timeseries.html"))
        
        # Also save as static PNG
        savefig(combined_plot, joinpath(interactive_dir, "interactive_timeseries.png"))
        
        println("📊 Created interactive time series: $(joinpath(interactive_dir, "interactive_timeseries.html"))")
        return combined_plot
    end
    
    return nothing
end

"""Create correlation heatmap of multiple time series."""
function create_correlation_heatmap(data_matrix::Matrix{Float64}, 
                                  labels::Vector{String},
                                  output_dir::String)
    
    interactive_dir = joinpath(output_dir, "interactive")
    
    # Calculate correlation matrix
    corr_matrix = cor(data_matrix, dims=1)
    
    # Create heatmap
    p = heatmap(corr_matrix,
               xticks=(1:length(labels), labels),
               yticks=(1:length(labels), labels),
               title="Correlation Matrix",
               color=collect(palette(:RdBu, size(corr_matrix,1))),
               aspect_ratio=:equal)
    
    # Add correlation values as text
    for i in 1:size(corr_matrix, 1)
        for j in 1:size(corr_matrix, 2)
            annotate!(p, j, i, text(string(round(corr_matrix[i, j], digits=2)), 10, :white))
        end
    end
    
    savefig(p, joinpath(interactive_dir, "correlation_heatmap.html"))
    savefig(p, joinpath(interactive_dir, "correlation_heatmap.png"))
    
    println("🔥 Created correlation heatmap: $(joinpath(interactive_dir, "correlation_heatmap.html"))")
    return p
end

# ====================================
# 3D VISUALIZATIONS
# ====================================

"""Create 3D belief space trajectory."""
function create_3d_belief_trajectory(beliefs_data::Matrix{Float64}, 
                                   output_dir::String)
    
    viz_3d_dir = joinpath(output_dir, "3d")
    
    n_steps, n_states = size(beliefs_data)
    
    if n_states >= 3
        # Use first 3 states as coordinates
        x = beliefs_data[:, 1]
        y = beliefs_data[:, 2] 
        z = beliefs_data[:, 3]
        
        # Create 3D trajectory plot
        p = plot3d(x, y, z,
                  title="3D Belief Space Trajectory",
                  xlabel="Belief State 1",
                  ylabel="Belief State 2", 
                  zlabel="Belief State 3",
                  linewidth=3,
                  marker=:circle,
                  markersize=2,
                  color=:blue,
                  legend=false)
        
        # Add start and end markers
        scatter3d!(p, [x[1]], [y[1]], [z[1]], 
                  markersize=8, color=:green, label="Start")
        scatter3d!(p, [x[end]], [y[end]], [z[end]], 
                  markersize=8, color=:red, label="End")
        
        savefig(p, joinpath(viz_3d_dir, "belief_trajectory_3d.html"))
        savefig(p, joinpath(viz_3d_dir, "belief_trajectory_3d.png"))
        
        println("🌐 Created 3D belief trajectory: $(joinpath(viz_3d_dir, "belief_trajectory_3d.html"))")
        
    elseif n_states == 2
        # Create 2D trajectory with time as color
        time_steps = 1:n_steps
        
        p = scatter(beliefs_data[:, 1], beliefs_data[:, 2],
                   title="2D Belief Space Trajectory",
                   xlabel="Belief State 1",
                   ylabel="Belief State 2",
                   color=:blue,
                   colorbar_title="Time Step",
                   markersize=4,
                   legend=false)
        
        # Add trajectory line
        plot!(p, beliefs_data[:, 1], beliefs_data[:, 2], 
             linewidth=2, color=:gray, alpha=0.5)
        
        # Add start and end markers
        scatter!(p, [beliefs_data[1, 1]], [beliefs_data[1, 2]], 
                markersize=10, color=:green, label="Start")
        scatter!(p, [beliefs_data[end, 1]], [beliefs_data[end, 2]], 
                markersize=10, color=:red, label="End")
        
        savefig(p, joinpath(viz_3d_dir, "belief_trajectory_2d.html"))
        savefig(p, joinpath(viz_3d_dir, "belief_trajectory_2d.png"))
        
        println("📍 Created 2D belief trajectory: $(joinpath(viz_3d_dir, "belief_trajectory_2d.html"))")
    end
end

"""Create 3D surface plot of free energy landscape."""
function create_free_energy_surface(X::Matrix{Float64}, Y::Matrix{Float64}, Z::Matrix{Float64}, output_dir::String)
    viz_3d_dir = joinpath(output_dir, "3d")
    p = surface(X, Y, Z,
               title="Free Energy Landscape",
               xlabel="Belief State 1",
               ylabel="Belief State 2",
               zlabel="Free Energy",
               alpha=0.8)
    savefig(p, joinpath(viz_3d_dir, "free_energy_surface.html"))
    savefig(p, joinpath(viz_3d_dir, "free_energy_surface.png"))
    println("🏔️  Created free energy surface: $(joinpath(viz_3d_dir, "free_energy_surface.html"))")
    return p
end

# ====================================
# NETWORK DIAGRAMS
# ====================================

"""Create state transition network diagram."""
function create_transition_network(B_matrix::Array{Float64, 3}, 
                                 output_dir::String)
    
    networks_dir = joinpath(output_dir, "networks")
    
    if !ENHANCED_PLOTTING_AVAILABLE
        @warn "Enhanced plotting not available, skipping network diagrams"
        return
    end
    
    n_states = size(B_matrix, 1)
    n_actions = size(B_matrix, 3)
    
    # Create separate networks for each action
    for action in 1:n_actions
        # Extract transition matrix for this action
        transition_matrix = B_matrix[:, :, action]
        
        # Create graph
        g = SimpleDiGraph(n_states)
        
        # Add edges with weights above threshold
        threshold = 0.1
        edge_weights = Float64[]
        
        for i in 1:n_states
            for j in 1:n_states
                if transition_matrix[i, j] > threshold
                    add_edge!(g, i, j)
                    push!(edge_weights, transition_matrix[i, j])
                end
            end
        end
        
        # Create layout
        layout = spring_layout(g)
        
        # Plot network
        p = gplot(g, layout,
                 nodelabel=1:n_states,
                 nodesize=0.3,
                 edgelinewidth=edge_weights.*5,
                 arrowlengthfrac=0.1)
        
        # Save plot (note: gplot may not support all save formats)
        try
            savefig(p, joinpath(networks_dir, "transition_network_action_$action.png"))
            println("🕸️  Created transition network for action $action")
        catch e
            @warn "Failed to save network plot for action $action: $e"
        end
    end
end

"""Create belief update flow diagram."""
function create_belief_flow_diagram(beliefs_trace::Matrix{Float64},
                                  observations_trace::Vector{Int},
                                  output_dir::String)
    
    networks_dir = joinpath(output_dir, "networks") 
    
    # Create simplified flow diagram showing major belief updates
    n_steps, n_states = size(beliefs_trace)
    
    # Find significant belief updates (large changes)
    significant_updates = Int[]
    threshold = 0.1
    
    for t in 2:n_steps
        belief_change = norm(beliefs_trace[t, :] - beliefs_trace[t-1, :])
        if belief_change > threshold
            push!(significant_updates, t)
        end
    end
    
    # Create flow visualization
    p = plot(title="Belief Update Flow", 
            xlabel="Time Step",
            ylabel="State",
            legend=false)
    
    # Plot belief evolution for each state
    colors = create_color_palette(n_states)
    for s in 1:n_states
        plot!(p, 1:n_steps, beliefs_trace[:, s],
             linewidth=2,
             color=colors[s],
             label="State $s")
    end
    
    # Mark significant updates
    for update_time in significant_updates
        vline!(p, [update_time], 
              linestyle=:dash, 
              color=:red, 
              alpha=0.5)
        
        # Annotate with observation
        if update_time <= length(observations_trace)
            annotate!(p, update_time, 0.9, 
                     text("obs=$(observations_trace[update_time])", 8))
        end
    end
    
    savefig(p, joinpath(networks_dir, "belief_flow_diagram.html"))
    savefig(p, joinpath(networks_dir, "belief_flow_diagram.png"))
    
    println("🌊 Created belief flow diagram: $(joinpath(networks_dir, "belief_flow_diagram.html"))")
    return p
end

# ====================================
# MATRIX VISUALIZATIONS
# ====================================

"""Create comprehensive matrix visualization suite."""
function create_matrix_visualizations(A_matrix::Matrix{Float64},
                                    B_matrices::Array{Float64, 3},
                                    output_dir::String)
    
    matrices_dir = joinpath(output_dir, "matrices")
    
    # A matrix (observation model) heatmap
    p_A = heatmap(A_matrix,
                 title="Observation Model (A Matrix)",
                 xlabel="State", 
                 ylabel="Observation",
                 color=collect(palette(:viridis, size(A_matrix,1))),
                 aspect_ratio=:equal)
    
    # Add values as text annotations
    for i in 1:size(A_matrix, 1)
        for j in 1:size(A_matrix, 2)
            annotate!(p_A, j, i, text(string(round(A_matrix[i, j], digits=2)), 8))
        end
    end
    
    savefig(p_A, joinpath(matrices_dir, "A_matrix_heatmap.html"))
    savefig(p_A, joinpath(matrices_dir, "A_matrix_heatmap.png"))
    
    # B matrices (transition models) for each action
    n_actions = size(B_matrices, 3)
    B_plots = []
    
    for action in 1:n_actions
        B_action = B_matrices[:, :, action]
        
        p_B = heatmap(B_action,
                     title="Transition Model Action $action",
                     xlabel="Current State",
                     ylabel="Next State", 
                     color=collect(palette(:plasma, size(B_action,1))),
                     aspect_ratio=:equal)
        
        # Add values as text
        for i in 1:size(B_action, 1)
            for j in 1:size(B_action, 2)
                annotate!(p_B, j, i, text(string(round(B_action[i, j], digits=2)), 8))
            end
        end
        
        savefig(p_B, joinpath(matrices_dir, "B_matrix_action_$(action)_heatmap.png"))
        push!(B_plots, p_B)
    end
    
    # Create combined B matrices plot
    if length(B_plots) > 0
        combined_B = plot(B_plots..., 
                         layout=(1, length(B_plots)),
                         size=(300*length(B_plots), 300))
        
        savefig(combined_B, joinpath(matrices_dir, "B_matrices_combined.html"))
        savefig(combined_B, joinpath(matrices_dir, "B_matrices_combined.png"))
    end
    
    println("🔲 Created matrix visualizations in: $matrices_dir")
    return p_A, B_plots
end

"""Create eigenvalue/eigenvector analysis of transition matrices."""
function create_eigenanalysis_plots(B_matrices::Array{Float64, 3},
                                  output_dir::String)
    
    matrices_dir = joinpath(output_dir, "matrices")
    n_actions = size(B_matrices, 3)
    
    eigenvalue_plots = []
    
    for action in 1:n_actions
        B_action = B_matrices[:, :, action]
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = eigen(B_action)
        
        # Plot eigenvalues in complex plane
        p_eig = scatter(real(eigenvals), imag(eigenvals),
                       title="Eigenvalues - Action $action",
                       xlabel="Real Part",
                       ylabel="Imaginary Part", 
                       markersize=8,
                       legend=false)
        
        # Add unit circle
        theta = range(0, 2π, length=100)
        plot!(p_eig, cos.(theta), sin.(theta), 
             linestyle=:dash, color=:gray, alpha=0.5)
        
        push!(eigenvalue_plots, p_eig)
        
        savefig(p_eig, joinpath(matrices_dir, "eigenvalues_action_$action.png"))
    end
    
    # Combined eigenvalue plot
    if length(eigenvalue_plots) > 0
        combined_eigen = plot(eigenvalue_plots...,
                             layout=(1, length(eigenvalue_plots)),
                             size=(300*length(eigenvalue_plots), 300))
        
        savefig(combined_eigen, joinpath(matrices_dir, "eigenvalues_combined.png"))
    end
    
    println("👁️  Created eigenanalysis plots in: $matrices_dir")
    return eigenvalue_plots
end

# ====================================
# STATISTICAL DISTRIBUTION PLOTS
# ====================================

"""Create distribution analysis plots."""
function create_distribution_plots(data_dict::Dict{String, Vector{Float64}},
                                 output_dir::String)
    
    distributions_dir = joinpath(output_dir, "distributions")
    
    dist_plots = []
    
    for (name, data) in data_dict
        # Histogram with density overlay
        p = histogram(data,
                     title="Distribution of $name",
                     xlabel=name,
                     ylabel="Frequency",
                     normalize=:probability,
                     alpha=0.7,
                     legend=false)
        
        # Add statistical information
        mean_val = mean(data)
        std_val = std(data)
        median_val = median(data)
        
        # Add vertical lines for statistics
        vline!(p, [mean_val], linewidth=2, color=:red, label="Mean")
        vline!(p, [median_val], linewidth=2, color=:blue, label="Median")
        vline!(p, [mean_val - std_val, mean_val + std_val], 
              linewidth=1, color=:orange, linestyle=:dash, label="±1σ")
        
        # Add text annotation
        annotate!(p, mean_val, maximum(p.series_list[1].plotattributes[:y]) * 0.8,
                 text("μ=$(round(mean_val, digits=3))\nσ=$(round(std_val, digits=3))", 10))
        
        push!(dist_plots, p)
        savefig(p, joinpath(distributions_dir, "$(name)_distribution.png"))
    end
    
    # Create combined distributions plot
    if length(dist_plots) > 0
        combined_dist = plot(dist_plots...,
                           layout=(length(dist_plots), 1),
                           size=(600, 200*length(dist_plots)))
        
        savefig(combined_dist, joinpath(distributions_dir, "distributions_combined.html"))
        savefig(combined_dist, joinpath(distributions_dir, "distributions_combined.png"))
    end
    
    println("📊 Created distribution plots in: $distributions_dir")
    return dist_plots
end

"""Create Q-Q plots for normality testing."""
function create_qq_plots(data_dict::Dict{String, Vector{Float64}},
                        output_dir::String)
    
    distributions_dir = joinpath(output_dir, "distributions")
    
    qq_plots = []
    
    for (name, data) in data_dict
        # Create Q-Q plot against normal distribution
        sorted_data = sort(data)
        n = length(data)
        theoretical_quantiles = [quantile(Normal(0, 1), (i - 0.5) / n) for i in 1:n]
        
        # Standardize data
        standardized_data = (sorted_data .- mean(data)) ./ std(data)
        
        p = scatter(theoretical_quantiles, standardized_data,
                   title="Q-Q Plot: $name vs Normal",
                   xlabel="Theoretical Quantiles",
                   ylabel="Sample Quantiles",
                   legend=false,
                   markersize=3)
        
        # Add reference line
        min_q, max_q = extrema(theoretical_quantiles)
        plot!(p, [min_q, max_q], [min_q, max_q], 
             color=:red, linewidth=2, linestyle=:dash)
        
        push!(qq_plots, p)
        savefig(p, joinpath(distributions_dir, "$(name)_qqplot.png"))
    end
    
    println("📈 Created Q-Q plots in: $distributions_dir")
    return qq_plots
end

# ====================================
# COMPARATIVE ANALYSIS PLOTS
# ====================================

"""Create model comparison visualizations."""
function create_comparison_plots(models_data::Dict{String, Dict{String, Vector{Float64}}},
                               output_dir::String)
    
    comparisons_dir = joinpath(output_dir, "comparisons")
    
    # Box plots for comparing distributions across models
    comparison_plots = []
    
    # Get all metric names
    all_metrics = Set{String}()
    for model_data in values(models_data)
        union!(all_metrics, keys(model_data))
    end
    
    for metric in all_metrics
        # Collect data for this metric across models
        metric_data = []
        model_names = String[]
        
        for (model_name, model_data) in models_data
            if haskey(model_data, metric)
                append!(metric_data, model_data[metric])
                append!(model_names, fill(model_name, length(model_data[metric])))
            end
        end
        
        if !isempty(metric_data)
            # Create box plot
            p = boxplot(model_names, metric_data,
                       title="Comparison: $metric",
                       xlabel="Model",
                       ylabel=metric,
                       legend=false)
            
            push!(comparison_plots, p)
            savefig(p, joinpath(comparisons_dir, "comparison_$(metric).png"))
        end
    end
    
    # Create combined comparison plot
    if length(comparison_plots) > 0
        combined_comparison = plot(comparison_plots...,
                                 layout=(length(comparison_plots), 1),
                                 size=(800, 200*length(comparison_plots)))
        
        savefig(combined_comparison, joinpath(comparisons_dir, "model_comparisons.html"))
        savefig(combined_comparison, joinpath(comparisons_dir, "model_comparisons.png"))
    end
    
    println("⚖️  Created comparison plots in: $comparisons_dir")
    return comparison_plots
end

"""Create performance evolution comparison."""
function create_performance_evolution_plot(performance_data::Dict{String, Matrix{Float64}},
                                         output_dir::String)
    
    comparisons_dir = joinpath(output_dir, "comparisons")
    
    p = plot(title="Performance Evolution Comparison",
            xlabel="Time/Episode",
            ylabel="Performance Metric")
    
    colors = create_color_palette(length(performance_data))
    
    for (i, (model_name, data)) in enumerate(performance_data)
        time_steps = data[:, 1]
        performance = data[:, 2]
        
        plot!(p, time_steps, performance,
             label=model_name,
             linewidth=2,
             color=colors[i])
        
        # Add confidence interval if multiple runs
        if size(data, 2) > 2
            std_error = data[:, 3]
            plot!(p, time_steps, performance .- std_error,
                 fillto=performance .+ std_error,
                 alpha=0.3,
                 color=colors[i],
                 label="")
        end
    end
    
    savefig(p, joinpath(comparisons_dir, "performance_evolution.html"))
    savefig(p, joinpath(comparisons_dir, "performance_evolution.png"))
    
    println("📈 Created performance evolution plot")
    return p
end

# ====================================
# ANIMATION CAPABILITIES
# ====================================

"""Create animated belief evolution."""
function create_belief_animation(beliefs_trace::Matrix{Float64},
                                output_dir::String)
    
    animations_dir = joinpath(output_dir, "animations")
    
    n_steps, n_states = size(beliefs_trace)
    
    # Create animation of belief evolution
    anim = Animation()
    
    for t in 1:min(n_steps, 100)  # Limit to 100 frames
        p = bar(1:n_states, beliefs_trace[t, :],
               title="Belief Evolution - Step $t",
               xlabel="State",
               ylabel="Belief Probability",
               ylim=(0, 1),
               legend=false,
               color=:blue)
        
        # Add probability values on bars
        for s in 1:n_states
            annotate!(p, s, beliefs_trace[t, s] + 0.05,
                     text(string(round(beliefs_trace[t, s], digits=3)), 8))
        end
        
        frame(anim, p)
    end
    
    try
        gif(anim, joinpath(animations_dir, "belief_evolution.gif"), fps=5)
        println("🎬 Created belief evolution animation: $(joinpath(animations_dir, "belief_evolution.gif"))")
    catch e
        @warn "Failed to create animation: $e"
    end
    
    return anim
end

# ====================================
# MAIN ENHANCED VISUALIZATION FUNCTION
# ====================================

"""Comprehensive enhanced visualization suite."""
function create_enhanced_visualizations(output_dir::String)
    viz_dir = setup_viz_output_dir(output_dir)
    
    println("\n🎨 Enhanced Visualization Suite")
    println("="^50)
    
    data_traces_dir = joinpath(output_dir, "data_traces")
    
    try
        # Load all available data
        data_files = Dict{String, Matrix{Float64}}()
        
        # Core simulation data
        if isfile(joinpath(data_traces_dir, "beliefs_over_time.csv"))
            data_files["beliefs"] = load_and_validate_data(joinpath(data_traces_dir, "beliefs_over_time.csv"), 2)
        end
        
        if isfile(joinpath(data_traces_dir, "actions_over_time.csv"))
            data_files["actions"] = load_and_validate_data(joinpath(data_traces_dir, "actions_over_time.csv"), 2)
        end
        
        if isfile(joinpath(data_traces_dir, "observations_over_time.csv"))
            data_files["observations"] = load_and_validate_data(joinpath(data_traces_dir, "observations_over_time.csv"), 2)
        end
        
        # Learning data
        if isfile(joinpath(data_traces_dir, "learning_curve.csv"))
            data_files["learning"] = load_and_validate_data(joinpath(data_traces_dir, "learning_curve.csv"), 2)
        end
        
        # Planning data
        if isfile(joinpath(data_traces_dir, "planning_rewards.csv"))
            data_files["planning_rewards"] = load_and_validate_data(joinpath(data_traces_dir, "planning_rewards.csv"), 2)
        end
        
        if isfile(joinpath(data_traces_dir, "planning_actions.csv"))
            data_files["planning_actions"] = load_and_validate_data(joinpath(data_traces_dir, "planning_actions.csv"), 2)
        end
        
        # Advanced analysis data
        advanced_dir = joinpath(output_dir, "advanced_analysis")
        if isfile(joinpath(advanced_dir, "belief_entropy_trace.csv"))
            data_files["entropy"] = load_and_validate_data(joinpath(advanced_dir, "belief_entropy_trace.csv"), 2)
        end
        
        if isfile(joinpath(advanced_dir, "information_gain_trace.csv"))
            data_files["info_gain"] = load_and_validate_data(joinpath(advanced_dir, "information_gain_trace.csv"), 2)
        end
        
        # ===== INTERACTIVE TIME SERIES =====
        println("📊 Creating interactive time series plots...")
        create_interactive_timeseries(data_files, viz_dir, "ActiveInference.jl Simulation Dashboard")
        
        # ===== 3D VISUALIZATIONS =====
        if haskey(data_files, "beliefs")
            println("🌐 Creating 3D visualizations...")
            println("  [DEBUG] About to call create_3d_belief_trajectory...")
            create_3d_belief_trajectory(data_files["beliefs"][:, 2:end], viz_dir)
            # Create belief animation
            println("🎬 Creating belief evolution animation...")
            println("  [DEBUG] About to call create_belief_animation...")
            create_belief_animation(data_files["beliefs"][:, 2:end], viz_dir)
            # Free energy surface (synthetic demo if not available)
            println("🏔️  Creating free energy surface...")
            println("  [DEBUG] About to call create_free_energy_surface...")
            x = collect(0:0.1:1)
            y = collect(0:0.1:1)
            X = repeat(x', length(y), 1)
            Y = repeat(y, 1, length(x))
            Z = reshape([sin(xi*π)*cos(yi*π) for (xi, yi) in zip(X[:], Y[:])], size(X))
            create_free_energy_surface(X, Y, Z, viz_dir)
        end
        
        # ===== NETWORK DIAGRAMS =====
        println("🕸️  Creating network diagrams...")
        if haskey(data_files, "beliefs") && haskey(data_files, "observations")
            println("  [DEBUG] About to call create_belief_flow_diagram...")
            create_belief_flow_diagram(data_files["beliefs"][:, 2:end],
                                     Int.(data_files["observations"][:, 2]),
                                     viz_dir)
        end
        # Transition network (synthetic demo if not available)
        println("🕸️  Creating transition network diagrams...")
        println("  [DEBUG] About to call create_transition_network...")
        n_states = 2
        n_actions = 2
        B_matrices = zeros(n_states, n_states, n_actions)
        B_matrices[:, :, 1] = [0.8 0.2; 0.2 0.8]
        B_matrices[:, :, 2] = [0.2 0.8; 0.8 0.2]
        create_transition_network(B_matrices, viz_dir)
        
        # ===== MATRIX VISUALIZATIONS =====
        println("🔲 Creating matrix visualizations...")
        println("  [DEBUG] About to call create_matrix_visualizations...")
        # Create synthetic matrices for demonstration (in real use, these would be loaded)
        n_states = 2
        n_obs = 2
        n_actions = 2
        
        A_matrix = [0.8 0.2; 0.2 0.8]  # Example observation model
        B_matrices = zeros(n_states, n_states, n_actions)
        B_matrices[:, :, 1] = [0.8 0.2; 0.2 0.8]  # Action 1
        B_matrices[:, :, 2] = [0.2 0.8; 0.8 0.2]  # Action 2
        
        create_matrix_visualizations(A_matrix, B_matrices, viz_dir)
        println("  [DEBUG] About to call create_eigenanalysis_plots...")
        create_eigenanalysis_plots(B_matrices, viz_dir)
        
        # ===== DISTRIBUTION ANALYSIS =====
        println("📊 Creating distribution plots...")
        println("  [DEBUG] About to call create_distribution_plots...")
        distribution_data = Dict{String, Vector{Float64}}()
        
        for (name, data) in data_files
            if size(data, 2) >= 2
                distribution_data[name] = data[:, 2]  # Use first data column
            end
        end
        
        if !isempty(distribution_data)
            create_distribution_plots(distribution_data, viz_dir)
            println("  [DEBUG] About to call create_qq_plots...")
            create_qq_plots(distribution_data, viz_dir)
        end
        
        # ===== CORRELATION ANALYSIS =====
        if haskey(data_files, "beliefs") && size(data_files["beliefs"], 2) > 2
            println("🔥 Creating correlation analysis...")
            println("  [DEBUG] About to call create_correlation_heatmap...")
            beliefs_matrix = data_files["beliefs"][:, 2:end]
            n_states = size(beliefs_matrix, 2)
            state_labels = ["State $i" for i in 1:n_states]
            
            create_correlation_heatmap(beliefs_matrix, state_labels, viz_dir)
        end
        
        # ===== COMPARISON ANALYSIS =====
        println("⚖️  Creating comparison plots...")
        println("  [DEBUG] About to call create_comparison_plots...")
        # Synthetic model comparison data
        models_data = Dict(
            "ModelA" => Dict("accuracy" => randn(20) .+ 0.8, "reward" => randn(20) .+ 10),
            "ModelB" => Dict("accuracy" => randn(20) .+ 0.7, "reward" => randn(20) .+ 9)
        )
        create_comparison_plots(models_data, viz_dir)
        println("  [DEBUG] About to call create_performance_evolution_plot...")
        # Synthetic performance evolution data
        performance_data = Dict(
            "ModelA" => hcat(1:20, cumsum(randn(20) .+ 1)),
            "ModelB" => hcat(1:20, cumsum(randn(20) .+ 0.8))
        )
        create_performance_evolution_plot(performance_data, viz_dir)
        
        # ===== CREATE SUMMARY DASHBOARD =====
        println("📋 Creating summary dashboard...")
        
        # Create comprehensive summary HTML
        summary_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ActiveInference.jl Enhanced Visualization Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; border: 1px solid #ccc; padding: 15px; }
                .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
                img { max-width: 100%; height: auto; }
                h1, h2 { color: #333; }
                .metrics { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>ActiveInference.jl Enhanced Visualization Dashboard</h1>
            <p>Generated on: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))</p>
            
            <div class="section">
                <h2>📊 Interactive Time Series</h2>
                <p>Interactive plots with zoom and pan capabilities.</p>
                <a href="interactive/interactive_timeseries.html">Open Interactive Dashboard</a>
            </div>
            
            <div class="section">
                <h2>🌐 3D Visualizations</h2>
                <div class="grid">
                    <div>
                        <h3>Belief Trajectory</h3>
                        <img src="3d/belief_trajectory_3d.png" alt="3D Belief Trajectory" />
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🔲 Matrix Analysis</h2>
                <div class="grid">
                    <div>
                        <h3>Observation Model (A Matrix)</h3>
                        <img src="matrices/A_matrix_heatmap.png" alt="A Matrix Heatmap" />
                    </div>
                    <div>
                        <h3>Transition Models (B Matrices)</h3>
                        <img src="matrices/B_matrices_combined.png" alt="B Matrices Combined" />
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Statistical Distributions</h2>
                <div class="grid">
                    <div>
                        <h3>Distribution Analysis</h3>
                        <img src="distributions/distributions_combined.png" alt="Distributions" />
                    </div>
                </div>
            </div>
            
            <div class="section">
                <h2>🎬 Animations</h2>
                <div>
                    <h3>Belief Evolution</h3>
                    <img src="animations/belief_evolution.gif" alt="Belief Evolution Animation" />
                </div>
            </div>
            
            <div class="section">
                <h2>🕸️ Network Analysis</h2>
                <div class="grid">
                    <div>
                        <h3>Belief Flow Diagram</h3>
                        <img src="networks/belief_flow_diagram.png" alt="Belief Flow" />
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        open(joinpath(viz_dir, "dashboard.html"), "w") do f
            write(f, summary_html)
        end
        
        println("✅ Enhanced visualizations completed successfully!")
        println("📁 Results saved to: $viz_dir")
        println("🌐 Open dashboard: $(joinpath(viz_dir, "dashboard.html"))")
        
    catch e
        println("❌ Error in enhanced visualization: $e")
        @warn "Enhanced visualization failed" exception=e
    end
end

"""Main entry point for enhanced visualization."""
function main()
    if length(ARGS) > 0
        output_dir = ARGS[1]
        if isdir(output_dir)
            create_enhanced_visualizations(output_dir)
        else
            error("❌ Directory not found: $output_dir")
        end
    else
        println("Usage: julia enhanced_visualization.jl <output_directory>")
        println("Example: julia enhanced_visualization.jl activeinference_outputs_YYYY-MM-DD_HH-MM-SS")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end 