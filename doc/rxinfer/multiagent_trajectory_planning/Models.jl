"""
    Models

Multi-agent Trajectory Planning Models

This module serves as the main entry point for all model components by:
1. Integrating component modules (HalfspaceNode, DistanceFunctions, InferenceModel)
2. Re-exporting their functionality through a single interface
3. Providing a cleaner API for external modules to import

Components:
- HalfspaceNode: Custom node definition for probabilistic constraints
- DistanceFunctions: Functions for distance calculations and soft-min operations
- InferenceModel: Core probabilistic model and inference implementation

Note: This file does not contain model logic itself, but rather organizes and 
re-exports functionality from the component modules. The actual model implementation
is in InferenceModel.jl.
"""
module Models

# Include the modular components
include("HalfspaceNode.jl")
include("DistanceFunctions.jl")
include("InferenceModel.jl")

# Re-export all the components
using .HalfspaceNode
using .DistanceFunctions
using .InferenceModel

# Import and re-export visualization functions needed for ELBO plots
using ..Visualizations: plot_elbo_convergence

# Export all necessary functions and types
export Halfspace
export distance, g, h, softmin, configure_softmin
export path_planning, path_planning_model, path_planning_constraints, compute_diagnostics
export plot_elbo_convergence

end # module 