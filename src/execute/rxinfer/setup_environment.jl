#!/usr/bin/env julia

"""
RxInfer.jl Environment Setup Script

This script sets up the Julia environment for RxInfer.jl simulations using
the committed Project.toml and Manifest.toml under src/execute/rxinfer/.
It replaces the previous runtime Pkg.add approach with a reproducible,
committed environment.

NOTE: The GnnRxInferModels precompile cache (built by importing the module
once) is machine-local and NOT portable across machines or Julia versions.
It must be rebuilt on each machine after setup.

Usage: julia setup_environment.jl [--verbose] [--validate-only]
"""

using Pkg
using Printf
using Dates
using Logging

const SETUP_VERSION = "2.0.0"

struct SetupConfig
    verbose::Bool
    validate_only::Bool
end

function parse_args()
    args = ARGS
    verbose = "--verbose" in args || "-v" in args
    validate_only = "--validate-only" in args || "--validate" in args
    return SetupConfig(verbose, validate_only)
end

function setup_logging(config::SetupConfig)
    log_level = config.verbose ? Logging.Debug : Logging.Info
    global_logger(ConsoleLogger(stderr, log_level))
end

function setup_project_environment(config::SetupConfig)
    @info "Setting up project environment from committed Project.toml..."
    project_dir = @__DIR__
    @info "Using project directory: $project_dir"

    try
        Pkg.activate(project_dir)
        Pkg.instantiate()
        @info "✅ Project environment instantiated from committed Project.toml + Manifest.toml"
        return true
    catch e
        @error "❌ Failed to instantiate project environment: $e"
        return false
    end
end

function validate_environment(config::SetupConfig)
    @info "Validating environment..."

    required_packages = ["RxInfer", "Distributions", "JSON", "StatsBase", "Plots"]
    validation_results = Dict{String, Bool}()

    for pkg in required_packages
        try
            @debug "Validating package: $pkg"
            eval(Meta.parse("using $pkg"))
            @debug "✅ Package $pkg loaded successfully"
            validation_results[pkg] = true
        catch e
            @warn "❌ Failed to load package $pkg: $e"
            validation_results[pkg] = false
        end
    end

    total = length(validation_results)
    valid_count = count(values(validation_results))
    @info "Validation summary: $valid_count/$total packages valid"

    if valid_count == total
        @info "✅ Environment validation passed"
        return true
    else
        failed = [pkg for (pkg, valid) in validation_results if !valid]
        @warn "⚠️  Failed packages: $(join(failed, ", "))"
        return false
    end
end

function run_setup(config::SetupConfig)
    @info "="^60
    @info "RxInfer.jl Environment Setup v$SETUP_VERSION"
    @info "="^60
    @info "Julia version: $(VERSION)"
    @info "Date: $(now())"
    @info ""

    if !setup_project_environment(config)
        return false
    end

    if config.validate_only
        @info "🔍 Running validation only..."
        return validate_environment(config)
    end

    @info "🔍 Running final validation..."
    return validate_environment(config)
end

function main()
    try
        config = parse_args()
        setup_logging(config)
        success = run_setup(config)
        exit(success ? 0 : 1)
    catch e
        @error "Fatal error during setup: $e"
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
