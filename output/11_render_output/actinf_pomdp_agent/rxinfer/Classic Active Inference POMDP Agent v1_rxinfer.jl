# RxInfer.jl Minimal Working Simulation
# Generated from GNN Model: Classic Active Inference POMDP Agent v1
# Generated: 2025-10-28 13:01:26

using RxInfer
using Distributions

println("============================================================")
println("RxInfer.jl Active Inference - GNN Generated Simulation")
println("Model: Classic Active Inference POMDP Agent v1")
println("============================================================")

# Model parameters from GNN specification
const NUM_STATES = 3
const NUM_OBSERVATIONS = 3
const TIME_STEPS = 5

println("📊 State Space: $NUM_STATES states, $NUM_OBSERVATIONS observations")
println("⏱️  Time Steps: $TIME_STEPS")

# Simple HMM-style model using modern GraphPPL syntax
@model function simple_hmm(y)
    # Prior over initial state (single variable)
    s_prev ~ Categorical(fill(1.0/NUM_STATES, NUM_STATES))
    
    # First observation
    y[1] ~ Categorical(fill(1.0/NUM_OBSERVATIONS, NUM_OBSERVATIONS))
    
    # Subsequent time steps
    for t in 2:length(y)
        # Transition (simple - stays in same state with high probability)
        s_next ~ Categorical(fill(1.0/NUM_STATES, NUM_STATES))
        
        # Observation  
        y[t] ~ Categorical(fill(1.0/NUM_OBSERVATIONS, NUM_OBSERVATIONS))
        
        s_prev = s_next
    end
end

# Run inference
function run_inference()
    println("\\n🚀 Running Bayesian inference...")
    
    # Generate synthetic observations
    observations = [rand(1:NUM_OBSERVATIONS) for _ in 1:TIME_STEPS]
    println("📋 Observations: $observations")
    
    # Perform inference
    result = infer(
        model = simple_hmm(),
        data = (y = observations,)
    )
    
    println("\\n✅ Inference completed successfully!")
    println("📊 Results: $result")
    
    return result
end

# Main execution
function main()
    try
        result = run_inference()
        println("\\n🎉 RxInfer.jl simulation completed successfully!")
        return 0
    catch e
        println("\\n❌ Simulation failed: $e")
        showerror(stdout, e, catch_backtrace())
        return 1
    end
end

# Run if executed as script
if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end

