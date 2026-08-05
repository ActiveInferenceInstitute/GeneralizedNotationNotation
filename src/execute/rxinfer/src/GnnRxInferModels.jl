module GnnRxInferModels

using RxInfer
using Distributions
using LinearAlgebra

# Generative POMDP model for offline batch inference (Bayesian smoothing).
#
# Arguments: y (observations), A (likelihood), B (transition), D (prior),
# u (actions), T (timesteps). Hidden states evolve via DiscreteTransition
# conditioned on the previous state and selected action; observations are
# emitted via DiscreteTransition through the likelihood matrix A.
#
# NOTE: The posteriors from infer() on this model are smoothed (joint)
# posteriors from batch inference, not filtered (online) beliefs. EFE and
# policy selection are computed post-hoc in the generated script, not here.
@model function pomdp_model(y, A, B, D, u, T)
    s[1] ~ Categorical(D)
    y[1] ~ DiscreteTransition(s[1], A)
    for t in 2:T
        s[t] ~ DiscreteTransition(s[t-1], B[:, :, u[t-1]])
        y[t] ~ DiscreteTransition(s[t], A)
    end
end

# Precompile the model with workloads covering common GNN exemplar
# state-space sizes and T values. Julia specializes @model code per
# (n_states, n_obs, n_actions, T) tuple, so we precompile each common
# configuration to eliminate per-run JIT overhead.
#
# IMPORTANT: The precompile cache is machine-local and NOT portable across
# machines or Julia versions. It must be rebuilt on each machine via
# `julia --project=src/execute/rxinfer -e 'using PrecompileTools;
# PrecompileTools.workload()'` or by importing this module once.
#
# One-time cost: ~10-20 min for all configs. Runtime savings: ~80s per run.
using PrecompileTools: @compile_workload
@compile_workload begin
    # Cover common GNN exemplar dimensions: 2, 3, 4, 8, 9, 16 states
    # paired with their typical observation/action counts.
    configs = [
        (2, 2, 2),   # minimal binary models
        (3, 3, 3),   # actinf_pomdp_agent (3-state)
        (4, 4, 4),   # simple_mdp, hmm_baseline (4-state, most common)
        (8, 8, 4),   # larger gridworlds
        (9, 9, 5),   # 3x3 grid POMDPs
        (16, 16, 4), # large state spaces
    ]
    for (n_states, n_obs, n_actions) in configs
        A_pc = Matrix{Float64}(I, n_obs, n_states)
        # If n_obs != n_states, A must be n_obs x n_states
        if n_obs != n_states
            A_pc = fill(1.0 / n_states, n_obs, n_states)
        end
        B_pc = zeros(Float64, n_states, n_states, n_actions)
        for a in 1:n_actions
            B_pc[:, :, a] = fill(1.0 / n_states, n_states, n_states)
        end
        D_pc = fill(1.0 / n_states, n_states)
        for T_pc in [3, 5, 10, 15, 20, 25, 30]
            obs_pc = [[i == ((t - 1) % n_obs + 1) ? 1.0 : 0.0 for i in 1:n_obs] for t in 1:T_pc]
            acts_pc = fill(1, T_pc)
            try
                infer(
                    model = pomdp_model(A=A_pc, B=B_pc, D=D_pc, u=acts_pc, T=T_pc),
                    data = (y = obs_pc,),
                    iterations = 1,
                    free_energy = true
                )
                println("  precompile OK: $n_states states, $n_obs obs, $n_actions actions, T=$T_pc")
            catch e
                # Log the failure but do not abort — some configs may not
                # precompile cleanly (e.g. very large state spaces with
                # limited memory). The actual run-time infer() will still
                # crash if the model is genuinely broken.
                println("  precompile SKIP: $n_states states, $n_obs obs, $n_actions actions, T=$T_pc — $e")
            end
        end
    end
end

end # module
