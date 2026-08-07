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

# --- Continuous state-space model (roadmap A2) ---
#
# Gaussian linear dynamical system for continuous POMDPs:
#   s[t] = F * s[t-1] + u[t-1] + noise (process noise Q)
#   y[t] = H * s[t] + noise            (observation noise R)
#
# Uses MvNormal nodes for true continuous inference. The prior s[1] is
# Gaussian with mean D_mean and covariance D_cov.
#
# NOTE: the linear-Gaussian composition MUST stay inline in the ~
# expression. GraphPPL only rewrites ~ / .~ / := statements; a plain
# local assignment like `mean_t = F * s[t-1]` executes as ordinary Julia
# at graph-construction time and throws MethodError on the variable
# labels. This matches RxInfer's own tested LGSSM pattern.
#
# Arguments:
#   y: observations (Vector of Vector{Float64})
#   F: state transition matrix (n_states x n_states)
#   H: observation matrix (n_obs x n_states)
#   Q: process noise covariance (n_states x n_states)
#   R: observation noise covariance (n_obs x n_obs)
#   D_mean: prior mean for s[1] (Vector)
#   D_cov: prior covariance for s[1] (Matrix)
#   u: control inputs (Vector of Vector{Float64}, u[t-1] applied at t)
#   T: number of timesteps
@model function continuous_pomdp_model(y, F, H, Q, R, D_mean, D_cov, u, T)
    s[1] ~ MvNormalMeanCovariance(D_mean, D_cov)
    y[1] ~ MvNormalMeanCovariance(H * s[1], R)
    for t in 2:T
        s[t] ~ MvNormalMeanCovariance(F * s[t-1] + u[t-1], Q)
        y[t] ~ MvNormalMeanCovariance(H * s[t], R)
    end
end

# --- Hierarchical POMDP model (roadmap A3) ---
#
# Two-level hierarchical model matching the semantics the hierarchical
# GNN exemplars actually declare (hierarchical_pomdp.md):
#   z: slow contextual state (single Categorical latent per episode)
#   s[t]: fast hidden state (action-driven chain over B)
#   Cross-level coupling: the context modulates the FAST-STATE PRIOR via
#   A_ctx (the exemplar's A_level2, n_fast x n_slow):
#     s[1] ~ DiscreteTransition(z, A_ctx)
#   Fast transitions are indexed by observed actions u (data), exactly
#   like the flat model — NOT by the latent z (indexing a tensor by a
#   latent NodeLabel is invalid at graph-construction time).
#
# Design notes (empirically validated against RxInfer 5.5):
# - The context is a SINGLE latent, not a per-timestep chain. The only
#   declared evidence channel for context is the fast-state prior at
#   episode start, so a within-episode z-chain is unidentifiable beyond
#   prior propagation AND its tail creates a half-edge RxInfer rejects
#   ("Half-edge has been found"). Context dynamics (the exemplar's
#   B_level2) are applied post-hoc in the generated script as
#   deterministic prior propagation of q(z), and the results label it
#   as propagation — it is never presented as inference output.
# - infer() on this model REQUIRES a mean-field constraint on the
#   coupling plus marginal initialization (see
#   hierarchical_constraints / hierarchical_initialization below):
#   without them, Bethe free-energy scoring of the non-square A_ctx
#   coupling hits ReactiveMP's square-matrix mul_trace assertion.
#
# Arguments:
#   y: observations (Vector of one-hot vectors)
#   A: fast likelihood matrix (n_obs x n_fast)
#   B: fast transition tensor (n_fast x n_fast x n_actions)
#   A_ctx: context-to-fast-prior map (n_fast x n_slow)
#   D_slow: prior over slow contextual states
#   u: action sequence (Vector{Int} data)
#   T: number of timesteps
@model function hierarchical_pomdp_model(y, A, B, A_ctx, D_slow, u, T)
    z ~ Categorical(D_slow)
    s[1] ~ DiscreteTransition(z, A_ctx)
    y[1] ~ DiscreteTransition(s[1], A)
    for t in 2:T
        s[t] ~ DiscreteTransition(s[t-1], B[:, :, u[t-1]])
        y[t] ~ DiscreteTransition(s[t], A)
    end
end

# Mean-field factorization over the context/fast coupling — required for
# free-energy scoring with a non-square A_ctx (see design notes above).
@constraints function hierarchical_constraints()
    q(z, s) = q(z)q(s)
end

# Uniform marginal initialization for the coupled latents — required so
# variational iteration has starting marginals on both sides of the
# mean-field cut.
@initialization function hierarchical_initialization(n_fast, n_slow)
    q(s) = Categorical(fill(1.0 / n_fast, n_fast))
    q(z) = Categorical(fill(1.0 / n_slow, n_slow))
end

# --- Factored mean-field POMDP model (roadmap D3) ---
#
# Native two-factor model matching the semantics the factored GNN
# exemplar actually declares (structured/factorized_posterior.md):
#   s1[t]: state factor 0 (location, n_f0=4) — action-driven chain over B_f0
#   s2[t]: state factor 1 (goal, n_f1=2) — passive chain over static B_f1
#   y_m0[t]: modality 0 (visual) depends on BOTH factors via the 3-tensor
#     A_m0 (n_obs_m0 x n_f0 x n_f1): the third DiscreteTransition argument
#     binds s2[t] to the T1 interface of the tensor (verified on RxInfer 5.5).
#   y_m1[t]: modality 1 (proprioceptive) depends on s1 only via A_m1.
#
# The exemplar's ## Equations declare the mean-field factorization
# Q(s_f0, s_f1) = Q(s_f0) Q(s_f1); factored_constraints() states exactly
# that cut, so the posterior family IS the declared model, not an
# approximation bolted on afterwards.
#
# Design notes (empirically validated against RxInfer 5.5):
# - infer() on this model REQUIRES BOTH the mean-field constraint AND the
#   uniform marginal initialization below. Verified negatives: with
#   neither, and with initialization but no constraint, infer() dies with
#   "Variables [ s1, s2 ] have not been updated after an update event".
# - Fast transitions are indexed by observed actions u (data),
#   B_f0[:, :, u[t-1]], exactly like the flat model.
# - The static factor uses the exemplar's exact identity B_f1 (contains
#   exact zeros); RxInfer handles it — free energy stays finite.
#
# Arguments:
#   y_m0: modality-0 observations (Vector of one-hot Vector{Float64}, dim n_obs_m0)
#   y_m1: modality-1 observations (Vector of one-hot Vector{Float64}, dim n_obs_m1)
#   A_m0: joint likelihood tensor (n_obs_m0 x n_f0 x n_f1)
#   A_m1: factor-0 likelihood matrix (n_obs_m1 x n_f0)
#   B_f0: factor-0 transition tensor (n_f0 x n_f0 x n_actions), canonical
#         (next_state, previous_state, action)
#   B_f1: factor-1 static transition matrix (n_f1 x n_f1), (next, prev)
#   D_f0: prior over factor 0 (Vector, dim n_f0)
#   D_f1: prior over factor 1 (Vector, dim n_f1)
#   u: action sequence (Vector{Int} data; u[t-1] applied at t)
#   T: number of timesteps
@model function factored_pomdp_model(y_m0, y_m1, A_m0, A_m1, B_f0, B_f1, D_f0, D_f1, u, T)
    s1[1] ~ Categorical(D_f0)
    s2[1] ~ Categorical(D_f1)
    y_m0[1] ~ DiscreteTransition(s1[1], A_m0, s2[1])
    y_m1[1] ~ DiscreteTransition(s1[1], A_m1)
    for t in 2:T
        s1[t] ~ DiscreteTransition(s1[t-1], B_f0[:, :, u[t-1]])
        s2[t] ~ DiscreteTransition(s2[t-1], B_f1)
        y_m0[t] ~ DiscreteTransition(s1[t], A_m0, s2[t])
        y_m1[t] ~ DiscreteTransition(s1[t], A_m1)
    end
end

# Mean-field factorization across the two state factors — this is the
# exemplar's declared Q(s_f0)Q(s_f1) posterior family, and it is also
# operationally required (see design notes above).
@constraints function factored_constraints()
    q(s1, s2) = q(s1)q(s2)
end

# Uniform marginal initialization for both factor chains — required so
# variational iteration has starting marginals on both sides of the
# mean-field cut.
@initialization function factored_initialization(n_f0, n_f1)
    q(s1) = Categorical(fill(1.0 / n_f0, n_f0))
    q(s2) = Categorical(fill(1.0 / n_f1, n_f1))
end

# --- Likelihood-learning POMDP model (roadmap D1) ---
#
# Identical chain to pomdp_model, but the likelihood matrix A is a LATENT
# DirichletCollection variable instead of a fixed constant. dirichlet_A
# carries the prior pseudo-counts (n_obs x n_states); DirichletCollection
# stores independent Dirichlet counts along the FIRST dimension, so each
# COLUMN dirichlet_A[:, s] is a Dirichlet over observations for state s —
# exactly matching the A[obs, state] orientation of DiscreteTransition(s, A).
#
# IMPORTANT (empirically validated on RxInfer 5.5):
# - infer() on this model REQUIRES the structured mean-field cut
#   q(s, A) = q(s)q(A) plus marginal initialization of BOTH q(A) and q(s);
#   without them variational iteration has no starting marginals on the cut.
# - The dirichlet_A prior must break column-permutation symmetry (e.g.
#   identity-biased counts: diag 3.0, off-diag 1.0). A fully uniform
#   ones(n,n) prior with uniform q(s) init converges to a label-switched
#   local optimum (verified: uniform prior moved q(A) AWAY from true A,
#   L1/9 distance 0.3556 -> 0.406; identity-biased prior converged
#   correctly, 0.1778 -> 0.0518).
@model function learning_pomdp_model(y, dirichlet_A, B, D, u, T)
    A ~ DirichletCollection(dirichlet_A)
    s[1] ~ Categorical(D)
    y[1] ~ DiscreteTransition(s[1], A)
    for t in 2:T
        s[t] ~ DiscreteTransition(s[t-1], B[:, :, u[t-1]])
        y[t] ~ DiscreteTransition(s[t], A)
    end
end

# Structured mean-field cut between the state chain and the likelihood
# tensor — required for tractable message passing on DiscreteTransition
# with a latent A (matches RxInfer 5.5's own HMM test:
# ~/.julia/packages/RxInfer/ssyz5/test/models/statespace/hmm_tests.jl).
@constraints function learning_constraints()
    q(s, A) = q(s)q(A)
end

# Marginal initialization: q(A) starts at the prior counts; q(s) starts
# uniform so the first VMP sweep has marginals on both sides of the cut.
@initialization function learning_initialization(prior_A, n_states)
    q(A) = DirichletCollection(prior_A)
    q(s) = Categorical(fill(1.0 / n_states, n_states))
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

    # Precompile continuous model with a small config. NO try/catch: a
    # broken model must fail package precompilation loudly (no-fallback
    # doctrine) — a swallowed failure here would be the model's only
    # execution and would mask breakage forever.
    let
        F_pc = Matrix{Float64}(I, 2, 2)
        H_pc = Matrix{Float64}(I, 2, 2)
        Q_pc = 0.1 .* Matrix{Float64}(I, 2, 2)
        R_pc = 0.1 .* Matrix{Float64}(I, 2, 2)
        D_mean_pc = zeros(2)
        D_cov_pc = Matrix{Float64}(I, 2, 2)
        u_pc = [zeros(2) for _ in 1:4]
        y_pc = [[0.1, -0.1] .* t for t in 1:5]
        infer(
            model = continuous_pomdp_model(F=F_pc, H=H_pc, Q=Q_pc, R=R_pc,
                                           D_mean=D_mean_pc, D_cov=D_cov_pc,
                                           u=u_pc, T=5),
            data = (y = y_pc,),
            iterations = 1,
            free_energy = true
        )
        println("  precompile OK: continuous model (2D, T=5)")
    end

    # Precompile hierarchical model with a small config (same no-try/catch
    # rationale as above).
    let
        A_h = fill(0.25, 4, 4)
        B_h = zeros(4, 4, 3)
        for a in 1:3
            B_h[:, :, a] = fill(0.25, 4, 4)
        end
        A_ctx_h = [0.7 0.1; 0.1 0.7; 0.1 0.1; 0.1 0.1]
        A_ctx_h = A_ctx_h ./ sum(A_ctx_h, dims = 1)
        D_slow_h = [0.5, 0.5]
        u_h = [1, 2, 3, 1]
        obs_h = [[1.0, 0.0, 0.0, 0.0] for _ in 1:5]
        infer(
            model = hierarchical_pomdp_model(A=A_h, B=B_h, A_ctx=A_ctx_h,
                                             D_slow=D_slow_h, u=u_h, T=5),
            data = (y = obs_h,),
            constraints = hierarchical_constraints(),
            initialization = hierarchical_initialization(4, 2),
            iterations = 2,
            free_energy = true
        )
        println("  precompile OK: hierarchical model (4 fast, 2 slow, T=5)")
    end

    # Precompile factored model with a small config (no try/catch — a broken
    # model must fail package precompilation loudly, no-fallback doctrine).
    let
        A_m0_pc = zeros(3, 4, 2)
        for i in 1:4, j in 1:2
            A_m0_pc[:, i, j] = fill(1.0 / 3.0, 3)
        end
        A_m1_pc = fill(0.5, 2, 4)
        B_f0_pc = zeros(4, 4, 3)
        for a in 1:3
            B_f0_pc[:, :, a] = fill(0.25, 4, 4)
        end
        B_f1_pc = Matrix{Float64}(I, 2, 2)
        u_pc = [1, 2, 3, 1, 2]
        y0_pc = [[1.0, 0.0, 0.0] for _ in 1:5]
        y1_pc = [[1.0, 0.0] for _ in 1:5]
        infer(
            model = factored_pomdp_model(A_m0=A_m0_pc, A_m1=A_m1_pc,
                                         B_f0=B_f0_pc, B_f1=B_f1_pc,
                                         D_f0=fill(0.25, 4), D_f1=[0.5, 0.5],
                                         u=u_pc, T=5),
            data = (y_m0 = y0_pc, y_m1 = y1_pc),
            constraints = factored_constraints(),
            initialization = factored_initialization(4, 2),
            iterations = 2,
            free_energy = true
        )
        println("  precompile OK: factored model (4x2 factors, T=5)")
    end

    # Precompile learning model with a small config (same no-try/catch rationale).
    let
        prior_pc = ones(3, 3) + 2.0 .* Matrix{Float64}(I, 3, 3)
        B_pc = zeros(3, 3, 2)
        for a in 1:2
            B_pc[:, :, a] = fill(1.0 / 3.0, 3, 3)
        end
        u_pc = [1, 2, 1, 2, 1]
        y_pc = [[1.0, 0.0, 0.0] for _ in 1:5]
        infer(
            model = learning_pomdp_model(dirichlet_A=prior_pc, B=B_pc,
                                         D=fill(1.0 / 3.0, 3), u=u_pc, T=5),
            data = (y = y_pc,),
            constraints = learning_constraints(),
            initialization = learning_initialization(prior_pc, 3),
            iterations = 2,
            free_energy = true
        )
        println("  precompile OK: learning model (3 states, T=5)")
    end
end

end # module
