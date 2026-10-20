# Native RxInfer release checks; run under this directory's committed project.
using Test
using RxInfer
using GnnRxInferModels
using LinearAlgebra

const Models = GnnRxInferModels

@testset "Locked RxInfer release" begin
    @test pkgversion(RxInfer) == v"6.0.2"
    @test pkgversion(ReactiveMP).major == 7
end

# Exercise the custom node shipped with the standalone trajectory example.
include(joinpath(@__DIR__, "../../../../docs/rxinfer/multiagent_trajectory_planning/HalfspaceNode.jl"))
using .HalfspaceNode: Halfspace

@testset "Trajectory Halfspace rules use the RxInfer 6 API" begin
    forward = getresult(@call_message_update_rule(
        node=Halfspace, target=:out,
        q=(a=PointMass(1.0), σ2=PointMass(0.5), γ=PointMass(2.0)),
    ))
    @test mean(forward) ≈ 2.0
    @test var(forward) ≈ 0.5
    variance = getresult(@call_message_update_rule(
        node=Halfspace, target=:σ2,
        q=(out=NormalMeanVariance(3.0, 0.25), a=PointMass(1.0), γ=PointMass(2.0)),
    ))
    @test mean(variance) ≈ sqrt(4.25) / 2
end

@testset "Categorical smoothing agrees with exhaustive enumeration" begin
    A = [0.85 0.2; 0.15 0.8]
    B = reshape([0.7 0.25; 0.3 0.75], 2, 2, 1)
    D = [0.6, 0.4]
    observations = [1, 2, 2]
    y = [[Float64(k == o) for k in 1:2] for o in observations]
    result = infer(
        model = Models.pomdp_model(A=A, B=B, D=D, u=[1, 1], T=3),
        data = (y=y,), iterations=3, returnvars=KeepLast(), free_energy=true,
    )
    # Independent exact joint enumeration: includes evidence on BOTH sides of t.
    expected = zeros(2, 3)
    evidence = 0.0
    for states in Iterators.product(1:2, 1:2, 1:2)
        weight = D[states[1]] * prod(A[observations[t], states[t]] for t in 1:3)
        weight *= prod(B[states[t], states[t-1], 1] for t in 2:3)
        evidence += weight
        for t in 1:3
            expected[states[t], t] += weight
        end
    end
    expected ./= evidence
    @test length(result.posteriors[:s]) == 3
    for t in 1:3
        @test probvec(result.posteriors[:s][t]) ≈ expected[:, t] atol=1e-8
    end
    @test all(isfinite, result.free_energy)
    @test last(result.free_energy) ≈ -log(evidence) atol=1e-8
end

@testset "Gaussian posterior agrees with information-form conditioning" begin
    prior = [0.3, -0.2]
    P = [1.0 0.2; 0.2 0.7]
    H = [1.0 -0.4]
    R = fill(0.25, 1, 1)
    observation = [0.8]
    result = infer(
        model = Models.continuous_pomdp_model(
            F=Matrix{Float64}(I, 2, 2), H=H, Q=0.1 * Matrix{Float64}(I, 2, 2),
            R=R, D_mean=prior, D_cov=P, u=Vector{Float64}[], T=1,
        ),
        data=(y=[observation],), iterations=1, returnvars=KeepLast(), free_energy=true,
    )
    expected_cov = inv(inv(P) + H' * inv(R) * H)
    expected_mean = expected_cov * (P \ prior + H' * (R \ observation))
    @test mean(result.posteriors[:s][1]) ≈ expected_mean atol=1e-8
    @test cov(result.posteriors[:s][1]) ≈ expected_cov atol=1e-8
    @test all(isfinite, result.free_energy)
end

@testset "Structured model families keep finite native inference" begin
    A = [0.8 0.3; 0.2 0.7]
    B = reshape([0.7 0.2; 0.3 0.8], 2, 2, 1)
    y = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
    hierarchical = infer(
        model=Models.hierarchical_pomdp_model(
            A=A, B=B, A_ctx=[0.8 0.2; 0.2 0.8], D_slow=[0.6, 0.4], u=[1, 1], T=3,
        ),
        data=(y=y,), constraints=Models.hierarchical_constraints(),
        initialization=Models.hierarchical_initialization(2, 2),
        iterations=5, returnvars=KeepLast(), free_energy=true,
    )
    joint_A = cat(A, reverse(A; dims=2); dims=3)
    factored = infer(
        model=Models.factored_pomdp_model(
            A_m0=joint_A, A_m1=A, B_f0=B, B_f1=Matrix{Float64}(I, 2, 2),
            D_f0=[0.6, 0.4], D_f1=[0.5, 0.5], u=[1, 1], T=3,
        ),
        data=(y_m0=y, y_m1=y,), constraints=Models.factored_constraints(),
        initialization=Models.factored_initialization(2, 2),
        iterations=5, returnvars=KeepLast(), free_energy=true,
    )
    prior_A = [3.0 1.0; 1.0 3.0]
    learning = infer(
        model=Models.learning_pomdp_model(dirichlet_A=prior_A, B=B, D=[0.6, 0.4], u=[1, 1], T=3),
        data=(y=y,), constraints=Models.learning_constraints(),
        initialization=Models.learning_initialization(prior_A, 2),
        iterations=5, returnvars=KeepLast(), free_energy=true,
    )
    for result in (hierarchical, factored, learning)
        @test length(result.free_energy) == 5
        @test all(isfinite, result.free_energy)
    end
    @test length(hierarchical.posteriors[:s]) == 3
    @test length(factored.posteriors[:s1]) == length(factored.posteriors[:s2]) == 3
    @test size(mean(learning.posteriors[:A])) == (2, 2)
end
