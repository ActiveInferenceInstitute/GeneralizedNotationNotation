# Native bounded four-agent inference check; no plots or experiment outputs.
using RxInfer, LinearAlgebra, StableRNGs, Test
include(joinpath(@__DIR__, "TrajectoryPlanning.jl"))
const TP = TrajectoryPlanning
agents = TP.create_standard_agents()
goals = hcat([[collect((a.initial_position[1], 0.0, a.initial_position[2], 0.0)), collect((a.target_position[1], 0.0, a.target_position[2], 0.0))] for a in agents]...)
rng = StableRNG(42)
init = @initialization begin
    q(dσ2) = repeat([PointMass(1.0)], 3)
    q(zσ2) = repeat([PointMass(1.0)], 4, 3)
    q(control) = repeat([PointMass(0.0)], 3)
    μ(state) = MvNormalMeanCovariance(randn(rng, 4), 100I)
    μ(path) = MvNormalMeanCovariance(randn(rng, 2), 100I)
end
algorithm = @algorithm begin
    TP.h() -> Linearization()
    TP.g() -> Linearization()
end
result = infer(
    model=TP.path_planning_model(environment=TP.create_door_environment(), agents=agents, nr_steps=3),
    data=(goals=goals,), initialization=init, constraints=TP.path_planning_constraints(),
    algorithm=algorithm, iterations=3, returnvars=KeepLast(),
)
@test size(result.posteriors[:path]) == (4, 3)
@test all(q -> all(isfinite, mean(q)), result.posteriors[:path])
println("Four-agent trajectory native smoke passed")
