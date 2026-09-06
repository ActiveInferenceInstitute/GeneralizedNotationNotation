module HalfspaceNode

using RxInfer
using LinearAlgebra

export Halfspace

"""
    Halfspace

A custom RxInfer node implementing a probabilistic constraint that enforces a variable to be 
greater than a threshold.

## Mathematical Background

The Halfspace node implements the soft constraint `out > a`, where:
- `out`: Output variable (typically a distance to obstacle or between agents)
- `a`: Threshold value (typically 0, representing the boundary of the constraint)
- `σ2`: Variance parameter that controls the softness of the constraint
- `γ`: Scaling parameter that affects the constraint strength

## Usage in Trajectory Planning

In the trajectory planning model, Halfspace nodes are used for two key purposes:

1. **Obstacle Avoidance**: Ensuring distances to obstacles remain positive
   ```julia
   z[k, t] ~ g(environment, rs[k], path[k, t])  # Distance to obstacles
   z[k, t] ~ Halfspace(0, zσ2[k, t], γ)         # Constraint: distance > 0
   ```

2. **Agent-Agent Collision Avoidance**: Ensuring distances between agents remain positive
   ```julia
   d[t] ~ h(environment, rs, path[1, t], ...)   # Minimum distance between agents
   d[t] ~ Halfspace(0, dσ2[t], γ)               # Constraint: distance > 0
   ```

## Message Passing

The node implements two message passing rules:
1. Forward message (`out`): Creates a distribution shifted away from the constraint boundary
2. Variance message (`σ2`): Adaptively adjusts constraint softness based on proximity to violation

This adaptive behavior is key to effective trajectory planning, as it allows:
- Strong enforcement of constraints when they're close to being violated
- Softer constraints when the system is far from constraint boundaries
"""
struct Halfspace end

@node Halfspace Stochastic [out, a, σ2, γ]

"""
    Halfspace(:out, Marginalisation) (q_a, q_σ2, q_γ)

Rule for computing the outgoing message for the output variable.

This rule creates a Normal distribution with:
- Mean: `mean(q_a) + mean(q_γ) * mean(q_σ2)`
- Variance: `mean(q_σ2)`

This effectively pushes the output distribution away from the constraint boundary
(represented by `a`) by an amount proportional to the constraint strength (`γ`) 
and softness (`σ2`).
"""
@rule Halfspace(:out, Marginalisation) (q_a::Any, q_σ2::Any, q_γ::Any) = begin
    return NormalMeanVariance(mean(q_a) + mean(q_γ) * mean(q_σ2), mean(q_σ2))
end

"""
    Halfspace(:σ2, Marginalisation) (q_out, q_a, q_γ)

Rule for computing the variance parameter of the constraint.

This rule adaptively adjusts the constraint softness based on:
- The current distance from the output to the constraint boundary
- The variance of the output distribution

The variance increases with:
- Increased distance from the constraint boundary
- Increased uncertainty in the output variable

This adaptive behavior allows constraints to be enforced more strongly when
they are close to being violated.
"""
@rule Halfspace(:σ2, Marginalisation) (q_out::Any, q_a::Any, q_γ::Any, ) = begin
    # `BayesBase.TerminalProdArgument` is used to ensure that the result of the posterior computation is equal to this value
    return BayesBase.TerminalProdArgument(PointMass( 1 / mean(q_γ) * sqrt(abs2(mean(q_out) - mean(q_a)) + var(q_out))))
end

end # module 