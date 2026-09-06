module DistanceFunctions

using LinearAlgebra
using LogExpFunctions
using ..Environment

export distance, g, h, softmin, configure_softmin

# Global model parameters that can be updated from configuration
SOFTMIN_TEMPERATURE = 10.0

# Configurable softmin function
softmin(x; l=SOFTMIN_TEMPERATURE) = -logsumexp(-l .* x) / l

# Update softmin temperature from configuration
function configure_softmin(temperature::Number)
    global SOFTMIN_TEMPERATURE = temperature
end

# Calculate distance from a point to a rectangle
function distance(r::Rectangle, state)
    if abs(state[1] - r.center[1]) > r.size[1] / 2 || abs(state[2] - r.center[2]) > r.size[2] / 2
        # outside of rectangle
        dx = max(abs(state[1] - r.center[1]) - r.size[1] / 2, 0)
        dy = max(abs(state[2] - r.center[2]) - r.size[2] / 2, 0)
        return sqrt(dx^2 + dy^2)
    else
        # inside rectangle
        return max(abs(state[1] - r.center[1]) - r.size[1] / 2, abs(state[2] - r.center[2]) - r.size[2] / 2)
    end
end

# Calculate distance from a point to an environment (minimum distance to any obstacle)
function distance(env::Environment, state)
    return softmin([distance(obstacle, state) for obstacle in env.obstacles])
end

# Helper function, distance with radius offset
function g(environment, radius, state)
    return distance(environment, state) - radius
end

# Helper function, finds minimum distances between agents pairwise
function h(environment, radiuses, states...)
    # Calculate pairwise distances between all agents
    distances = Real[]
    n = length(states)

    for i in 1:n
        for j in (i+1):n
            push!(distances, norm(states[i] - states[j]) - radiuses[i] - radiuses[j])
        end
    end

    return softmin(distances)
end

end # module 