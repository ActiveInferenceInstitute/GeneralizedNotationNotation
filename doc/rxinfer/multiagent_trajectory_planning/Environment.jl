module Environment

using Plots: Shape

export Rectangle, Environment, Agent
export create_door_environment, create_wall_environment, create_combined_environment, create_standard_agents

# A simple struct to represent a rectangle, which is defined by its center (x, y) and size (width, height)
Base.@kwdef struct Rectangle
    center::Tuple{Float64, Float64}
    size::Tuple{Float64, Float64}
end

# A simple struct to represent an environment, which is defined by a list of obstacles,
# and in this demo the obstacles are just rectangles
Base.@kwdef struct Environment
    obstacles::Vector{Rectangle}
end

# Agent plan, encodes start and goal states
Base.@kwdef struct Agent
    radius::Float64
    initial_position::Tuple{Float64, Float64}
    target_position::Tuple{Float64, Float64}
end

# Create standard environments
function create_door_environment()
    return Environment(obstacles = [
        Rectangle(center = (-40, 0), size = (70, 5)),
        Rectangle(center = (40, 0), size = (70, 5))
    ])
end

function create_wall_environment()
    return Environment(obstacles = [
        Rectangle(center = (0, 0), size = (10, 5))
    ])
end

function create_combined_environment()
    return Environment(obstacles = [
        Rectangle(center = (-50, 0), size = (70, 2)),
        Rectangle(center = (50, -0), size = (70, 2)),
        Rectangle(center = (5, -1), size = (3, 10))
    ])
end

# Create standard agents
function create_standard_agents()
    return [
        Agent(radius = 2.5, initial_position = (-4, 10), target_position = (-10, -10)),
        Agent(radius = 1.5, initial_position = (-10, 5), target_position = (10, -15)),
        Agent(radius = 1.0, initial_position = (-15, -10), target_position = (10, 10)),
        Agent(radius = 2.5, initial_position = (0, -10), target_position = (-10, 15))
    ]
end

end # module 