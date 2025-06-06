# Multi-agent Trajectory Planning Configuration

[model]

[priors]

[visualization]

# Environment definitions
[environments.door]
description = "Two parallel walls with a gap between them"

[[environments.door.obstacles]]
center = [-40.0, 0.0]
size = 
center = 
size = [70.0, 5.0]

[[environments.door.obstacles]]
center = [40.0, 0.0]
size = 
center = 
size = [70.0, 5.0]

[environments.wall]
description = "A single wall obstacle in the center"

[[environments.wall.obstacles]]
center = [0.0, 0.0]
size = 
center = 
size = [10.0, 5.0]

[environments.combined]
description = "A combination of walls and obstacles"

[[environments.combined.obstacles]]
center = [-50.0, 0.0]
size = 
center = 
size = [70.0, 2.0]

[[environments.combined.obstacles]]
center = [50.0, 0.0]
size = 
center = 
size = [70.0, 2.0]

[[environments.combined.obstacles]]
center = [5.0, -1.0]
size = 
center = 
size = [3.0, 10.0]

# Agent configurations
[experiments]