# Multi-Agent Coordination Acceptance Fixture

## ModelName
Multi-Agent Coordination Acceptance Fixture

## ModelAnnotation
Compact fixture for RxInfer and DisCoPy roadmap acceptance tests.

## StateSpaceBlock
s[2,1,type=categorical]
o[2,1,type=categorical]
u[2,1,type=categorical]

## Connections
s > o
s > s
u > s

## InitialParameterization
nr_agents=3
agent_ids=[1,2,3]
agent_initial_positions=[[0.0,0.0],[1.0,0.0],[0.0,1.0]]
agent_target_positions=[[2.0,2.0],[3.0,2.0],[2.0,3.0]]
agent_radii=[1.0,1.0,1.0]
agent_edges=[[1,2],[2,3]]
agent_clusters=[{"name":"left","agent_ids":[1,2]},{"name":"right","agent_ids":[3]}]
message_passing=clustered_mean_field
A={(0.9,0.1),(0.1,0.9)}
B={((0.9,0.1),(0.1,0.9)),((0.1,0.9),(0.9,0.1))}
C={(1.0,0.0)}
D={(0.5,0.5)}

## Time
Dynamic

## ActInfOntologyAnnotation
s=HiddenState
o=Observation
u=Action
