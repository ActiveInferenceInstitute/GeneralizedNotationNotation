# Type System Implementations

This directory contains type system implementations of GNN models:

- `haskell.hs` - Haskell strongly-typed implementation
- `scala.scala` - Scala categorical implementation
- `categorical.scala` - Categorical structures in Scala
- `mapping.md` - Mapping between GNN model elements and type system representations
- `examples/` - Example implementations of concrete GNN models

## Recent Enhancements

The type systems have been updated to better align with concrete GNN models:

1. Added explicit `HabitVector` (E) representation in both Scala and Haskell implementations
2. Created a mapping document to show how each element in concrete models maps to the type systems
3. Added a POMDP example that demonstrates how to represent the `actinf_pomdp_agent.md` model using the type systems

## Coverage Note

The current implementations provide strong theoretical foundations for Active Inference models but still don't fully capture all elements of concrete GNN models like the one in `src/gnn/gnn_examples/actinf_pomdp_agent.md`. Remaining gaps include:

1. The connection syntax between variables is implied rather than explicit in its full detail
2. Initial parameterization values are not directly represented in a standardized way
3. Time representation needs enhancement to better match the GNN specification

Future extensions will address these remaining gaps to ensure complete coverage of all GNN model features.
