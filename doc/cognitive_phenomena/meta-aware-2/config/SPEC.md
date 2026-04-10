# Specification: Config

## Design Requirements
This module (`config`) maps structural logic to the overall execution graph.
It ensures that `Config` tasks resolve without runtime dependency loops.

## Components
Expected available types: GNNConfigParser, LevelConfig, ModelConfig, _TomlFallback
