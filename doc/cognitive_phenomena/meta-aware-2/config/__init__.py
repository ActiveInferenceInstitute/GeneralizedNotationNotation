"""Configuration module for meta-aware-2 simulations."""

from .gnn_parser import GNNConfigParser, LevelConfig, ModelConfig, load_gnn_config

__all__ = ["load_gnn_config", "ModelConfig", "LevelConfig", "GNNConfigParser"]
