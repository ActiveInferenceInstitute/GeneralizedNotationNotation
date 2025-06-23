"""Configuration module for meta-aware-2 simulations."""

from .gnn_parser import load_gnn_config, ModelConfig, LevelConfig, GNNConfigParser

__all__ = ['load_gnn_config', 'ModelConfig', 'LevelConfig', 'GNNConfigParser'] 