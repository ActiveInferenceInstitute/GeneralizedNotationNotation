# This file makes src a package
"""
GNN Processing Pipeline - Source Package

This package contains the complete GNN (Generalized Notation Notation) processing pipeline
with integrated steps for parsing, validation, export, visualization, analysis, audio generation, and more!.
"""

__version__ = "1.1.0"
__author__ = "GNN Pipeline Team (@docxology, Active Inference Institute)"
__description__ = "Generalized Notation Notation Processing Pipeline"

# Import all modules for proper access
try:
    import gnn
    import export
    import render
    import website
    import audio
    import ontology
    import type_checker
    import visualization
    import execute
    import llm
    import mcp
    import setup
    import utils
    import pipeline
    import tests
    import validation
except ImportError as e:
    # Handle missing modules gracefully
    pass

# Create aliases for backward compatibility
try:
    # Alias audio as sapf for backward compatibility
    sapf = audio
except NameError:
    # If audio module is not available, create a placeholder
    class AudioPlaceholder:
        """Placeholder for audio module when not available."""
        pass
    sapf = AudioPlaceholder()

# Export all modules
__all__ = [
    'gnn',
    'export', 
    'render',
    'website',
    'audio',
    'sapf',  # Alias for audio
    'ontology',
    'type_checker',
    'visualization',
    'execute',
    'llm',
    'mcp',
    'setup',
    'utils',
    'pipeline',
    'tests',
    'validation'
] 