#!/usr/bin/env python3
"""
Core module for GNN pipeline.
"""

def get_module_info() -> dict[str, object]:
    """Get information about the core module."""
    return {
        "name": "GNN Pipeline Core",
        "version": "1.0.0",
        "description": "Core functionality for GNN processing pipeline",
        "modules": [
            "pipeline",
            "gnn",
            "analysis",
            "llm",
            "render",
            "website",
            "security",
            "advanced_visualization"
        ],
        "features": [
            "Pipeline orchestration",
            "GNN processing",
            "Analysis and statistics",
            "LLM integration",
            "Code generation",
            "Website generation",
            "Security validation",
            "Advanced visualization"
        ]
    }
