"""
Template Step Module - Architectural Pattern Reference

This module serves as the reference implementation for the GNN pipeline's architectural pattern.
It demonstrates the complete flow from main.py through numbered scripts as thin orchestrators 
to modular scripts in dedicated folders.

Architectural Pattern:
    main.py → Numbered Scripts (Thin Orchestrators) → Modular Scripts in Folders

This module provides the core template processing functionality that is called by
src/0_template.py (the thin orchestrator) which is executed by src/main.py (the pipeline orchestrator).

For complete documentation of this architectural pattern, see:
- src/template/README.md: Comprehensive architectural documentation
- src/0_template.py: Example thin orchestrator implementation
- src/main.py: Main pipeline orchestrator
"""

__version__ = "1.0.0"
__author__ = "GNN Pipeline Team"
__description__ = "Standardized template for GNN pipeline steps"

# Export main functionality
from .processor import (
    process_template_standardized,
    process_single_file,
    validate_file,
    generate_correlation_id,
    safe_template_execution,
    demonstrate_utility_patterns
)

# Version information
VERSION_INFO = {
    "version": __version__,
    "name": "Template Step",
    "description": __description__,
    "author": __author__
}

def get_version_info():
    """Get version information for the template step."""
    return VERSION_INFO 