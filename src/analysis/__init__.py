"""
Analysis module for GNN Processing Pipeline.

This module provides comprehensive analysis and statistical processing for GNN models.
"""

from .processor import process_analysis, convert_numpy_types
from .analyzer import (
    perform_statistical_analysis,
    extract_variables_for_analysis,
    extract_connections_for_analysis,
    extract_sections_for_analysis,
    calculate_variable_statistics,
    calculate_connection_statistics,
    calculate_section_statistics,
    count_type_distribution,
    build_connectivity_matrix,
    analyze_distributions,
    calculate_correlations,
    calculate_cyclomatic_complexity,
    calculate_cognitive_complexity,
    calculate_structural_complexity,
    calculate_complexity_metrics,
    calculate_maintainability_index,
    calculate_technical_debt,
    run_performance_benchmarks,
    perform_model_comparisons,
    generate_analysis_summary
)


def process_analysis(target_dir, output_dir, verbose=False, logger=None, **kwargs):
    """
    Main processing function for analysis.
    
    Args:
        target_dir: Directory containing files to process
        output_dir: Output directory for results
        verbose: Whether to enable verbose logging
        logger: Logger instance
        **kwargs: Additional processing options
        
    Returns:
        True if processing succeeded, False otherwise
    """
    import logging
    import json
    from pathlib import Path
    from datetime import datetime
    
    if logger is None:
        logger = logging.getLogger(__name__)
        if verbose:
            logger.setLevel(logging.DEBUG)
    
    try:
        # Ensure output directory exists
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Processing analysis for files in {target_dir}")
        
        # Check analysis dependencies
        analysis_tools = check_analysis_tools()
        
        # Create processing summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "target_dir": str(target_dir),
            "output_dir": str(output_dir),
            "analysis_tools": analysis_tools,
            "processing_status": "completed",
            "tools_available": [tool for tool, info in analysis_tools.items() if info.get('available')],
            "message": "Analysis module ready for statistical and complexity analysis"
        }
        
        # Save summary
        summary_file = output_dir / "analysis_processing_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"📊 Analysis processing summary saved to: {summary_file}")
        
        # Save tool details
        tools_file = output_dir / "analysis_tools_status.json"
        with open(tools_file, 'w') as f:
            json.dump(analysis_tools, f, indent=2)
        logger.info(f"🔧 Analysis tools status saved to: {tools_file}")
        
        logger.info(f"✅ Analysis processing completed")
        return True
    except Exception as e:
        logger.error(f"❌ Analysis processing failed: {e}")
        return False

def check_analysis_tools():
    """Check availability of analysis tools."""
    tools = {}
    
    # Check numpy
    try:
        import numpy
        tools['numpy'] = {
            'available': True,
            'version': numpy.__version__
        }
    except ImportError:
        tools['numpy'] = {'available': False, 'version': None}
    
    # Check pandas
    try:
        import pandas
        tools['pandas'] = {
            'available': True,
            'version': pandas.__version__
        }
    except ImportError:
        tools['pandas'] = {'available': False, 'version': None}
    
    # Check scipy
    try:
        import scipy
        tools['scipy'] = {
            'available': True,
            'version': scipy.__version__
        }
    except ImportError:
        tools['scipy'] = {'available': False, 'version': None}
    
    # Check matplotlib
    try:
        import matplotlib
        tools['matplotlib'] = {
            'available': True,
            'version': matplotlib.__version__
        }
    except ImportError:
        tools['matplotlib'] = {'available': False, 'version': None}
    
    return tools


__all__ = [
    'process_analysis',
    'convert_numpy_types',
    'perform_statistical_analysis',
    'extract_variables_for_analysis',
    'extract_connections_for_analysis',
    'extract_sections_for_analysis',
    'calculate_variable_statistics',
    'calculate_connection_statistics',
    'calculate_section_statistics',
    'count_type_distribution',
    'build_connectivity_matrix',
    'analyze_distributions',
    'calculate_correlations',
    'calculate_cyclomatic_complexity',
    'calculate_cognitive_complexity',
    'calculate_structural_complexity',
    'calculate_complexity_metrics',
    'calculate_maintainability_index',
    'calculate_technical_debt',
    'run_performance_benchmarks',
    'perform_model_comparisons',
    'generate_analysis_summary'
]
