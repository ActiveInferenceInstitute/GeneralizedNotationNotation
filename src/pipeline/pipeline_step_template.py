#!/usr/bin/env python3
"""
Standardized Pipeline Step Template

This template provides a consistent structure for all GNN pipeline steps.
Copy this template and modify the CUSTOMIZE sections to create new pipeline steps.

The GNN pipeline consists of 24 steps (0-23):

    Steps 0-9 (Core):
        0: Template initialization    5: Type checker
        1: Environment setup          6: Validation
        2: Test execution             7: Export
        3: GNN parsing                8: Visualization
        4: Model registry             9: Advanced visualization

    Steps 10-16 (Simulation):
        10: Ontology                  14: ML integration
        11: Render (code generation)  15: Audio processing
        12: Execute (simulation)      16: Analysis
        13: LLM analysis

    Steps 17-23 (Output):
        17: Integration               21: MCP (Model Context Protocol)
        18: Security                  22: GUI
        19: Research                  23: Report generation
        20: Website

Usage:
    python X_step_name.py [options]
    (Typically called by main.py)

To create a new step:
    1. Copy this file to src/N_stepname.py (where N is the step number)
    2. Search for "CUSTOMIZE" and update each marked section
    3. Create corresponding src/stepname/ module directory
    4. Add step to main.py orchestration
"""

import sys
from pathlib import Path
from typing import Optional, List, Any

# Standard imports for all pipeline steps
from utils import (
    setup_step_logging,
    log_step_start,
    log_step_success, 
    log_step_warning,
    log_step_error,
    performance_tracker,
    UTILS_AVAILABLE
)

from pipeline import (
    get_output_dir_for_script,
    get_pipeline_config
)

import datetime
import json
import yaml

# Initialize logger for this step
# CUSTOMIZE: Replace "X_step_name" with your step name (e.g., "3_gnn", "11_render")
logger = setup_step_logging("X_step_name", verbose=False)

# Step-specific module imports
# CUSTOMIZE: Import the modules your step requires. Examples:
#   from gnn import parse_gnn_file, discover_gnn_files
#   from render.pymdp import PyMDPRenderer
#   from execute import run_simulation
try:
    # Example imports for common step types:
    # - Parsing steps: from gnn.parsers import MarkdownGNNParser
    # - Render steps: from render.{framework} import {Framework}Renderer
    # - Execute steps: from execute.{framework} import {Framework}Runner
    # - Analysis steps: from analysis import run_analysis
    pass

    DEPENDENCIES_AVAILABLE = True
    logger.debug("Successfully imported step-specific dependencies")

except ImportError as e:
    log_step_warning(logger, f"Failed to import step-specific modules: {e}")
    DEPENDENCIES_AVAILABLE = False

def validate_step_requirements() -> bool:
    """
    Validate that all requirements for this step are met.
    
    Returns:
        True if step can proceed, False otherwise
    """
    if not DEPENDENCIES_AVAILABLE:
        log_step_error(logger, "Required dependencies are not available")
        return False

    # CUSTOMIZE: Add validation checks specific to your step. Examples:
    #
    # Check required files exist:
    #   required_file = Path("config/step_config.yaml")
    #   if not required_file.exists():
    #       log_step_error(logger, f"Required config not found: {required_file}")
    #       return False
    #
    # Validate environment variables:
    #   api_key = os.environ.get("LLM_API_KEY")
    #   if not api_key:
    #       log_step_error(logger, "LLM_API_KEY environment variable not set")
    #       return False
    #
    # Test external service connections:
    #   try:
    #       response = requests.get("http://localhost:8080/health", timeout=5)
    #       if response.status_code != 200:
    #           return False
    #   except requests.RequestException:
    #       log_step_error(logger, "Cannot connect to required service")
    #       return False

    return True

def process_single_file(
    input_file: Path, 
    output_dir: Path, 
    options: dict
) -> bool:
    """
    Process a single input file with comprehensive analysis and transformation.
    
    Args:
        input_file: Path to the input file
        output_dir: Directory for outputs
        options: Processing options from arguments
        
    Returns:
        True if processing succeeded, False otherwise
    """
    logger.debug(f"Processing file: {input_file}")
    
    try:
        # Read and analyze file content
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Perform comprehensive file analysis
        analysis_results = {
            'file_name': input_file.name,
            'file_size_bytes': len(content),
            'file_size_lines': len(content.splitlines()),
            'file_extension': input_file.suffix,
            'content_type': _detect_content_type(content),
            'processing_timestamp': datetime.datetime.now().isoformat(),
            'processing_options': options
        }
        
        # Extract metadata based on file type
        if input_file.suffix.lower() == '.md':
            analysis_results['metadata'] = _extract_markdown_metadata(content)
        elif input_file.suffix.lower() in ['.json', '.yaml', '.yml']:
            analysis_results['metadata'] = _extract_structured_metadata(content, input_file.suffix)
        else:
            analysis_results['metadata'] = _extract_generic_metadata(content)
        
        # Generate comprehensive output files
        base_name = input_file.stem
        
        # 1. Analysis report (JSON)
        analysis_file = output_dir / f"{base_name}_analysis.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # 2. Processed content with annotations
        processed_file = output_dir / f"{base_name}_processed{input_file.suffix}"
        processed_content = _generate_processed_content(content, analysis_results)
        with open(processed_file, 'w', encoding='utf-8') as f:
            f.write(processed_content)
        
        # 3. Summary report (Markdown)
        summary_file = output_dir / f"{base_name}_summary.md"
        summary_content = _generate_summary_report(analysis_results)
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        logger.debug(f"Generated outputs: {analysis_file}, {processed_file}, {summary_file}")
        return True
        
    except Exception as e:
        log_step_error(logger, f"Failed to process {input_file}: {e}")
        return False

def _detect_content_type(content: str) -> str:
    """Detect the type of content in the file."""
    if content.strip().startswith('{') or content.strip().startswith('['):
        return 'json'
    elif '---' in content[:100] and '\n' in content[:100]:
        return 'yaml'
    elif content.strip().startswith('#'):
        return 'markdown'
    elif any(keyword in content.lower() for keyword in ['gnn', 'statespaceblock', 'connections']):
        return 'gnn_specification'
    else:
        return 'text'

def _extract_markdown_metadata(content: str) -> dict:
    """Extract metadata from markdown content."""
    metadata = {}
    lines = content.splitlines()
    
    # Extract headers
    headers = [line.strip('# ') for line in lines if line.strip().startswith('#')]
    metadata['headers'] = headers[:10]  # First 10 headers
    
    # Extract code blocks
    code_blocks = []
    in_code_block = False
    current_block = []
    
    for line in lines:
        if line.strip().startswith('```'):
            if in_code_block:
                code_blocks.append('\n'.join(current_block))
                current_block = []
            in_code_block = not in_code_block
        elif in_code_block:
            current_block.append(line)
    
    metadata['code_blocks_count'] = len(code_blocks)
    metadata['total_lines'] = len(lines)
    
    return metadata

def _extract_structured_metadata(content: str, file_type: str) -> dict:
    """Extract metadata from structured files (JSON, YAML)."""
    try:
        if file_type.lower() == '.json':
            data = json.loads(content)
        else:  # YAML
            import yaml
            data = yaml.safe_load(content)
        
        metadata = {
            'structure_type': 'structured',
            'top_level_keys': list(data.keys()) if isinstance(data, dict) else ['array'],
            'data_type': type(data).__name__,
            'is_valid': True
        }
        
        if isinstance(data, dict):
            metadata['key_count'] = len(data)
            metadata['nested_structure'] = _analyze_nested_structure(data)
        
        return metadata
    except Exception as e:
        return {
            'structure_type': 'structured',
            'is_valid': False,
            'error': str(e)
        }

def _extract_generic_metadata(content: str) -> dict:
    """Extract metadata from generic text content."""
    lines = content.splitlines()
    words = content.split()
    
    return {
        'structure_type': 'text',
        'line_count': len(lines),
        'word_count': len(words),
        'character_count': len(content),
        'non_empty_lines': len([line for line in lines if line.strip()]),
        'average_line_length': sum(len(line) for line in lines) / max(len(lines), 1)
    }

def _analyze_nested_structure(data: dict, max_depth: int = 3) -> dict:
    """Analyze the nested structure of a dictionary."""
    def _analyze_level(obj, depth=0):
        if depth >= max_depth:
            return {'type': 'max_depth_reached'}
        
        if isinstance(obj, dict):
            return {
                'type': 'dict',
                'keys': list(obj.keys()),
                'key_count': len(obj),
                'sample_values': {k: type(v).__name__ for k, v in list(obj.items())[:5]}
            }
        elif isinstance(obj, list):
            return {
                'type': 'list',
                'length': len(obj),
                'sample_types': [type(item).__name__ for item in obj[:5]]
            }
        else:
            return {'type': type(obj).__name__}
    
    return _analyze_level(data)

def _generate_processed_content(content: str, analysis: dict) -> str:
    """Generate processed content with annotations."""
    processed_lines = []
    
    # Add processing header
    processed_lines.append(f"# Processed File: {analysis['file_name']}")
    processed_lines.append(f"# Processing Timestamp: {analysis['processing_timestamp']}")
    processed_lines.append(f"# Content Type: {analysis['content_type']}")
    processed_lines.append(f"# File Size: {analysis['file_size_bytes']} bytes, {analysis['file_size_lines']} lines")
    processed_lines.append("")
    
    # Add original content with line numbers
    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
        processed_lines.append(f"{i:4d}: {line}")
    
    return '\n'.join(processed_lines)

def _generate_summary_report(analysis: dict) -> str:
    """Generate a comprehensive summary report in Markdown format."""
    report_lines = []
    
    report_lines.append(f"# File Processing Summary")
    report_lines.append("")
    report_lines.append(f"**File:** {analysis['file_name']}")
    report_lines.append(f"**Processed:** {analysis['processing_timestamp']}")
    report_lines.append(f"**Content Type:** {analysis['content_type']}")
    report_lines.append("")
    
    report_lines.append("## File Statistics")
    report_lines.append(f"- **Size:** {analysis['file_size_bytes']} bytes")
    report_lines.append(f"- **Lines:** {analysis['file_size_lines']}")
    report_lines.append(f"- **Extension:** {analysis['file_extension']}")
    report_lines.append("")
    
    if 'metadata' in analysis:
        metadata = analysis['metadata']
        report_lines.append("## Content Analysis")
        
        if metadata.get('structure_type') == 'structured':
            report_lines.append(f"- **Structure:** {metadata.get('structure_type', 'Unknown')}")
            report_lines.append(f"- **Valid:** {metadata.get('is_valid', 'Unknown')}")
            if 'top_level_keys' in metadata:
                report_lines.append(f"- **Top-level Keys:** {', '.join(metadata['top_level_keys'][:10])}")
        elif metadata.get('structure_type') == 'text':
            report_lines.append(f"- **Words:** {metadata.get('word_count', 0)}")
            report_lines.append(f"- **Characters:** {metadata.get('character_count', 0)}")
            report_lines.append(f"- **Non-empty Lines:** {metadata.get('non_empty_lines', 0)}")
            report_lines.append(f"- **Average Line Length:** {metadata.get('average_line_length', 0):.1f}")
        elif 'headers' in metadata:
            report_lines.append(f"- **Headers Found:** {len(metadata.get('headers', []))}")
            report_lines.append(f"- **Code Blocks:** {metadata.get('code_blocks_count', 0)}")
    
    report_lines.append("")
    report_lines.append("## Processing Options")
    for key, value in analysis.get('processing_options', {}).items():
        report_lines.append(f"- **{key}:** {value}")
    
    return '\n'.join(report_lines)

def main(parsed_args) -> int:
    """
    Main function for the pipeline step.
    
    Args:
        parsed_args: Parsed command line arguments
        
    Returns:
        Exit code (0=success, 1=error, 2=warnings)
    """
    
    # CUSTOMIZE: Update the step description for your specific step
    # Example: "Starting GNN parsing step" or "Starting PyMDP render step"
    log_step_start(logger, "Starting standardized pipeline step")

    # Update logger verbosity based on arguments
    if getattr(parsed_args, 'verbose', False):
        import logging
        logger.setLevel(logging.DEBUG)

    # Validate step requirements
    if not validate_step_requirements():
        log_step_error(logger, "Step requirements not met")
        return 1

    # Get configuration
    config = get_pipeline_config()
    # CUSTOMIZE: Replace "X_step_name.py" with your script name (e.g., "3_gnn.py")
    step_config = config.get_step_config("X_step_name.py")

    # Set up paths
    input_dir = getattr(parsed_args, 'target_dir', Path("input/gnn_files"))
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)

    output_dir = Path(getattr(parsed_args, 'output_dir', 'output'))
    # CUSTOMIZE: Replace "X_step_name.py" with your script name (e.g., "3_gnn.py")
    step_output_dir = get_output_dir_for_script("X_step_name.py", output_dir)
    step_output_dir.mkdir(parents=True, exist_ok=True)

    # Get processing options
    recursive = getattr(parsed_args, 'recursive', True)
    verbose = getattr(parsed_args, 'verbose', False)

    # CUSTOMIZE: Extract additional arguments specific to your step. Examples:
    #   strict_mode = getattr(parsed_args, 'strict', False)  # For validation steps
    #   timeout = getattr(parsed_args, 'timeout', 300)  # For execution steps
    #   frameworks = getattr(parsed_args, 'frameworks', 'all')  # For render/execute
    #   llm_provider = getattr(parsed_args, 'llm_provider', 'ollama')  # For LLM steps
    #   audio_format = getattr(parsed_args, 'audio_format', 'wav')  # For audio steps
    
    logger.info(f"Processing files from: {input_dir}")
    logger.info(f"Recursive processing: {'enabled' if recursive else 'disabled'}")
    logger.info(f"Output directory: {step_output_dir}")
    
    # Validate input directory
    if not input_dir.exists():
        log_step_error(logger, f"Input directory does not exist: {input_dir}")
        return 1
    
    # Find input files
    # CUSTOMIZE: Update the pattern for your step's input file types. Examples:
    #   GNN steps: "**/*.md" or "**/*.gnn"
    #   JSON steps: "**/*.json"
    #   Python steps: "**/*.py"
    #   Julia steps: "**/*.jl"
    #   Multiple types: Use multiple globs and combine results
    pattern = "**/*.md" if recursive else "*.md"
    input_files = list(input_dir.glob(pattern))
    
    if not input_files:
        log_step_warning(logger, f"No input files found in {input_dir} using pattern '{pattern}'")
        return 2  # Warning exit code
    
    logger.info(f"Found {len(input_files)} files to process")
    
    # Process files with performance tracking
    successful_files = 0
    failed_files = 0
    
    processing_options = {
        'verbose': verbose,
        'recursive': recursive,
        # CUSTOMIZE: Add step-specific options to pass to process_single_file(). Examples:
        #   'strict_mode': strict_mode,
        #   'timeout': timeout,
        #   'frameworks': frameworks.split(',') if isinstance(frameworks, str) else frameworks,
        #   'output_format': output_format,
    }
    
    with performance_tracker.track_operation("process_all_files"):
        for input_file in input_files:
            try:
                with performance_tracker.track_operation(f"process_{input_file.name}"):
                    success = process_single_file(
                        input_file, 
                        step_output_dir, 
                        processing_options
                    )
                
                if success:
                    successful_files += 1
                else:
                    failed_files += 1
                    
            except Exception as e:
                log_step_error(logger, f"Unexpected error processing {input_file}: {e}")
                failed_files += 1
    
    # Report results
    total_files = successful_files + failed_files
    logger.info(f"Processing complete: {successful_files}/{total_files} files successful")
    
    # Generate summary report with processing results
    # CUSTOMIZE: Add step-specific fields to the summary as needed
    summary_file = step_output_dir / "processing_summary.json"
    import json
    summary = {
        # CUSTOMIZE: Replace "X_step_name" with your step name (e.g., "3_gnn")
        "step_name": "X_step_name",
        "input_directory": str(input_dir),
        "output_directory": str(step_output_dir),
        "total_files": total_files,
        "successful_files": successful_files,
        "failed_files": failed_files,
        "processing_options": processing_options,
        "performance_summary": performance_tracker.get_summary(),
        # CUSTOMIZE: Add step-specific summary fields. Examples:
        #   "models_parsed": len(parsed_models),
        #   "frameworks_rendered": list(rendered_frameworks),
        #   "simulations_completed": simulation_count,
        #   "warnings": warning_messages,
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Summary report saved: {summary_file}")
    
    # Determine exit code
    if failed_files == 0:
        log_step_success(logger, "All files processed successfully")
        return 0
    elif successful_files > 0:
        log_step_warning(logger, f"Partial success: {failed_files} files failed")
        return 2  # Success with warnings
    else:
        log_step_error(logger, "All files failed to process")
        return 1

# Standardized execution using the template
if __name__ == '__main__':
    # CUSTOMIZE: List the module names your step depends on for import validation.
    # These are checked during startup to provide clear error messages.
    # Examples by step type:
    #   GNN parsing: ["gnn", "gnn.parsers"]
    #   Rendering: ["render", "render.pymdp", "jinja2"]
    #   Execution: ["execute", "execute.pymdp", "numpy"]
    #   Analysis: ["analysis", "pandas", "matplotlib"]
    #   LLM: ["llm", "openai"] or ["llm", "anthropic"]
    step_dependencies = [
        # "your_required_module",
        # "another_dependency"
    ]

    # CUSTOMIZE: Update step_name and step_description for your step.
    # step_name should match the filename (e.g., "3_gnn.py" for src/3_gnn.py)
    # step_description should briefly describe what the step does
    exit_code = execute_pipeline_step_template(
        step_name="X_step_name.py",
        step_description="Standardized pipeline step template",
        main_function=main,
        import_dependencies=step_dependencies
    )

    sys.exit(exit_code) 